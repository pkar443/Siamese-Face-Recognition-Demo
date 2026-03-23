import argparse
import random
import re
from datetime import datetime, timezone

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runtime import configure_device
from similarity import (
    cosine_similarities,
    default_metric_for_model_type,
    default_threshold_for,
    find_best_threshold,
)
from utils import (
    DEFAULT_EVAL_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAIN_DIR,
    MODEL_METRIC_COSINE,
    MODEL_METRIC_EUCLIDEAN,
    MODEL_METRIC_PROBABILITY,
    MODEL_TYPE_CONTRASTIVE,
    MODEL_TYPE_LEGACY,
    list_image_files,
    metadata_path_for_weights,
    read_model_metadata,
    resolve_path,
    write_model_metadata,
)

IMAGE_SHAPE = (100, 100)
RUN_NAME_SUFFIX_PATTERN = re.compile(r"_(legacy_classifier|contrastive_embedding)_\d+iter(?:_v\d+)?$")


def load_face_image(image_path, image_shape=IMAGE_SHAPE):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if image.shape != image_shape:
        image = cv2.resize(image, image_shape, interpolation=cv2.INTER_AREA)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, -1)


def build_person_index(path):
    dataset_path = resolve_path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    person_index = {}
    for person_dir in sorted(dataset_path.iterdir()):
        if not person_dir.is_dir():
            continue
        images = list_image_files(person_dir)
        if images:
            person_index[person_dir.name] = images

    if len(person_index) < 2:
        raise ValueError(
            f"Need at least 2 people with images in {dataset_path}, found {len(person_index)}."
        )
    return person_index


def summarize_people(path):
    dataset_path = resolve_path(path)
    if not dataset_path.exists():
        return {
            "path": str(dataset_path),
            "people": 0,
            "images": 0,
        }

    people = 0
    images = 0
    for person_dir in dataset_path.iterdir():
        if not person_dir.is_dir():
            continue
        image_files = list_image_files(person_dir)
        if image_files:
            people += 1
            images += len(image_files)
    return {
        "path": str(dataset_path),
        "people": people,
        "images": images,
    }


def get_minibatch(batch_size=32, prob=0.5, path="train", person_index=None):
    if person_index is None:
        person_index = build_person_index(path)

    left = []
    right = []
    targets = []
    people = list(person_index)
    positive_people = [person for person in people if len(person_index[person]) >= 2]

    if not positive_people:
        raise ValueError("Need at least one identity with 2 or more images for positive pairs.")

    for _ in range(batch_size):
        same_person = np.random.choice([0, 1], p=[1 - prob, prob]) == 1
        if same_person:
            person = random.choice(positive_people)
            image_one, image_two = random.sample(person_index[person], 2)
            targets.append(1.0)
        else:
            person_one, person_two = random.sample(people, 2)
            image_one = random.choice(person_index[person_one])
            image_two = random.choice(person_index[person_two])
            targets.append(0.0)

        left.append(load_face_image(image_one))
        right.append(load_face_image(image_two))

    return [np.asarray(left), np.asarray(right)], np.asarray(targets, dtype=np.float32)


def _predict_pair_metrics(model, inputs, model_type):
    if model_type == MODEL_TYPE_LEGACY:
        probabilities = np.atleast_1d(np.squeeze(model.predict(inputs, verbose=0)))
        return {MODEL_METRIC_PROBABILITY: probabilities}

    if model_type == MODEL_TYPE_CONTRASTIVE:
        distances = np.atleast_1d(np.squeeze(model.predict(inputs, verbose=0)))
        encoder = model.get_layer("shared_encoder")
        left_embeddings = encoder.predict(inputs[0], verbose=0)
        right_embeddings = encoder.predict(inputs[1], verbose=0)
        cosines = cosine_similarities(left_embeddings, right_embeddings)
        return {
            MODEL_METRIC_EUCLIDEAN: distances,
            MODEL_METRIC_COSINE: cosines,
        }

    raise ValueError(f"Unsupported model type: {model_type}")


def sample_pair_metrics(model, n_samples, path="eval", person_index=None, model_type=MODEL_TYPE_LEGACY):
    inputs, targets = get_minibatch(n_samples, path=path, person_index=person_index)
    metric_values = _predict_pair_metrics(model, inputs, model_type)
    return targets, metric_values


def evaluate_pair_metrics(model, n_samples, path="eval", person_index=None, model_type=MODEL_TYPE_LEGACY):
    targets, metric_values = sample_pair_metrics(
        model,
        n_samples=n_samples,
        path=path,
        person_index=person_index,
        model_type=model_type,
    )
    results = {}
    for metric_name, values in metric_values.items():
        threshold, accuracy = find_best_threshold(values, targets, metric_name)
        results[metric_name] = {
            "threshold": threshold,
            "accuracy": accuracy,
        }
    return results


def _artifact_base_name(save_path):
    save_path = resolve_path(save_path)
    name = save_path.name
    for suffix in (".weights.h5", ".h5"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return save_path.with_name(name)


def suggest_run_save_path(save_path, model_type, iterations, make_unique=True):
    save_path = resolve_path(save_path)
    base_path = _artifact_base_name(save_path)
    base_name = RUN_NAME_SUFFIX_PATTERN.sub("", base_path.name)
    candidate = save_path.with_name(f"{base_name}_{model_type}_{int(iterations)}iter.weights.h5")
    if not make_unique:
        return candidate

    if not candidate.exists() and not metadata_path_for_weights(candidate).exists():
        return candidate

    version = 2
    while True:
        candidate = save_path.with_name(f"{base_name}_{model_type}_{int(iterations)}iter_v{version}.weights.h5")
        if not candidate.exists() and not metadata_path_for_weights(candidate).exists():
            return candidate
        version += 1


def _save_line_plot(output_path, title, xlabel, ylabel, series):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, x_values, y_values in series:
        if x_values and y_values:
            ax.plot(x_values, y_values, marker="o", linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def _save_loss_curve(save_path, loss_steps, train_losses, val_losses):
    output_path = _artifact_base_name(save_path).with_name(f"{_artifact_base_name(save_path).name}_loss_curve.png")
    return _save_line_plot(
        output_path=output_path,
        title="Training and Validation Loss",
        xlabel="Iteration",
        ylabel="Loss",
        series=[
            ("Train loss", loss_steps, train_losses),
            ("Validation loss", loss_steps, val_losses),
        ],
    )


def _save_accuracy_curve(save_path, accuracy_steps, accuracy_values, metric_name):
    output_path = _artifact_base_name(save_path).with_name(f"{_artifact_base_name(save_path).name}_accuracy_curve.png")
    return _save_line_plot(
        output_path=output_path,
        title=f"Validation Accuracy ({metric_name})",
        xlabel="Iteration",
        ylabel="Accuracy (%)",
        series=[("Validation accuracy", accuracy_steps, accuracy_values)],
    )


def _save_distance_histogram(save_path, positive_distances, negative_distances):
    output_path = _artifact_base_name(save_path).with_name(
        f"{_artifact_base_name(save_path).name}_distance_histogram.png"
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = 20
    if len(positive_distances):
        ax.hist(positive_distances, bins=bins, alpha=0.65, label="Same identity", color="#2563eb")
    if len(negative_distances):
        ax.hist(negative_distances, bins=bins, alpha=0.65, label="Different identities", color="#dc2626")
    ax.set_title("Contrastive Pair Distance Histogram")
    ax.set_xlabel("Euclidean distance")
    ax.set_ylabel("Pair count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def test_oneshot(
    model,
    n_samples,
    verbose=0,
    path="eval",
    person_index=None,
    model_type=MODEL_TYPE_LEGACY,
    metric=None,
):
    metric = metric or default_metric_for_model_type(model_type)
    results = evaluate_pair_metrics(
        model,
        n_samples=n_samples,
        path=path,
        person_index=person_index,
        model_type=model_type,
    )
    chosen = results[metric]
    percent_correct = chosen["accuracy"]
    if verbose:
        print(
            f"Got an average of {percent_correct:.2f}% {n_samples} pair accuracy "
            f"with {metric} threshold {chosen['threshold']:.4f}"
        )
    return percent_correct


def _build_training_metadata(
    model_type,
    save_path,
    train_dir,
    eval_dir,
    iterations,
    batch_size,
    positive_prob,
    loss_every,
    eval_every,
    oneshot_n,
    seed,
    selected_device,
    visible_gpus,
    recommended_thresholds,
    best_accuracy,
    checkpoint_metric,
    contrastive_margin,
    embedding_dim,
):
    metadata = {
        "model_type": model_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "save_path": str(save_path),
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "iterations": int(iterations),
        "batch_size": int(batch_size),
        "positive_prob": float(positive_prob),
        "loss_every": int(loss_every),
        "eval_every": int(eval_every),
        "oneshot_n": int(oneshot_n),
        "seed": int(seed),
        "selected_device": selected_device,
        "visible_gpus": visible_gpus,
        "checkpoint_metric": checkpoint_metric,
        "best_accuracy": float(best_accuracy),
        "recommended_thresholds": {name: round(float(value), 6) for name, value in recommended_thresholds.items()},
        "input_shape": [IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1],
    }
    if model_type == MODEL_TYPE_LEGACY:
        metadata["training_objective"] = "binary_crossentropy_pair_classification"
    else:
        metadata["training_objective"] = "contrastive_loss"
        metadata["contrastive_margin"] = float(contrastive_margin)
        metadata["embedding_dim"] = int(embedding_dim)
    return metadata


def train_model(
    train_dir=DEFAULT_TRAIN_DIR,
    eval_dir=DEFAULT_EVAL_DIR,
    save_path=DEFAULT_MODEL_PATH,
    resume_from=None,
    iterations=4000,
    batch_size=16,
    positive_prob=0.5,
    loss_every=100,
    eval_every=500,
    oneshot_n=250,
    device="auto",
    seed=42,
    model_type=MODEL_TYPE_LEGACY,
    contrastive_margin=1.0,
    embedding_dim=256,
    log_callback=print,
):
    def log(message):
        log_callback(str(message))

    np.random.seed(seed)
    random.seed(seed)

    _, selected_device, visible_gpus = configure_device(device)
    log(f"TensorFlow device mode: {selected_device}")
    if visible_gpus:
        log(f"Visible GPUs: {', '.join(visible_gpus)}")
    log(f"Training model type: {model_type}")

    from model import build_model_for_type

    train_dir = resolve_path(train_dir)
    eval_dir = resolve_path(eval_dir)
    save_path = suggest_run_save_path(save_path, model_type=model_type, iterations=iterations, make_unique=True)
    resume_path = resolve_path(resume_from) if resume_from else None
    log(f"Resolved weights output path: {save_path}")

    train_index = build_person_index(train_dir)
    eval_index = build_person_index(eval_dir)

    model = build_model_for_type(
        model_type,
        compile_model=True,
        embedding_dim=embedding_dim,
        contrastive_margin=contrastive_margin,
    )

    if resume_path:
        model.load_weights(str(resume_path))
        existing_metadata = read_model_metadata(resume_path)
        if existing_metadata:
            log(f"Loaded weights from {resume_path} ({existing_metadata.get('model_type', 'unknown model type')})")
        else:
            log(f"Loaded weights from {resume_path}")

    checkpoint_metric = default_metric_for_model_type(model_type)
    best_accuracy = -1.0
    best_thresholds = {checkpoint_metric: default_threshold_for(model_type, checkpoint_metric)}
    loss_history = []
    loss_steps = []
    train_loss_points = []
    validation_loss_points = []
    accuracy_steps = []
    accuracy_points = []
    plot_paths = {
        "loss_curve": None,
        "accuracy_curve": None,
        "distance_histogram": None,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for step in range(iterations):
        inputs, targets = get_minibatch(
            batch_size=batch_size,
            prob=positive_prob,
            path=train_dir,
            person_index=train_index,
        )
        metrics = model.train_on_batch(inputs, targets, return_dict=True)
        loss_history.append(float(metrics["loss"]))

        if step % loss_every == 0:
            val_inputs, val_targets = get_minibatch(
                batch_size=batch_size,
                prob=positive_prob,
                path=eval_dir,
                person_index=eval_index,
            )
            validation_metrics = model.test_on_batch(val_inputs, val_targets, return_dict=True)
            log(
                f"iteration {step}, training loss: {np.mean(loss_history):.7f}, "
                f"validation loss: {float(validation_metrics['loss']):.7f}"
            )
            loss_steps.append(int(step))
            train_loss_points.append(float(np.mean(loss_history)))
            validation_loss_points.append(float(validation_metrics["loss"]))
            loss_history.clear()

        if step % eval_every == 0:
            eval_results = evaluate_pair_metrics(
                model,
                n_samples=oneshot_n,
                path=eval_dir,
                person_index=eval_index,
                model_type=model_type,
            )
            metric_summary = []
            for metric_name, values in eval_results.items():
                metric_summary.append(
                    f"{metric_name}: {values['accuracy']:.2f}% @ threshold {values['threshold']:.4f}"
                )
            log(f"Validation metrics on {oneshot_n} random pairs -> " + "; ".join(metric_summary))

            validation_accuracy = eval_results[checkpoint_metric]["accuracy"]
            accuracy_steps.append(int(step))
            accuracy_points.append(float(validation_accuracy))
            if validation_accuracy >= best_accuracy:
                model.save_weights(str(save_path))
                best_accuracy = validation_accuracy
                best_thresholds = {name: values["threshold"] for name, values in eval_results.items()}
                metadata = _build_training_metadata(
                    model_type=model_type,
                    save_path=save_path,
                    train_dir=train_dir,
                    eval_dir=eval_dir,
                    iterations=iterations,
                    batch_size=batch_size,
                    positive_prob=positive_prob,
                    loss_every=loss_every,
                    eval_every=eval_every,
                    oneshot_n=oneshot_n,
                    seed=seed,
                    selected_device=selected_device,
                    visible_gpus=visible_gpus,
                    recommended_thresholds=best_thresholds,
                    best_accuracy=best_accuracy,
                    checkpoint_metric=checkpoint_metric,
                    contrastive_margin=contrastive_margin,
                    embedding_dim=embedding_dim,
                )
                metadata_path = write_model_metadata(save_path, metadata)
                log(f"Saved improved weights to {save_path}")
                log(f"Wrote model metadata to {metadata_path}")

    try:
        if loss_steps:
            plot_paths["loss_curve"] = _save_loss_curve(save_path, loss_steps, train_loss_points, validation_loss_points)
            log(f"Saved loss curve to {plot_paths['loss_curve']}")
        if accuracy_steps:
            plot_paths["accuracy_curve"] = _save_accuracy_curve(save_path, accuracy_steps, accuracy_points, checkpoint_metric)
            log(f"Saved accuracy curve to {plot_paths['accuracy_curve']}")
        if model_type == MODEL_TYPE_CONTRASTIVE and best_accuracy >= 0.0 and save_path.exists():
            model.load_weights(str(save_path))
            histogram_targets, histogram_metrics = sample_pair_metrics(
                model,
                n_samples=max(oneshot_n, 400),
                path=eval_dir,
                person_index=eval_index,
                model_type=model_type,
            )
            euclidean_values = np.asarray(histogram_metrics[MODEL_METRIC_EUCLIDEAN], dtype=np.float32)
            positive_distances = euclidean_values[np.asarray(histogram_targets) == 1]
            negative_distances = euclidean_values[np.asarray(histogram_targets) == 0]
            plot_paths["distance_histogram"] = _save_distance_histogram(
                save_path,
                positive_distances,
                negative_distances,
            )
            log(f"Saved distance histogram to {plot_paths['distance_histogram']}")
    except Exception as exc:
        log(f"Plot generation skipped: {exc}")

    log(f"Best validation accuracy observed: {best_accuracy:.2f}%")
    log(
        "Recommended thresholds: "
        + ", ".join(f"{name}={value:.4f}" for name, value in best_thresholds.items())
    )
    log(f"Best weights path: {save_path}")
    return {
        "best_accuracy": best_accuracy,
        "save_path": str(save_path),
        "selected_device": selected_device,
        "visible_gpus": visible_gpus,
        "model_type": model_type,
        "checkpoint_metric": checkpoint_metric,
        "recommended_thresholds": best_thresholds,
        "plot_paths": plot_paths,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Siamese face matching model.")
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR), help="Training dataset directory")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR), help="Validation dataset directory")
    parser.add_argument(
        "--save-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to save best weights, e.g. saved_best.weights.h5",
    )
    parser.add_argument("--resume-from", default=None, help="Optional weights file to resume from")
    parser.add_argument("--iterations", type=int, default=4000, help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--positive-prob", type=float, default=0.5, help="Probability of sampling a positive pair")
    parser.add_argument("--loss-every", type=int, default=100, help="Steps between training loss logs")
    parser.add_argument("--eval-every", type=int, default=500, help="Steps between validation checks")
    parser.add_argument(
        "--oneshot-n",
        type=int,
        default=250,
        help="Number of random validation pairs used during each evaluation",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-type",
        choices=[MODEL_TYPE_LEGACY, MODEL_TYPE_CONTRASTIVE],
        default=MODEL_TYPE_LEGACY,
        help="Training objective and inference style",
    )
    parser.add_argument(
        "--contrastive-margin",
        type=float,
        default=1.0,
        help="Margin used only for the contrastive model type",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding size used only for the contrastive model type",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(
        train_dir=args.train_dir,
        eval_dir=args.eval_dir,
        save_path=args.save_path,
        resume_from=args.resume_from,
        iterations=args.iterations,
        batch_size=args.batch_size,
        positive_prob=args.positive_prob,
        loss_every=args.loss_every,
        eval_every=args.eval_every,
        oneshot_n=args.oneshot_n,
        device=args.device,
        seed=args.seed,
        model_type=args.model_type,
        contrastive_margin=args.contrastive_margin,
        embedding_dim=args.embedding_dim,
        log_callback=print,
    )


if __name__ == "__main__":
    main()
