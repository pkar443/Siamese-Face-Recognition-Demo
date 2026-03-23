import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from runtime import configure_device
from similarity import (
    cosine_similarities,
    decisions_from_values,
    default_metric_for_model_type,
    default_threshold_for,
    euclidean_distances,
    metric_is_higher_better,
)
from utils import (
    DEFAULT_CASCADE_PATH,
    DEFAULT_DATABASE_DIR,
    DEFAULT_OUTPUT_DIR,
    MODEL_METRIC_AUTO,
    MODEL_METRIC_COSINE,
    MODEL_METRIC_EUCLIDEAN,
    MODEL_METRIC_PROBABILITY,
    MODEL_TYPE_AUTO,
    MODEL_TYPE_CONTRASTIVE,
    MODEL_TYPE_LEGACY,
    list_image_files,
    is_lfs_pointer,
    read_model_metadata,
    resolve_path,
)

IMAGE_SHAPE = (100, 100)


def load_face_image(image_path, image_shape=IMAGE_SHAPE):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if image.shape != image_shape:
        image = cv2.resize(image, image_shape, interpolation=cv2.INTER_AREA)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, -1)


def load_database(database_dir, return_paths=False):
    database_dir = resolve_path(database_dir)
    if not database_dir.exists():
        raise FileNotFoundError(f"Database directory not found: {database_dir}")

    labels = []
    reference_images = []
    reference_paths = []

    for entry in sorted(database_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            labels.append(entry.stem)
            reference_images.append(load_face_image(entry))
            reference_paths.append(str(entry))
            continue

        if entry.is_dir():
            images = list_image_files(entry)
            for image_path in images:
                labels.append(entry.name)
                reference_images.append(load_face_image(image_path))
                reference_paths.append(str(image_path))

    if not reference_images:
        raise ValueError(f"No reference images found in {database_dir}")

    if return_paths:
        return np.asarray(reference_images), labels, reference_paths
    return np.asarray(reference_images), labels


def build_face_detector(cascade_path):
    cascade_path = resolve_path(cascade_path)
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise FileNotFoundError(f"Could not load Haar cascade: {cascade_path}")
    return detector


def to_bgr_image(image_input):
    if isinstance(image_input, (str, Path)):
        image = cv2.imread(str(resolve_path(image_input)), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_input}")
        return image

    image = np.asarray(image_input)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("Unsupported image input shape.")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def to_rgb_image(image_bgr):
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def detect_faces(image, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30),
    )
    ordered_faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    detections = []
    for x, y, width, height in ordered_faces:
        detections.append(
            {
                "face": gray[y : y + height, x : x + width],
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            }
        )
    return detections


def iter_faces(image, detector):
    for detection in detect_faces(image, detector):
        yield (
            detection["face"],
            detection["x"],
            detection["y"],
            detection["width"],
            detection["height"],
        )


def prepare_face(face, image_shape=IMAGE_SHAPE):
    face = cv2.resize(face, image_shape, interpolation=cv2.INTER_AREA)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, -1)


def _resolve_model_type(requested_model_type, metadata):
    if requested_model_type != MODEL_TYPE_AUTO:
        return requested_model_type
    return metadata.get("model_type", MODEL_TYPE_LEGACY)


def resolve_metric(model_type, requested_metric):
    if requested_metric == MODEL_METRIC_AUTO:
        return default_metric_for_model_type(model_type)
    if model_type == MODEL_TYPE_LEGACY and requested_metric != MODEL_METRIC_PROBABILITY:
        raise ValueError(
            "Legacy classifier weights only support probability-based decisions. "
            "Use contrastive weights for Euclidean or cosine scoring."
        )
    if model_type == MODEL_TYPE_CONTRASTIVE and requested_metric not in {
        MODEL_METRIC_EUCLIDEAN,
        MODEL_METRIC_COSINE,
    }:
        raise ValueError(
            "Contrastive weights support Euclidean distance or cosine similarity. "
            "Use legacy classifier weights for probability scoring."
        )
    return requested_metric


def determine_threshold(model_info, metric, threshold=None):
    if threshold is not None:
        return float(threshold)
    recommended = model_info.get("recommended_thresholds", {}).get(metric)
    if recommended is not None:
        return float(recommended)
    return default_threshold_for(model_info["model_type"], metric)


def draw_prediction(image, x, y, width, height, label, score, metric):
    metric_label = metric if metric != MODEL_METRIC_PROBABILITY else "prob"
    text = f"{label} ({metric_label}:{score:.2f})"
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(
        image,
        text,
        (x, max(20, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def load_model(model_path, device="cpu", model_type=MODEL_TYPE_AUTO):
    model_path = resolve_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if is_lfs_pointer(model_path):
        raise RuntimeError(
            f"{model_path} is a Git LFS pointer, not real model weights. Train locally or fetch the real file first."
        )

    metadata = read_model_metadata(model_path)
    resolved_model_type = _resolve_model_type(model_type, metadata)
    _, selected_device, visible_gpus = configure_device(device)

    from model import (
        DEFAULT_CONTRASTIVE_MARGIN,
        DEFAULT_EMBEDDING_DIM,
        build_model_for_type,
        get_encoder_for_model,
    )

    pair_model = build_model_for_type(
        resolved_model_type,
        compile_model=False,
        embedding_dim=int(metadata.get("embedding_dim", DEFAULT_EMBEDDING_DIM)),
        contrastive_margin=float(metadata.get("contrastive_margin", DEFAULT_CONTRASTIVE_MARGIN)),
    )
    pair_model.load_weights(str(model_path))
    encoder = get_encoder_for_model(pair_model, resolved_model_type)

    runtime_model = {
        "pair_model": pair_model,
        "encoder": encoder,
        "model_type": resolved_model_type,
        "metadata": metadata,
    }
    return runtime_model, {
        "path": str(model_path),
        "device": selected_device,
        "visible_gpus": visible_gpus,
        "model_type": resolved_model_type,
        "training_objective": metadata.get("training_objective"),
        "recommended_thresholds": metadata.get("recommended_thresholds", {}),
    }


def _compute_pair_details(face_left, face_right, runtime_model):
    left = np.expand_dims(prepare_face(face_left), axis=0)
    right = np.expand_dims(prepare_face(face_right), axis=0)
    left_embedding = runtime_model["encoder"].predict(left, verbose=0)
    right_embedding = runtime_model["encoder"].predict(right, verbose=0)

    euclidean = float(euclidean_distances(left_embedding, right_embedding)[0])
    cosine = float(cosine_similarities(left_embedding, right_embedding)[0])
    probability = None

    if runtime_model["model_type"] == MODEL_TYPE_LEGACY:
        probability = float(np.squeeze(runtime_model["pair_model"].predict([left, right], verbose=0)))
    elif runtime_model["model_type"] == MODEL_TYPE_CONTRASTIVE:
        euclidean = float(np.squeeze(runtime_model["pair_model"].predict([left, right], verbose=0)))

    return {
        "metrics": {
            MODEL_METRIC_EUCLIDEAN: euclidean,
            MODEL_METRIC_COSINE: cosine,
            MODEL_METRIC_PROBABILITY: probability,
        },
        "left_embedding": np.squeeze(left_embedding),
        "right_embedding": np.squeeze(right_embedding),
    }


def _embedding_for_face(face, runtime_model):
    face_batch = np.expand_dims(prepare_face(face), axis=0)
    embedding = runtime_model["encoder"].predict(face_batch, verbose=0)
    return np.squeeze(embedding)


def _aggregate_reference_rows(labels, probabilities=None, euclidean_values=None, cosine_values=None):
    label_metrics = defaultdict(lambda: {"probability": [], "euclidean": [], "cosine": []})
    for index, label in enumerate(labels):
        if probabilities is not None:
            label_metrics[label]["probability"].append(float(probabilities[index]))
        if euclidean_values is not None:
            label_metrics[label]["euclidean"].append(float(euclidean_values[index]))
        if cosine_values is not None:
            label_metrics[label]["cosine"].append(float(cosine_values[index]))

    rows = []
    for label, metrics in label_metrics.items():
        rows.append(
            {
                "label": label,
                MODEL_METRIC_PROBABILITY: max(metrics["probability"]) if metrics["probability"] else None,
                MODEL_METRIC_EUCLIDEAN: min(metrics["euclidean"]) if metrics["euclidean"] else None,
                MODEL_METRIC_COSINE: max(metrics["cosine"]) if metrics["cosine"] else None,
            }
        )
    return rows


def compare_pair_inputs(
    image_left,
    image_right,
    runtime_model,
    cascade_path=DEFAULT_CASCADE_PATH,
    threshold=0.5,
    metric=MODEL_METRIC_AUTO,
):
    detector = build_face_detector(cascade_path)
    left_bgr = to_bgr_image(image_left)
    right_bgr = to_bgr_image(image_right)

    left_faces = detect_faces(left_bgr, detector)
    right_faces = detect_faces(right_bgr, detector)
    if not left_faces:
        raise ValueError("No face detected in the first image.")
    if not right_faces:
        raise ValueError("No face detected in the second image.")

    left_face = left_faces[0]
    right_face = right_faces[0]
    pair_details = _compute_pair_details(left_face["face"], right_face["face"], runtime_model)
    resolved_metric = resolve_metric(runtime_model["model_type"], metric)
    score = pair_details["metrics"][resolved_metric]
    threshold = float(threshold)
    verdict = "Same person" if decisions_from_values([score], threshold, resolved_metric)[0] else "Different people"

    annotated_left = left_bgr.copy()
    annotated_right = right_bgr.copy()
    draw_prediction(
        annotated_left,
        left_face["x"],
        left_face["y"],
        left_face["width"],
        left_face["height"],
        "Face A",
        score,
        resolved_metric,
    )
    draw_prediction(
        annotated_right,
        right_face["x"],
        right_face["y"],
        right_face["width"],
        right_face["height"],
        "Face B",
        score,
        resolved_metric,
    )

    return {
        "score": score,
        "metric": resolved_metric,
        "verdict": verdict,
        "left_annotated": to_rgb_image(annotated_left),
        "right_annotated": to_rgb_image(annotated_right),
        "left_face": to_rgb_image(left_face["face"]),
        "right_face": to_rgb_image(right_face["face"]),
        "face_count": (len(left_faces), len(right_faces)),
        "probability": pair_details["metrics"][MODEL_METRIC_PROBABILITY],
        "euclidean": pair_details["metrics"][MODEL_METRIC_EUCLIDEAN],
        "cosine": pair_details["metrics"][MODEL_METRIC_COSINE],
        "left_embedding": pair_details["left_embedding"],
        "right_embedding": pair_details["right_embedding"],
    }


def search_image_against_database(
    image_input,
    runtime_model,
    database_dir=DEFAULT_DATABASE_DIR,
    threshold=0.5,
    cascade_path=DEFAULT_CASCADE_PATH,
    top_k=5,
    metric=MODEL_METRIC_AUTO,
):
    detector = build_face_detector(cascade_path)
    database_images, database_labels = load_database(database_dir)
    image_bgr = to_bgr_image(image_input)
    detections = detect_faces(image_bgr, detector)
    if not detections:
        raise ValueError("No face detected in the query image.")

    resolved_metric = resolve_metric(runtime_model["model_type"], metric)
    threshold = float(threshold)
    annotated = image_bgr.copy()
    ranking_rows = []
    summary_rows = []

    gallery_embeddings = None
    if resolved_metric in {MODEL_METRIC_EUCLIDEAN, MODEL_METRIC_COSINE}:
        gallery_embeddings = runtime_model["encoder"].predict(database_images, verbose=0)

    for face_index, detection in enumerate(detections, start=1):
        if resolved_metric == MODEL_METRIC_PROBABILITY:
            query_face = prepare_face(detection["face"])
            query_batch = np.repeat(query_face[np.newaxis, ...], len(database_labels), axis=0)
            probabilities = np.atleast_1d(
                np.squeeze(runtime_model["pair_model"].predict([query_batch, database_images], verbose=0))
            )
            rows = _aggregate_reference_rows(database_labels, probabilities=probabilities)
        else:
            query_embedding = _embedding_for_face(detection["face"], runtime_model)[np.newaxis, :]
            repeated_query = np.repeat(query_embedding, len(database_labels), axis=0)
            euclidean_values = euclidean_distances(repeated_query, gallery_embeddings)
            cosine_values = cosine_similarities(repeated_query, gallery_embeddings)
            rows = _aggregate_reference_rows(
                database_labels,
                euclidean_values=euclidean_values,
                cosine_values=cosine_values,
            )

        ranked = sorted(
            rows,
            key=lambda row: row[resolved_metric],
            reverse=metric_is_higher_better(resolved_metric),
        )
        best = ranked[0]
        best_score = best[resolved_metric]
        shown_label = best["label"] if decisions_from_values([best_score], threshold, resolved_metric)[0] else "Unknown"

        draw_prediction(
            annotated,
            detection["x"],
            detection["y"],
            detection["width"],
            detection["height"],
            shown_label,
            best_score,
            resolved_metric,
        )
        summary_rows.append(
            f"Face {face_index}: {shown_label} "
            f"({resolved_metric}={best_score:.4f}, "
            f"euclidean={best[MODEL_METRIC_EUCLIDEAN] if best[MODEL_METRIC_EUCLIDEAN] is not None else 'n/a'}, "
            f"cosine={best[MODEL_METRIC_COSINE] if best[MODEL_METRIC_COSINE] is not None else 'n/a'})"
        )
        for rank, row in enumerate(ranked[:top_k], start=1):
            ranking_rows.append(
                [
                    face_index,
                    rank,
                    row["label"],
                    resolved_metric,
                    round(float(row[resolved_metric]), 4),
                    round(float(row[MODEL_METRIC_EUCLIDEAN]), 4) if row[MODEL_METRIC_EUCLIDEAN] is not None else None,
                    round(float(row[MODEL_METRIC_COSINE]), 4) if row[MODEL_METRIC_COSINE] is not None else None,
                    round(float(row[MODEL_METRIC_PROBABILITY]), 4)
                    if row[MODEL_METRIC_PROBABILITY] is not None
                    else None,
                ]
            )

    return {
        "annotated_image": to_rgb_image(annotated),
        "summary": "\n".join(summary_rows),
        "rankings": ranking_rows,
        "face_count": len(detections),
        "metric": resolved_metric,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with saved Siamese network weights.")
    parser.add_argument("-db", "--database", default=str(DEFAULT_DATABASE_DIR), help="Folder of enrolled faces")
    parser.add_argument("-m", "--model", required=True, help="Path to saved weights (.weights.h5)")
    parser.add_argument("-i", "--image", nargs="+", dest="images", required=True, help="Image paths to annotate")
    parser.add_argument("--cascade", default=str(DEFAULT_CASCADE_PATH), help="Path to Haar cascade XML")
    parser.add_argument("--threshold", type=float, default=None, help="Optional match threshold override")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save annotated images")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=0,
        help="Optional number of eval pairs to score after loading weights",
    )
    parser.add_argument("--eval-dir", default=None, help="Optional eval directory used with --eval-batches")
    parser.add_argument(
        "--model-type",
        choices=[MODEL_TYPE_AUTO, MODEL_TYPE_LEGACY, MODEL_TYPE_CONTRASTIVE],
        default=MODEL_TYPE_AUTO,
        help="Model architecture to load. Use auto for metadata-based detection.",
    )
    parser.add_argument(
        "--metric",
        choices=[MODEL_METRIC_AUTO, MODEL_METRIC_PROBABILITY, MODEL_METRIC_EUCLIDEAN, MODEL_METRIC_COSINE],
        default=MODEL_METRIC_AUTO,
        help="Decision metric used during inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runtime_model, model_info = load_model(args.model, device=args.device, model_type=args.model_type)
    resolved_metric = resolve_metric(model_info["model_type"], args.metric)
    threshold = determine_threshold(model_info, resolved_metric, threshold=args.threshold)

    print(f"TensorFlow device mode: {model_info['device']}")
    if model_info["visible_gpus"]:
        print(f"Visible GPUs: {', '.join(model_info['visible_gpus'])}")
    print(f"Loaded weights from {model_info['path']}")
    print(f"Model type: {model_info['model_type']}")
    if model_info.get("training_objective"):
        print(f"Training objective: {model_info['training_objective']}")
    print(f"Inference metric: {resolved_metric}")
    print(f"Threshold: {threshold:.4f}")

    if args.eval_batches:
        from train import test_oneshot

        eval_dir = resolve_path(args.eval_dir) if args.eval_dir else resolve_path("eval")
        accuracy = test_oneshot(
            runtime_model["pair_model"],
            args.eval_batches,
            verbose=1,
            path=eval_dir,
            model_type=model_info["model_type"],
            metric=resolved_metric,
        )
        print(f"Eval pair accuracy: {accuracy:.2f}%")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in args.images:
        image_path = resolve_path(image_path)
        result = search_image_against_database(
            image_path,
            runtime_model,
            database_dir=args.database,
            threshold=threshold,
            cascade_path=args.cascade,
            metric=resolved_metric,
        )
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR))
        print(f"{image_path.name}: detected {result['face_count']} face(s), wrote {output_path}")
        print(result["summary"])


if __name__ == "__main__":
    main()
