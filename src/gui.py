import html
import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import gradio as gr
import numpy as np

from datafetch import run_fetch
from main import (
    compare_pair_inputs,
    determine_threshold,
    load_database,
    load_model,
    resolve_metric,
    search_image_against_database,
)
from similarity import default_metric_for_model_type
from train import summarize_people, train_model
from utils import (
    DEFAULT_DATABASE_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAIN_DIR,
    MODEL_METRIC_AUTO,
    MODEL_METRIC_COSINE,
    MODEL_METRIC_EUCLIDEAN,
    MODEL_METRIC_PROBABILITY,
    MODEL_TYPE_AUTO,
    MODEL_TYPE_CONTRASTIVE,
    MODEL_TYPE_LEGACY,
    resolve_path,
)

MODEL_CACHE = {
    "key": None,
    "model": None,
    "info": None,
}
JOB_LOCK = threading.Lock()


def field_help_html(title, help_text):
    return (
        "<div style='font-weight:600; margin-bottom:4px;'>"
        f"{html.escape(title)} "
        f"<sup title='{html.escape(help_text, quote=True)}' "
        "style='cursor:help; color:#1d4ed8; font-size:0.9em;'>ⓘ</sup>"
        "</div>"
    )


def _create_dialog_root():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def _normalize_existing_path(value, fallback):
    if value:
        try:
            return Path(str(value)).expanduser().resolve()
        except OSError:
            pass
    return Path(fallback).expanduser().resolve()


def choose_directory_dialog(current_path):
    current = _normalize_existing_path(current_path, DEFAULT_TRAIN_DIR)
    root = _create_dialog_root()
    try:
        selected = filedialog.askdirectory(initialdir=str(current), mustexist=False)
    finally:
        root.destroy()
    return selected or str(current)


def choose_directory_for_state(current_path):
    selected = choose_directory_dialog(current_path)
    return selected, selected


def choose_open_file_dialog(current_path):
    current_value = str(current_path).strip() if current_path else ""
    current = _normalize_existing_path(current_value or DEFAULT_MODEL_PATH, DEFAULT_MODEL_PATH)
    initial_dir = current.parent if current.suffix else current
    root = _create_dialog_root()
    try:
        selected = filedialog.askopenfilename(
            initialdir=str(initial_dir),
            filetypes=[("H5 weights", "*.h5"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return selected or current_value or ""


def choose_save_file_dialog(current_path):
    current = _normalize_existing_path(current_path, DEFAULT_MODEL_PATH)
    initial_dir = current.parent if current.parent.exists() else Path(DEFAULT_MODEL_PATH).parent
    root = _create_dialog_root()
    try:
        selected = filedialog.asksaveasfilename(
            initialdir=str(initial_dir),
            initialfile=current.name,
            defaultextension=".h5",
            filetypes=[("H5 weights", "*.h5"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return selected or str(current)


def summarize_training_state(train_dir, eval_dir):
    train_summary = summarize_people(train_dir)
    eval_summary = summarize_people(eval_dir)
    return (
        f"Train path: {train_summary['path']}\n"
        f"Train people: {train_summary['people']}\n"
        f"Train images: {train_summary['images']}\n\n"
        f"Eval path: {eval_summary['path']}\n"
        f"Eval people: {eval_summary['people']}\n"
        f"Eval images: {eval_summary['images']}"
    )


def preview_gallery(database_dir, max_items=12):
    database_dir = resolve_path(database_dir)
    try:
        _, labels, paths = load_database(database_dir, return_paths=True)
    except Exception as exc:
        return f"Gallery status: {exc}", []

    preview_items = []
    for path, label in zip(paths[:max_items], labels[:max_items]):
        preview_items.append((path, label))

    return (
        f"Gallery path: {database_dir}\n"
        f"Unique labels: {len(set(labels))}\n"
        f"Reference images: {len(paths)}"
    ), preview_items


def format_metric_value(value):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def format_embedding_preview(embedding, limit=12):
    values = list(np.asarray(embedding).reshape(-1)) if embedding is not None else []
    if not values:
        return "No embedding available."
    shown = ", ".join(f"{value:.4f}" for value in values[:limit])
    if len(values) > limit:
        shown += ", ..."
    return f"Dimension: {len(values)}\n[{shown}]"


def get_cached_model(model_path, device, model_type):
    model_path = resolve_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    cache_key = (str(model_path), device, model_type, model_path.stat().st_mtime_ns)
    if MODEL_CACHE["key"] == cache_key and MODEL_CACHE["model"] is not None:
        return MODEL_CACHE["model"], MODEL_CACHE["info"], True

    runtime_model, info = load_model(model_path, device=device, model_type=model_type)
    MODEL_CACHE["key"] = cache_key
    MODEL_CACHE["model"] = runtime_model
    MODEL_CACHE["info"] = info
    return runtime_model, info, False


def load_model_action(model_path, device, model_type):
    try:
        _, info, cached = get_cached_model(model_path, device, model_type)
        cache_note = "reused cached model" if cached else "loaded model from disk"
        gpu_text = ", ".join(info["visible_gpus"]) if info["visible_gpus"] else "none"
        resolved_metric = default_metric_for_model_type(info["model_type"])
        recommended_threshold = determine_threshold(info, resolved_metric, threshold=None)
        threshold_lines = ", ".join(
            f"{name}={value:.4f}" for name, value in info.get("recommended_thresholds", {}).items()
        )
        return (
            f"Model ready\n"
            f"Path: {info['path']}\n"
            f"Model type: {info['model_type']}\n"
            f"Training objective: {info.get('training_objective') or 'unknown'}\n"
            f"Device: {info['device']}\n"
            f"Visible GPUs: {gpu_text}\n"
            f"Recommended thresholds: {threshold_lines or 'none saved'}\n"
            f"Status: {cache_note}",
            recommended_threshold,
            resolved_metric,
        )
    except Exception as exc:
        return f"Model load failed: {exc}", 0.5, MODEL_METRIC_AUTO


def sync_threshold_to_metric(model_path, device, model_type, metric, current_threshold):
    try:
        _, info, _ = get_cached_model(model_path, device, model_type)
        resolved_metric = resolve_metric(info["model_type"], metric)
        return determine_threshold(info, resolved_metric, threshold=None)
    except Exception:
        return current_threshold


def _normalize_job_result(value, result_count):
    if result_count == 1:
        return (value,)
    if isinstance(value, (tuple, list)) and len(value) == result_count:
        return tuple(value)
    raise ValueError(f"Expected {result_count} result values, received {value!r}")


def run_background_job(job_fn, result_count=1, running_result=None, failed_result=None):
    if not JOB_LOCK.acquire(blocking=False):
        running_result = running_result or tuple([None] * result_count)
        yield ("Another fetch/train job is already running.",) + tuple(running_result)
        return

    logs = []
    result = {}
    error = {}

    def log_callback(message):
        logs.append(str(message))

    def worker():
        try:
            result["value"] = job_fn(log_callback)
        except Exception:
            error["traceback"] = traceback.format_exc()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while thread.is_alive():
            time.sleep(0.5)
            current_logs = "\n".join(logs) if logs else "Working..."
            yield (current_logs,) + tuple(running_result or tuple([None] * result_count))
        thread.join()

        if "traceback" in error:
            logs.append(error["traceback"])
            yield ("\n".join(logs),) + tuple(failed_result or tuple([None] * result_count))
            return

        yield ("\n".join(logs),) + _normalize_job_result(result["value"], result_count)
    finally:
        JOB_LOCK.release()


def fetch_data_action(train_dir, eval_dir):
    def job(log_callback):
        run_fetch(train_dir=train_dir, eval_dir=eval_dir, max_rows=None, log_callback=log_callback)
        return summarize_training_state(train_dir, eval_dir)

    yield from run_background_job(job, result_count=1, running_result=("Running",), failed_result=("Failed",))


def train_action(
    train_dir,
    eval_dir,
    save_path,
    resume_from,
    model_type,
    iterations,
    batch_size,
    positive_prob,
    loss_every,
    eval_every,
    oneshot_n,
    device,
    seed,
    contrastive_margin,
    embedding_dim,
):
    def job(log_callback):
        result = train_model(
            train_dir=train_dir,
            eval_dir=eval_dir,
            save_path=save_path,
            resume_from=resume_from or None,
            iterations=int(iterations),
            batch_size=int(batch_size),
            positive_prob=float(positive_prob),
            loss_every=int(loss_every),
            eval_every=int(eval_every),
            oneshot_n=int(oneshot_n),
            device=device,
            seed=int(seed),
            model_type=model_type,
            contrastive_margin=float(contrastive_margin),
            embedding_dim=int(embedding_dim),
            log_callback=log_callback,
        )
        threshold_summary = ", ".join(
            f"{name}={value:.4f}" for name, value in result["recommended_thresholds"].items()
        )
        status_text = (
            f"Training finished\n"
            f"Model type: {result['model_type']}\n"
            f"Best accuracy: {result['best_accuracy']:.2f}%\n"
            f"Checkpoint metric: {result['checkpoint_metric']}\n"
            f"Recommended thresholds: {threshold_summary}\n"
            f"Weights: {result['save_path']}\n"
            f"Device: {result['selected_device']}"
        )
        return (
            status_text,
            result["plot_paths"].get("loss_curve"),
            result["plot_paths"].get("accuracy_curve"),
            result["plot_paths"].get("distance_histogram"),
        )

    yield from run_background_job(
        job,
        result_count=4,
        running_result=("Running", None, None, None),
        failed_result=("Failed", None, None, None),
    )


def pairwise_action(model_path, device, model_type, metric, threshold, image_left, image_right):
    if image_left is None or image_right is None:
        raise gr.Error("Provide both images for 1:1 matching.")

    runtime_model, info, _ = get_cached_model(model_path, device, model_type)
    result = compare_pair_inputs(
        image_left,
        image_right,
        runtime_model,
        threshold=float(threshold),
        metric=metric,
    )
    status = (
        f"Decision: {result['verdict']}\n"
        f"Model type: {info['model_type']}\n"
        f"Decision metric: {result['metric']}\n"
        f"Decision value: {result['score']:.4f}\n"
        f"Probability: {format_metric_value(result['probability'])}\n"
        f"Euclidean distance: {format_metric_value(result['euclidean'])}\n"
        f"Cosine similarity: {format_metric_value(result['cosine'])}\n"
        f"Threshold used: {float(threshold):.4f}\n"
        f"Faces detected: left={result['face_count'][0]}, right={result['face_count'][1]}\n"
        f"Detection mode: left={result['detection_mode'][0]}, right={result['detection_mode'][1]}\n"
        f"Model device: {info['device']}"
    )
    return (
        status,
        result["left_annotated"],
        result["right_annotated"],
        result["left_face"],
        result["right_face"],
        format_embedding_preview(result["left_embedding"]),
        format_embedding_preview(result["right_embedding"]),
    )


def search_action(model_path, device, model_type, metric, threshold, database_dir, query_image):
    if query_image is None:
        raise gr.Error("Provide a query image for 1:N search.")

    runtime_model, info, _ = get_cached_model(model_path, device, model_type)
    result = search_image_against_database(
        query_image,
        runtime_model,
        database_dir=database_dir,
        threshold=float(threshold),
        metric=metric,
    )
    summary = (
        f"Model type: {info['model_type']}\n"
        f"Decision metric: {result['metric']}\n"
        f"Threshold used: {float(threshold):.4f}\n\n"
        f"{result['summary']}\n\n"
        f"Faces detected: {result['face_count']}\n"
        f"Model device: {info['device']}"
    )
    return result["annotated_image"], result["rankings"], summary


def build_demo():
    with gr.Blocks(title="Siamese Face Recognition Demo") as demo:
        gr.Markdown(
            """
            # Siamese Face Recognition Demo
            **Dr Prakash Karn**  
            **University of Auckland**  
            **COMSYS721 Machine Intelligence and Deep Learning**

            This app supports two model styles:
            - `legacy_classifier`: shared encoder + absolute difference + sigmoid probability
            - `contrastive_embedding`: shared encoder trained with contrastive loss for Euclidean and cosine similarity
            """
        )

        with gr.Tab("Training"):
            gr.Markdown(
                """
                **Simple flow**

                1. Click **Download Full Dataset**. This automatically creates and fills `train/` and `eval/`.
                2. Choose the model type you want to train.
                3. Click **Start Training**.
                4. Load the saved weights in the **Inference** tab.
                """
            )

            train_dir = gr.State(str(DEFAULT_TRAIN_DIR))
            eval_dir = gr.State(str(DEFAULT_EVAL_DIR))

            dataset_summary = gr.Textbox(
                label="Training Data Status",
                lines=8,
                value="Click 'Check Dataset' to scan train/ and eval/.",
            )

            with gr.Group():
                gr.Markdown("## Step 1: Download Dataset")
                gr.Markdown(
                    "Use **Download Full Dataset** to fetch the training images from the web links defined in `datafetch.py`. "
                    "The images will be saved automatically into `train/` and `eval/`."
                )
                with gr.Row():
                    summarize_btn = gr.Button("Check Dataset")
                    full_fetch_btn = gr.Button("Download Full Dataset", variant="primary")
                fetch_log = gr.Textbox(label="Fetch Log", lines=16)

            with gr.Group():
                gr.Markdown("## Step 2: Train Model")
                with gr.Row():
                    with gr.Column():
                        gr.HTML(
                            field_help_html(
                                "Model Type",
                                "Choose the training objective. Recommended for this lecture: contrastive_embedding.",
                            )
                        )
                        train_model_type = gr.Dropdown(
                            choices=[MODEL_TYPE_CONTRASTIVE, MODEL_TYPE_LEGACY],
                            value=MODEL_TYPE_CONTRASTIVE,
                            show_label=False,
                        )
                    with gr.Column():
                        gr.HTML(
                            field_help_html(
                                "Iterations",
                                "Number of training steps. Recommended: 4000 for a fuller run, 200-500 for quick tests.",
                            )
                        )
                        iterations = gr.Number(value=4000, precision=0, show_label=False)
                    with gr.Column():
                        gr.HTML(
                            field_help_html(
                                "Batch Size",
                                "Number of image pairs per update step. Recommended: 16 on CPU.",
                            )
                        )
                        batch_size = gr.Number(value=16, precision=0, show_label=False)
                    with gr.Column():
                        gr.HTML(
                            field_help_html(
                                "Device",
                                "Execution device for training. Recommended: cpu on this PC.",
                            )
                        )
                        train_device = gr.Dropdown(choices=["cpu", "auto", "gpu"], value="cpu", show_label=False)
                with gr.Row():
                    with gr.Column():
                        gr.HTML(
                            field_help_html(
                                "Save Weights Path",
                                "Where the best model weights will be saved. Recommended: saved_best.weights.h5",
                            )
                        )
                        with gr.Row():
                            save_path = gr.Textbox(value=str(DEFAULT_MODEL_PATH), show_label=False, interactive=False)
                            save_path_btn = gr.Button("Select File")
                    with gr.Column():
                        gr.HTML(
                            field_help_html(
                                "Resume Weights Path",
                                "Optional existing weights file to continue training from. Recommended: leave empty on first run.",
                            )
                        )
                        with gr.Row():
                            resume_from = gr.Textbox(value="", show_label=False, interactive=False)
                            resume_from_btn = gr.Button("Select File")
                with gr.Accordion("Advanced Options (optional)", open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.HTML(
                                field_help_html(
                                    "Positive Pair Probability",
                                    "Fraction of sampled pairs that come from the same identity. Recommended: 0.5.",
                                )
                            )
                            positive_prob = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, show_label=False)
                        with gr.Column():
                            gr.HTML(
                                field_help_html(
                                    "Loss Every",
                                    "How often to print training and validation loss. Recommended: 100.",
                                )
                            )
                            loss_every = gr.Number(value=100, precision=0, show_label=False)
                        with gr.Column():
                            gr.HTML(
                                field_help_html(
                                    "Eval Every",
                                    "How often to test validation pair accuracy and save the best model. Recommended: 500.",
                                )
                            )
                            eval_every = gr.Number(value=500, precision=0, show_label=False)
                    with gr.Row():
                        with gr.Column():
                            gr.HTML(
                                field_help_html(
                                    "Eval Pair Count",
                                    "Number of random validation pairs used each evaluation. Recommended: 250.",
                                )
                            )
                            oneshot_n = gr.Number(value=250, precision=0, show_label=False)
                        with gr.Column():
                            gr.HTML(
                                field_help_html(
                                    "Random Seed",
                                    "Seed for reproducible sampling and initialization. Recommended: 42.",
                                )
                            )
                            seed = gr.Number(value=42, precision=0, show_label=False)
                        with gr.Column():
                            gr.HTML(
                                field_help_html(
                                    "Contrastive Margin",
                                    "Used only for the contrastive model. Recommended: 1.0.",
                                )
                            )
                            contrastive_margin = gr.Number(value=1.0, precision=2, show_label=False)
                        with gr.Column():
                            gr.HTML(
                                field_help_html(
                                    "Embedding Dimension",
                                    "Size of the learned embedding used only for the contrastive model. Recommended: 256.",
                                )
                            )
                            embedding_dim = gr.Number(value=256, precision=0, show_label=False)
                    gr.Markdown("Optional folder overrides")
                    with gr.Row():
                        with gr.Column():
                            gr.HTML(
                                field_help_html(
                                    "Train Directory",
                                    "Folder containing identities for training. Recommended: train/",
                                )
                            )
                            with gr.Row():
                                train_dir_display = gr.Textbox(
                                    value=str(DEFAULT_TRAIN_DIR), show_label=False, interactive=False
                                )
                                train_dir_btn = gr.Button("Select Folder")
                        with gr.Column():
                            gr.HTML(
                                field_help_html(
                                    "Eval Directory",
                                    "Folder containing identities for validation. Recommended: eval/",
                                )
                            )
                            with gr.Row():
                                eval_dir_display = gr.Textbox(
                                    value=str(DEFAULT_EVAL_DIR), show_label=False, interactive=False
                                )
                                eval_dir_btn = gr.Button("Select Folder")
                train_btn = gr.Button("Start Training", variant="primary")
                train_log = gr.Textbox(label="Training Log", lines=18)
                train_status = gr.Textbox(label="Training Status", lines=7)
                with gr.Row():
                    loss_curve_plot = gr.Image(label="Loss Curve", type="filepath")
                    accuracy_curve_plot = gr.Image(label="Accuracy Curve", type="filepath")
                distance_histogram_plot = gr.Image(
                    label="Contrastive Distance Histogram",
                    type="filepath",
                )

            summarize_btn.click(
                fn=summarize_training_state,
                inputs=[train_dir, eval_dir],
                outputs=dataset_summary,
                queue=False,
            )

            full_fetch_btn.click(
                fn=fetch_data_action,
                inputs=[train_dir, eval_dir],
                outputs=[fetch_log, dataset_summary],
            )

            train_dir_btn.click(
                fn=choose_directory_for_state,
                inputs=[train_dir_display],
                outputs=[train_dir_display, train_dir],
            )
            eval_dir_btn.click(
                fn=choose_directory_for_state,
                inputs=[eval_dir_display],
                outputs=[eval_dir_display, eval_dir],
            )
            save_path_btn.click(fn=choose_save_file_dialog, inputs=[save_path], outputs=[save_path])
            resume_from_btn.click(fn=choose_open_file_dialog, inputs=[resume_from], outputs=[resume_from])

            train_btn.click(
                fn=train_action,
                inputs=[
                    train_dir,
                    eval_dir,
                    save_path,
                    resume_from,
                    train_model_type,
                    iterations,
                    batch_size,
                    positive_prob,
                    loss_every,
                    eval_every,
                    oneshot_n,
                    train_device,
                    seed,
                    contrastive_margin,
                    embedding_dim,
                ],
                outputs=[
                    train_log,
                    train_status,
                    loss_curve_plot,
                    accuracy_curve_plot,
                    distance_histogram_plot,
                ],
            )

        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML(
                        field_help_html(
                            "Model Weights Path",
                            "Saved weights file used for inference. Recommended: saved_best.weights.h5",
                        )
                    )
                    with gr.Row():
                        model_path = gr.Textbox(value=str(DEFAULT_MODEL_PATH), show_label=False, interactive=False)
                        model_path_btn = gr.Button("Select File")
                with gr.Column():
                    gr.HTML(
                        field_help_html(
                            "Model Type",
                            "Use auto if the weights were trained in this GUI. Otherwise choose the correct type manually.",
                        )
                    )
                    infer_model_type = gr.Dropdown(
                        choices=[MODEL_TYPE_AUTO, MODEL_TYPE_CONTRASTIVE, MODEL_TYPE_LEGACY],
                        value=MODEL_TYPE_AUTO,
                        show_label=False,
                    )
                with gr.Column():
                    gr.HTML(
                        field_help_html(
                            "Device",
                            "Execution device for inference. Recommended: cpu on this PC.",
                        )
                    )
                    infer_device = gr.Dropdown(choices=["cpu", "auto", "gpu"], value="cpu", show_label=False)
            with gr.Row():
                with gr.Column():
                    gr.HTML(
                        field_help_html(
                            "Scoring Metric",
                            "Auto selects probability for legacy weights and Euclidean distance for contrastive weights.",
                        )
                    )
                    infer_metric = gr.Dropdown(
                        choices=[
                            MODEL_METRIC_AUTO,
                            MODEL_METRIC_PROBABILITY,
                            MODEL_METRIC_EUCLIDEAN,
                            MODEL_METRIC_COSINE,
                        ],
                        value=MODEL_METRIC_AUTO,
                        show_label=False,
                    )
                with gr.Column():
                    gr.HTML(
                        field_help_html(
                            "Threshold",
                            "Decision threshold. The Load button will fill a recommended value from model metadata when available.",
                        )
                    )
                    threshold = gr.Number(value=0.5, precision=4, show_label=False)
            load_model_btn = gr.Button("Load / Check Model")
            model_status = gr.Textbox(label="Model Status", lines=8)

            with gr.Tab("1:1 Match"):
                with gr.Row():
                    pair_left = gr.Image(type="numpy", label="Image A")
                    pair_right = gr.Image(type="numpy", label="Image B")
                pair_btn = gr.Button("Compare Pair", variant="primary")
                pair_status = gr.Textbox(label="Pair Result", lines=10)
                with gr.Row():
                    pair_left_annotated = gr.Image(label="Annotated Image A")
                    pair_right_annotated = gr.Image(label="Annotated Image B")
                with gr.Row():
                    pair_left_face = gr.Image(label="Detected Face A")
                    pair_right_face = gr.Image(label="Detected Face B")
                with gr.Row():
                    pair_left_embedding = gr.Textbox(label="Embedding A Preview", lines=4)
                    pair_right_embedding = gr.Textbox(label="Embedding B Preview", lines=4)

            with gr.Tab("1:N Search"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML(
                            field_help_html(
                                "Gallery Directory",
                                "Folder containing enrolled identities for 1:N search. Recommended: database/",
                            )
                        )
                        with gr.Row():
                            gallery_dir = gr.Textbox(
                                value=str(DEFAULT_DATABASE_DIR), show_label=False, interactive=False
                            )
                            gallery_dir_btn = gr.Button("Select Folder")
                    refresh_gallery_btn = gr.Button("Refresh Gallery")
                gallery_status = gr.Textbox(
                    label="Gallery Status",
                    lines=4,
                    value="Click 'Refresh Gallery' to scan the current gallery folder.",
                )
                gallery_preview = gr.Gallery(label="Gallery Preview", columns=4, height=220)
                query_image = gr.Image(type="numpy", label="Query Image")
                search_btn = gr.Button("Run 1:N Search", variant="primary")
                search_summary = gr.Textbox(label="Search Summary", lines=10)
                search_annotated = gr.Image(label="Annotated Query Image")
                search_rankings = gr.Dataframe(
                    headers=[
                        "face_index",
                        "rank",
                        "label",
                        "metric",
                        "metric_value",
                        "euclidean",
                        "cosine",
                        "probability",
                    ],
                    datatype=["number", "number", "str", "str", "number", "number", "number", "number"],
                    label="Ranked Matches",
                )

            load_model_btn.click(
                fn=load_model_action,
                inputs=[model_path, infer_device, infer_model_type],
                outputs=[model_status, threshold, infer_metric],
                queue=False,
            )

            infer_metric.change(
                fn=sync_threshold_to_metric,
                inputs=[model_path, infer_device, infer_model_type, infer_metric, threshold],
                outputs=[threshold],
                queue=False,
            )

            model_path_btn.click(fn=choose_open_file_dialog, inputs=[model_path], outputs=[model_path])
            gallery_dir_btn.click(fn=choose_directory_dialog, inputs=[gallery_dir], outputs=[gallery_dir])

            pair_btn.click(
                fn=pairwise_action,
                inputs=[model_path, infer_device, infer_model_type, infer_metric, threshold, pair_left, pair_right],
                outputs=[
                    pair_status,
                    pair_left_annotated,
                    pair_right_annotated,
                    pair_left_face,
                    pair_right_face,
                    pair_left_embedding,
                    pair_right_embedding,
                ],
            )

            refresh_gallery_btn.click(
                fn=preview_gallery,
                inputs=[gallery_dir],
                outputs=[gallery_status, gallery_preview],
                queue=False,
            )

            search_btn.click(
                fn=search_action,
                inputs=[model_path, infer_device, infer_model_type, infer_metric, threshold, gallery_dir, query_image],
                outputs=[search_annotated, search_rankings, search_summary],
            )

        demo.queue(max_size=4)
    return demo


if __name__ == "__main__":
    build_demo().launch()
