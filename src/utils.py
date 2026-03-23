import json
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_DIR = REPO_ROOT / "train"
DEFAULT_EVAL_DIR = REPO_ROOT / "eval"
DEFAULT_DATABASE_DIR = REPO_ROOT / "database"
DEFAULT_MODEL_PATH = REPO_ROOT / "saved_best.weights.h5"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_CASCADE_PATH = REPO_ROOT / "haarcascade_frontalface_default.xml"
MODEL_TYPE_AUTO = "auto"
MODEL_TYPE_LEGACY = "legacy_classifier"
MODEL_TYPE_CONTRASTIVE = "contrastive_embedding"
MODEL_METRIC_AUTO = "auto"
MODEL_METRIC_PROBABILITY = "probability"
MODEL_METRIC_EUCLIDEAN = "euclidean"
MODEL_METRIC_COSINE = "cosine"


def resolve_path(path_like):
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def list_image_files(directory):
    directory = resolve_path(directory)
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def is_lfs_pointer(path_like):
    path = resolve_path(path_like)
    if not path.exists() or path.stat().st_size > 512:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def ensure_directory(path_like):
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def metadata_path_for_weights(path_like):
    path = resolve_path(path_like)
    return path.with_name(f"{path.name}.json")


def read_model_metadata(path_like):
    metadata_path = metadata_path_for_weights(path_like)
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_model_metadata(path_like, metadata):
    metadata_path = metadata_path_for_weights(path_like)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path
