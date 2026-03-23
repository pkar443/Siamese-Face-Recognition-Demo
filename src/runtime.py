import os


def configure_device(device="auto"):
    """Select CPU or GPU before TensorFlow is imported elsewhere."""
    normalized = device.lower()
    if normalized not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be one of: auto, cpu, gpu")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if normalized == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if normalized == "gpu" and not gpus:
        raise RuntimeError("GPU was requested but TensorFlow could not see any GPU.")

    selected = "gpu" if gpus and normalized != "cpu" else "cpu"
    return tf, selected, [gpu.name for gpu in gpus]
