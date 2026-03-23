import numpy as np

from utils import (
    MODEL_METRIC_COSINE,
    MODEL_METRIC_EUCLIDEAN,
    MODEL_METRIC_PROBABILITY,
    MODEL_TYPE_CONTRASTIVE,
    MODEL_TYPE_LEGACY,
)

EPSILON = 1e-12


def _flatten_embeddings(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings[np.newaxis, :]
    return embeddings.reshape((embeddings.shape[0], -1))


def euclidean_distances(left_embeddings, right_embeddings):
    left = _flatten_embeddings(left_embeddings)
    right = _flatten_embeddings(right_embeddings)
    return np.linalg.norm(left - right, axis=1)


def cosine_similarities(left_embeddings, right_embeddings):
    left = _flatten_embeddings(left_embeddings)
    right = _flatten_embeddings(right_embeddings)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = np.maximum(left_norm * right_norm, EPSILON)
    return np.sum(left * right, axis=1) / denominator


def metric_is_higher_better(metric):
    return metric in {MODEL_METRIC_PROBABILITY, MODEL_METRIC_COSINE}


def default_metric_for_model_type(model_type):
    if model_type == MODEL_TYPE_LEGACY:
        return MODEL_METRIC_PROBABILITY
    if model_type == MODEL_TYPE_CONTRASTIVE:
        return MODEL_METRIC_EUCLIDEAN
    raise ValueError(f"Unsupported model type: {model_type}")


def default_threshold_for(model_type, metric=None):
    metric = metric or default_metric_for_model_type(model_type)
    if metric == MODEL_METRIC_PROBABILITY:
        return 0.5
    if metric == MODEL_METRIC_EUCLIDEAN:
        return 0.75
    if metric == MODEL_METRIC_COSINE:
        return 0.8
    raise ValueError(f"Unsupported metric: {metric}")


def decisions_from_values(values, threshold, metric):
    values = np.asarray(values, dtype=np.float32)
    if metric_is_higher_better(metric):
        return values >= threshold
    return values <= threshold


def accuracy_from_values(values, targets, threshold, metric):
    targets = np.asarray(targets).astype(np.int32)
    predictions = decisions_from_values(values, threshold, metric).astype(np.int32)
    return float((predictions == targets).mean() * 100.0)


def find_best_threshold(values, targets, metric, steps=101):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets).astype(np.int32).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot search thresholds on an empty value array.")

    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        threshold = low
        return threshold, accuracy_from_values(values, targets, threshold, metric)

    candidate_thresholds = np.linspace(low, high, steps, dtype=np.float32)
    best_threshold = float(candidate_thresholds[0])
    best_accuracy = -1.0
    for threshold in candidate_thresholds:
        accuracy = accuracy_from_values(values, targets, float(threshold), metric)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold, best_accuracy
