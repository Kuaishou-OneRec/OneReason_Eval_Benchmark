"""
Evolution Select Utilities
"""

import ast
import json
from typing import Dict, List

from benchmark.tasks.v2_0.evolution_topic_gen.utils import extract_json_from_response


def normalize_items(items) -> List[str]:
    """Normalize item ids / search texts into a deduplicated ordered list."""
    if not isinstance(items, list):
        return []

    normalized = []
    seen = set()
    for item in items:
        if item is None:
            continue
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def parse_ground_truth_items(ground_truth) -> List[str]:
    """Parse ground truth serialized by BaseLoader from metadata.answer."""
    if isinstance(ground_truth, list):
        return normalize_items(ground_truth)

    if ground_truth is None:
        return []

    text = str(ground_truth).strip()
    if not text:
        return []

    try:
        value = ast.literal_eval(text)
        if isinstance(value, list):
            return normalize_items(value)
    except (SyntaxError, ValueError):
        pass

    try:
        value = json.loads(text)
        if isinstance(value, list):
            return normalize_items(value)
    except (json.JSONDecodeError, TypeError):
        pass

    return []


def parse_prediction_items(response_text: str) -> List[str]:
    """Parse model output JSON array into a normalized list."""
    cleaned = extract_json_from_response(response_text or "")
    if cleaned is None:
        return []

    try:
        value = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return []

    return normalize_items(value)


def compute_set_metrics(predicted: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """Compute precision / recall / F1 using set overlap."""
    pred_set = set(normalize_items(predicted))
    gt_set = set(normalize_items(ground_truth))
    hits = len(pred_set & gt_set)

    precision = hits / len(pred_set) if pred_set else 0.0
    recall = hits / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_count": hits,
        "pred_count": len(pred_set),
        "gt_count": len(gt_set),
    }
