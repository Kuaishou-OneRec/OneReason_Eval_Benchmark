"""
Evolution Topic Gen Utilities

Functions for response parsing, score computation, and debug information.
"""

import json
import re
from typing import Dict, Any, Optional


def extract_json_from_response(text: str) -> Optional[str]:
    """Extract JSON string from model response.

    Handles:
    - Pure JSON output
    - JSON wrapped in markdown code fences (```json ... ```)
    - JSON embedded in surrounding text

    Args:
        text: Raw model output text

    Returns:
        Extracted JSON string, or None if extraction fails
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Try 1: Strip markdown code fences and parse directly
    cleaned = re.sub(r"^```(?:json)?\s*", "", text)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        json.loads(cleaned)
        return cleaned
    except (json.JSONDecodeError, ValueError):
        pass

    # Try 2: Find JSON object in the text (greedy match from first { to last })
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    # Try 3: Find JSON array in the text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def compute_combined_f1_score(
    action_f1_score: float,
    logic_f1_score: float,
) -> float:
    """Compute combined F1 score aligned with RL training (rule_mix_v2).

    score = 0.5 * action_f1_score + 0.5 * logic_f1_score

    Args:
        action_f1_score: Action F1 score (bigram soft matching) in [0, 1]
        logic_f1_score: Logic F1 score (TokenF1 + ROUGE-L) in [0, 1]

    Returns:
        Combined F1 score in [0, 1]
    """
    return 0.5 * action_f1_score + 0.5 * logic_f1_score


def get_debug_info(
    sample_id: str,
    ground_truth: str,
    generation: str,
    eval_result: Dict[str, Any],
    parse_failed: bool = False,
) -> Dict[str, Any]:
    """Generate debug information for a single sample.

    Args:
        sample_id: Sample identifier
        ground_truth: Ground truth JSON string
        generation: Model generation text
        eval_result: Evaluation result from evaluate_action_logic
        parse_failed: Whether JSON parsing failed

    Returns:
        Debug information dictionary
    """
    return {
        "sample_id": sample_id,
        "parse_failed": parse_failed,
        "ground_truth": ground_truth,
        "generation": generation,
        "logic_entailment_rate": eval_result.get("logic_entailment_rate"),
        "logic_nli_distribution": eval_result.get("logic_nli_distribution"),
        "num_gt_events": eval_result.get("num_gt_events", 0),
        "num_gen_events": eval_result.get("num_gen_events", 0),
    }
