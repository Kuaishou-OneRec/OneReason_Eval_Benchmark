"""
8-Component SID to Embedding Utilities

Converts 8-component SIDs (ad/live/video domains) to embeddings
using RQ-VAE centroids, and computes cosine similarity metrics.
"""

import os
import re
import numpy as np
from typing import Collection, List, Optional

import torch


class Sid2Emb:
    """Convert SID codes to embeddings using RQ-VAE centroids.

    Loads 8 models, each with 3 layers of centroids.
    For 8-component SIDs, uses the first layer of each model.
    """

    def __init__(self, model_dir: str):
        self.centroids_list = [[] for _ in range(8)]
        for model_idx in range(8):
            model_path = os.path.join(model_dir, f'model_{model_idx}', 'model.pt')
            data = torch.load(model_path, map_location='cpu')
            for layer_idx in range(3):
                centroids = data[f'centroids.{layer_idx}'].numpy()
                self.centroids_list[model_idx].append(centroids)

    def sid2emb(self, sid_codes: List[int]) -> Optional[np.ndarray]:
        """Convert 8-component SID to embeddings.

        Args:
            sid_codes: List of 8 integer codes (one per model).

        Returns:
            np.ndarray of shape (8, emb_dim), or None if codes are invalid.
        """
        if len(sid_codes) != 8:
            return None
        try:
            emb_res = []
            for i in range(8):
                emb_res.append(self.centroids_list[i][0][sid_codes[i]])
            return np.array(emb_res)
        except (IndexError, KeyError):
            return None


def extract_sid_codes(text: str) -> Optional[List[int]]:
    """Extract numeric codes from an 8-component SID string.

    Handles formats like <s_a_X><s_b_Y><s_c_Z><s_d_W><s_e_V><s_f_U><s_g_T><s_h_S>.

    Args:
        text: SID string (content between begin/end tokens).

    Returns:
        List of integer codes, or None if extraction fails.
    """
    # Strip thinking content
    if '</think>' in text:
        text = text.split('</think>')[-1].strip()

    # Normalize domain tokens
    for token in ["<|video_begin|>", "<|prod_begin|>", "<|living_begin|>", "<|ad_begin|>"]:
        text = text.replace(token, '')
    for token in ["<|video_end|>", "<|prod_end|>", "<|living_end|>", "<|ad_end|>"]:
        text = text.replace(token, '')

    pattern = r'<s_[a-z]_(\d+)>'
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return [int(m) for m in matches]


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embedding arrays.

    Flattens (8, emb_dim) arrays to 1D before computing.

    Args:
        emb1: First embedding array.
        emb2: Second embedding array.

    Returns:
        Cosine similarity in [-1, 1].
    """
    v1 = emb1.flatten()
    v2 = emb2.flatten()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_all_pred_scores(
    predicted_sids: List[str],
    gt_sids: Collection[str],
    sid2emb: Sid2Emb,
    max_k: int
) -> List[float]:
    """Compute per-prediction best similarity scores for top-max_k predictions.

    For each predicted SID in top-max_k:
      - Compute cosine similarity with every GT SID
      - Take the max as this prediction's score

    Results are ordered by prediction index. To get metrics for a smaller k,
    just slice the first k scores.

    Args:
        predicted_sids: List of extracted SID strings (already ordered by strategy).
        gt_sids: Collection of ground truth SID strings.
        sid2emb: Sid2Emb instance for converting codes to embeddings.
        max_k: Maximum number of predictions to consider.

    Returns:
        List of per-prediction scores (length <= max_k). Empty if no valid embeddings.
    """
    # Extract GT embeddings (done once)
    gt_embs = []
    for sid_text in gt_sids:
        codes = extract_sid_codes(sid_text)
        if codes is not None and len(codes) == 8:
            emb = sid2emb.sid2emb(codes)
            if emb is not None:
                gt_embs.append(emb)

    if not gt_embs:
        return []

    # Compute per-prediction scores for top-max_k
    top_k_gens = predicted_sids[:max_k]
    pred_scores = []

    for gen in top_k_gens:
        codes = extract_sid_codes(gen)
        if codes is None or len(codes) != 8:
            pred_scores.append(0.0)
            continue
        pred_emb = sid2emb.sid2emb(codes)
        if pred_emb is None:
            pred_scores.append(0.0)
            continue
        best_sim = max(compute_cosine_similarity(pred_emb, gt_emb) for gt_emb in gt_embs)
        pred_scores.append(best_sim)

    return pred_scores
