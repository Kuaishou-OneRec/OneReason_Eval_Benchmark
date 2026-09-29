"""
SID to Embedding Utilities (v3.1)

Converts 3-component SIDs (1 model x 3 layers RQ-VAE) to embeddings
using residual reconstruction, and computes cosine similarity metrics.

v3.1 uses TOKENS=1, LAYERS=3:
  - 1 model with 3 layers of centroids
  - Embedding reconstructed by summing centroids across all layers
  - Output: 3 SID codes per item (1 token x 3 layers)
"""

import os
import numpy as np
from typing import Collection, List, Optional

import torch

from benchmark.tasks.v2_0.recommendation.utils_8sid2emb import (
    extract_sid_codes,
    compute_cosine_similarity,
)


class Sid2Emb:
    """Convert SID codes to embeddings using RQ-VAE residual reconstruction.

    Loads num_tokens models, each with num_layers layers of centroids.
    Reconstructs embedding by summing centroids across all layers (residual).

    For v3.1: num_tokens=1, num_layers=3 -> 3 SID codes per item.
    """

    def __init__(self, model_dir: str, num_tokens: int = 1, num_layers: int = 3):
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_codes = num_tokens * num_layers
        self.centroids_list = [[] for _ in range(num_tokens)]

        for model_idx in range(num_tokens):
            model_path = os.path.join(model_dir, f'model_{model_idx}', 'model.pt')
            data = torch.load(model_path, map_location='cpu')
            for layer_idx in range(num_layers):
                centroids = data[f'centroids.{layer_idx}'].numpy()
                self.centroids_list[model_idx].append(centroids)

    def sid2emb(self, sid_codes: List[int]) -> Optional[np.ndarray]:
        """Convert SID codes to embedding via residual reconstruction.

        Reconstructs: emb = centroids_0[code_0] + centroids_1[code_1] + centroids_2[code_2]

        Args:
            sid_codes: List of integer codes (length = num_tokens * num_layers).

        Returns:
            np.ndarray of shape (emb_dim,), or None if codes are invalid.
        """
        if len(sid_codes) != self.num_codes:
            return None
        try:
            emb = np.zeros_like(self.centroids_list[0][0][0])
            code_idx = 0
            for model_idx in range(self.num_tokens):
                for layer_idx in range(self.num_layers):
                    emb = emb + self.centroids_list[model_idx][layer_idx][sid_codes[code_idx]]
                    code_idx += 1
            return emb
        except (IndexError, KeyError):
            return None


def compute_all_pred_scores(
    predicted_sids: List[str],
    gt_sids: Collection[str],
    sid2emb: Sid2Emb,
    max_k: int
) -> List[float]:
    """Compute per-prediction best similarity scores for top-max_k predictions.

    Same interface as utils_8sid2emb.compute_all_pred_scores but works with
    variable-length SID codes (determined by sid2emb.num_codes).

    Args:
        predicted_sids: List of extracted SID strings (already ordered by strategy).
        gt_sids: Collection of ground truth SID strings.
        sid2emb: Sid2Emb instance for converting codes to embeddings.
        max_k: Maximum number of predictions to consider.

    Returns:
        List of per-prediction scores (length <= max_k). Empty if no valid embeddings.
    """
    num_codes = sid2emb.num_codes

    # Extract GT embeddings (done once)
    gt_embs = []
    for sid_text in gt_sids:
        codes = extract_sid_codes(sid_text)
        if codes is not None and len(codes) == num_codes:
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
        if codes is None or len(codes) != num_codes:
            pred_scores.append(0.0)
            continue
        pred_emb = sid2emb.sid2emb(codes)
        if pred_emb is None:
            pred_scores.append(0.0)
            continue
        best_sim = max(compute_cosine_similarity(pred_emb, gt_emb) for gt_emb in gt_embs)
        pred_scores.append(best_sim)

    return pred_scores
