"""
Action-Logic evaluation module for logic chain comparison.

Compares ground-truth and generated logic chains by:
1. Aligning actions via DTW maximization over a similarity matrix.
2. Optionally evaluating logic entailment on aligned pairs via NLI.

Version B: Based on ref.py (production-verified) with robustness improvements,
stripped of training-specific code and adapted for the benchmark framework.

Key improvements over original action_logic.py:
  - ``_to_str()`` helper: handles None/list/dict values from model output
  - ``parse_chains()`` is more robust: handles non-dict items, double-serialized
    JSON, non-list events, ``<think>`` tag stripping
  - ``_split_sub_actions()`` handles non-string action types (list, int)
  - Fix #2: zero-score aligned pairs are filtered before NLI evaluation
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_HF_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


def default_local_nli_model_dir() -> str:
    """Return the default on-disk directory for the local NLI model."""
    repo_root = Path(__file__).resolve().parents[4]
    return str(repo_root / "models" / "nli-deberta-v3-base")


def resolve_nli_model_name(
    model_name: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> str:
    """Resolve the NLI model identifier, preferring an existing local directory."""
    env_model_dir = os.environ.get("NLI_MODEL_DIR")
    env_model_name = os.environ.get("NLI_MODEL")

    if model_dir:
        return model_dir
    if env_model_dir:
        return env_model_dir

    default_local_dir = default_local_nli_model_dir()
    if os.path.isdir(default_local_dir):
        return default_local_dir

    if model_name:
        return model_name
    if env_model_name:
        return env_model_name
    return _DEFAULT_HF_NLI_MODEL

# ---------------------------------------------------------------------------
# 1. Data structures & parsing (from ref.py, with robustness improvements)
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single event in a logic chain."""
    action: str
    logic: str
    date: str = ""


@dataclass
class LogicChain:
    """A named sequence of events forming a logic chain."""
    name: str
    events: List[Event] = field(default_factory=list)


def _to_str(val) -> str:
    """Coerce a value to str. Lists are joined with '；', None becomes ''."""
    if val is None:
        return ""
    if isinstance(val, list):
        return "；".join(str(v) for v in val)
    if isinstance(val, dict):
        return str(val)
    return str(val)


def parse_chains(json_str) -> List[LogicChain]:
    """Parse a JSON string into a list of LogicChain objects.

    Supports two formats:
    - Wrapped: ``[{"logic_chain": {"name": ..., "events": [...]}}]``
    - Flat:    ``[{"name": ..., "events": [...]}]``

    Markdown code fences (```json ... ```) and ``<think>...</think>``
    blocks are stripped automatically. Handles non-string input,
    double-serialized JSON, and non-dict items gracefully.

    Args:
        json_str: Raw JSON string (possibly with markdown fences).

    Returns:
        List of parsed LogicChain objects.
    """
    # Coerce non-string input
    if not isinstance(json_str, str):
        if json_str is None:
            return []
        json_str = str(json_str)

    # Strip <think>...</think> blocks (reasoning models)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", json_str).strip()

    # Strip markdownde fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)
    if not isinstance(data, list):
        data = [data]

    chains: List[LogicChain] = []
    for item in data:
        # Handle nested wrapping
        if isinstance(item, dict) and "logic_chain" in item:
            item = item["logic_chain"]

        # Skip non-dict items (model may output strings or other types)
        if isinstance(item, str):
            # Try to parse as JSON in case of double-serialized data
            try:
                item = json.loads(item)
            except (json.JSONDecodeError, ValueError):
                continue
        if not isinstance(item, dict):
            continue

        # Normalize events: must be a list
        raw_events = item.get("events", [])
        if not isinstance(raw_events, list):
            raw_events = [raw_events] if isinstance(raw_events, dict) else []

        events = [
            Event(
                action=_to_str(e.get("action", "")),
                logic=_to_str(e.get("logic", "")),
                date=_to_str(e.get("date", "")),
            )
            for e in raw_events
            if isinstance(e, dict)
        ]
        chains.append(LogicChain(name=_to_str(item.get("name", "")), events=events))

    return chains


# ---------------------------------------------------------------------------
# 2. Action similarity (exact match on sub-actions)
# ---------------------------------------------------------------------------

def _split_sub_actions(action) -> List[str]:
    """Split an action string by ``；``、``/`` or ``、`` and propagate the action-type
    prefix (e.g. ``搜索：``) to sub-actions that lack one.

    Handles non-string types (model may output list, int, etc.).

    Example::

        "搜索：A；B"  →  ["搜索：A", "搜索：B"]
        "搜索：A；观看视频：B"  →  ["搜索：A", "观看视频：B"]
    """
    # Handle non-string types (model may output list, int, etc.)
    if isinstance(action, list):
        return [str(a).strip() for a in action if a]
    if not isinstance(action, str):
        action = str(action)

    parts = re.split(r"[；/、]", action)
    parts = [s.strip() for s in parts if s.strip()]
    if not parts:
        return []

    # Detect prefix from the first sub-action (text before first ：)
    prefix = ""
    first = parts[0]
    colon_idx = first.find("：")
    if colon_idx != -1:
        prefix = first[: colon_idx + 1]

    # Propagate prefix to sub-actions that don't have their own ：
    result = [first]
    for part in parts[1:]:
        if "：" not in part and prefix:
            result.append(prefix + part)
        else:
            result.append(part)

    # Strip scene tags like [视频-长播], [搜索-搜索] etc.
    result = [re.sub(r"^\[.*?\]\s*", "", s) for s in result]
    return result


def action_recall(action_gt: str, action_gen: str) -> float:
    """Compute recall of gt sub-actions matched exactly in gen sub-actions.

    Each action string is split by ``；``、``/`` or ``、`` into sub-actions.
    A gt sub-action counts as matched if it appears exactly in the gen
    sub-action list. Returns matched_count / len(gt sub-actions).
    """
    subs_gt = _split_sub_actions(action_gt)
    subs_gen = _split_sub_actions(action_gen)
    if not subs_gt or not subs_gen:
        return 0.0
    matched = sum(1 for sg in subs_gt if sg in subs_gen)
    return matched / len(subs_gt)


def action_score_matrix(
    actions_gt: List[str], actions_gen: List[str]
) -> List[List[float]]:
    """Build an |gt| x |gen| recall matrix."""
    return [
        [action_recall(gt, gen) for gen in actions_gen]
        for gt in actions_gt
    ]


# ---------------------------------------------------------------------------
# 2b. Bigram F1 soft matching + action_f1 (aligned with logic_chain_sid.py)
# ---------------------------------------------------------------------------

def _char_bigrams(text: str) -> List[str]:
    """Extract character-level bigrams from text (whitespace removed)."""
    text = re.sub(r"\s+", "", text)
    return [text[i:i + 2] for i in range(len(text) - 1)]


def _bigram_f1(text_a: str, text_b: str) -> float:
    """Compute character-bigram F1 between two strings.

    Returns 1.0 for exact match, 0.0 for no overlap.
    Falls back to exact match for very short strings (< 2 chars).
    """
    ng_a = _char_bigrams(text_a)
    ng_b = _char_bigrams(text_b)
    if not ng_a or not ng_b:
        return 1.0 if text_a.strip() == text_b.strip() else 0.0

    counter_a: Dict[str, int] = {}
    for ng in ng_a:
        counter_a[ng] = counter_a.get(ng, 0) + 1
    counter_b: Dict[str, int] = {}
    for ng in ng_b:
        counter_b[ng] = counter_b.get(ng, 0) + 1

    overlap = sum(min(count, counter_b.get(ng, 0)) for ng, count in counter_a.items())
    if overlap == 0:
        return 0.0

    precision = overlap / len(ng_b)
    recall = overlap / len(ng_a)
    return 2 * precision * recall / (precision + recall)


def _safe_f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall, safe for zero inputs."""
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def action_f1(action_gt: str, action_gen: str) -> float:
    """Compute F1 of sub-action matching using exact match.

    Same exact-match logic as action_recall, but returns F1 instead of
    pure recall to penalise extra generated sub-actions.
    """
    subs_gt = _split_sub_actions(action_gt)
    subs_gen = _split_sub_actions(action_gen)
    if not subs_gt or not subs_gen:
        return 0.0
    matched = sum(1 for sg in subs_gt if sg in subs_gen)
    recall = matched / len(subs_gt)
    precision = matched / len(subs_gen)
    return _safe_f1(precision, recall)


def action_f1_score_matrix(
    actions_gt: List[str], actions_gen: List[str]
) -> List[List[float]]:
    """Build an |gt| x |gen| action F1 matrix (penalises over-generation)."""
    return [
        [action_f1(gt, gen) for gen in actions_gen]
        for gt in actions_gt
    ]


# ---------------------------------------------------------------------------
# 2c. Logic text similarity (TokenF1 + ROUGE-L, no NLI model needed)
# ---------------------------------------------------------------------------

_LOGIC_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _logic_tokenize(text: str) -> List[str]:
    """Tokenize short logic text with a lightweight CJK/alnum tokenizer."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    if not normalized:
        return []
    tokens = _LOGIC_TOKEN_RE.findall(normalized)
    if tokens:
        return tokens
    return [ch for ch in normalized if not ch.isspace()]


def _logic_token_f1(reference: str, candidate: str) -> float:
    """Clipped token F1 between reference and candidate logic text."""
    ref_tokens = _logic_tokenize(reference)
    cand_tokens = _logic_tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_counter = Counter(ref_tokens)
    cand_counter = Counter(cand_tokens)
    overlap = sum(min(ref_counter[token], cand_counter[token]) for token in ref_counter)
    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    return _safe_f1(precision, recall)


def _lcs_length(tokens_a: List[str], tokens_b: List[str]) -> int:
    """Compute longest common subsequence length."""
    if not tokens_a or not tokens_b:
        return 0
    prev = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        curr = [0] * (len(tokens_b) + 1)
        for j, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _logic_rouge_l_f1(reference: str, candidate: str) -> float:
    """ROUGE-L F1 based on the longest common subsequence."""
    ref_tokens = _logic_tokenize(reference)
    cand_tokens = _logic_tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, cand_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    return _safe_f1(precision, recall)


def _logic_similarity(reference: str, candidate: str) -> Tuple[float, float, float]:
    """Return (mixed_score, token_f1, rouge_l_f1) for logic text."""
    token_f1 = _logic_token_f1(reference, candidate)
    rouge_l_f1 = _logic_rouge_l_f1(reference, candidate)
    mixed = 0.5 * token_f1 + 0.5 * rouge_l_f1
    return mixed, token_f1, rouge_l_f1


# ---------------------------------------------------------------------------
# 3. DTW maximisation alignment
# ---------------------------------------------------------------------------

def dtw_max_alignment(
    sim_matrix: List[List[float]],
) -> Tuple[float, List[Tuple[int, int]]]:
    """Find the maximum-weight ordered matching between two event sequences.

    This is similar to a weighted LCS: find a subset of 1-to-1 pairs
    ``(i, j)`` such that both index sequences are strictly increasing and the
    sum of ``sim_matrix[i][j]`` over all selected pairs is maximised.

    DP recurrence (1-indexed, ``F[0][*] = F[*][0] = 0``)::

        F[i][j] = max(
            F[i-1][j],                    # skip gt event i
            F[i][j-1],                    # skip gen event j
            F[i-1][j-1] + S[i-1][j-1],   # match gt[i] with gen[j]
        )

    Args:
        sim_matrix: ``n_gt x n_gen`` similarity matrix (values in [0, 1]).

    Returns:
        ``(total_score, aligned_pairs)``
        - total_score: cumulative similarity of the best ordered matching.
        - aligned_pairs: list of ``(i, j)`` index pairs (0-based) that form
          the optimal matching.
    """
    n_gt = len(sim_matrix)
    if n_gt == 0:
        return 0.0, []
    n_gen = len(sim_matrix[0])
    if n_gen == 0:
        return 0.0, []

    F = [[0.0] * (n_gen + 1) for _ in range(n_gt + 1)]
    B = [[0] * (n_gen + 1) for _ in range(n_gt + 1)]

    for i in range(1, n_gt + 1):
        for j in range(1, n_gen + 1):
            match_val = F[i - 1][j - 1] + sim_matrix[i - 1][j - 1]
            skip_gt = F[i - 1][j]
            skip_gen = F[i][j - 1]

            best = max(match_val, skip_gt, skip_gen)
            F[i][j] = best
            if best == match_val:
                B[i][j] = 0
            elif best == skip_gt:
                B[i][j] = 1
            else:
                B[i][j] = 2

    total_score = F[n_gt][n_gen]

    # Backtrack to recover aligned pairs
    aligned_pairs: List[Tuple[int, int]] = []
    i, j = n_gt, n_gen
    while i > 0 and j > 0:
        if B[i][j] == 0:
            aligned_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif B[i][j] == 1:
            i -= 1
        else:
            j -= 1
    aligned_pairs.reverse()

    return total_score, aligned_pairs


# ---------------------------------------------------------------------------
# 4. NLIBackend ABC + implementations
# ---------------------------------------------------------------------------

class NLIBackend(ABC):
    """Abstract base class for Natural Language Inference backends."""

    @abstractmethod
    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Return probability distribution over {entailment, neutral, contradiction}."""

    def predict_batch(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Dict[str, float]]:
        """Run NLI on a batch of (premise, hypothesis) pairs."""
        return [self.predict(p, h) for p, h in pairs]


class HuggingFaceNLI(NLIBackend):
    """NLI via a HuggingFace cross-encoder model (lazy-loads transformers & torch).

    Label order (from config.json id2label):
        index 0 → contradiction
        index 1 → entailment
        index 2 → neutral
    """

    _CANONICAL = {"entailment", "neutral", "contradiction"}

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self.model_name = model_name
        self._cross_encoder = None
        self._tokenizer = None
        self._model = None
        self._label_order: List[str] = []
        self._use_cross_encoder: bool = False
        self._is_local_path = os.path.isdir(self.model_name)

    def _ensure_model(self) -> None:
        if self._cross_encoder is not None or self._model is not None:
            return

        # Try CrossEncoder first (model card's recommended approach)
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(
                self.model_name,
                model_kwargs={"torch_dtype": "bfloat16"},
            )
            self._use_cross_encoder = True
            # Label order per model card & config.json
            self._label_order = ["contradiction", "entailment", "neutral"]
            logger.info(f"NLI model loaded via CrossEncoder (bf16): {self.model_name}")
            return
        except ImportError:
            logger.info(
                "sentence_transformers not installed, "
                "falling back to transformers AutoModel for NLI"
            )

        # Fallback: AutoModelForSequenceClassification
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self._is_local_path,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            local_files_only=self._is_local_path,
        )
        self._model.eval()
        id2label = self._model.config.id2label
        self._label_order = [
            id2label[i].lower() for i in range(len(id2label))
        ]
        logger.info(f"NLI model loaded via AutoModel: {self.model_name}")

    @staticmethod
    def _softmax(logits: List[float]) -> List[float]:
        max_v = max(logits)
        exps = [math.exp(v - max_v) for v in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Run NLI inference for a single pair."""
        self._ensure_model()

        if self._use_cross_encoder:
            scores = self._cross_encoder.predict([(premise, hypothesis)])
            logits = scores[0].tolist() if hasattr(scores[0], 'tolist') else list(scores[0])
            probs = self._softmax(logits)
            return dict(zip(self._label_order, probs))

        # AutoModel fallback
        import torch
        inputs = self._tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits[0].tolist()
        probs = self._softmax(logits)
        return dict(zip(self._label_order, probs))

    def predict_batch(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Dict[str, float]]:
        """Batch NLI inference with padding."""
        if not pairs:
            return []
        self._ensure_model()

        if self._use_cross_encoder:
            scores = self._cross_encoder.predict(list(pairs))
            results = []
            for row in scores:
                logits = row.tolist() if hasattr(row, 'tolist') else list(row)
                probs = self._softmax(logits)
                results.append(dict(zip(self._label_order, probs)))
            return results

        # AutoModel fallback
        import torch
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        inputs = self._tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits.tolist()
        results = []
        for row in logits:
            probs = self._softmax(row)
            results.append(
                {label: prob for label, prob in zip(self._label_order, probs)}
            )
        return results


class LLMNLIBackend(NLIBackend):
    """NLI via the benchmark's LLM API (uses api.get_client_from_config)."""

    _PROMPT_TEMPLATE = (
        "请判断以下两段逻辑描述之间的关系。\n\n"
        "前提 (Premise):\n{premise}\n\n"
        "假设 (Hypothesis):\n{hypothesis}\n\n"
        "请从以下三种关系中选择一种：\n"
        "- entailment: 前提蕴含假设（假设可以从前提中推导出来）\n"
        "- neutral: 前提与假设无直接推导关系\n"
        "- contradiction: 前提与假设矛盾\n\n"
        '请用JSON格式输出：{{"label": "entailment/neutral/contradiction", "confidence": 0.0-1.0}}\n'
        "只输出JSON，不要输出其他内容。"
    )

    def __init__(self, client=None, judge_model: str = "gemini"):
        self._client = client
        self._judge_model = judge_model
        if self._client is None:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize LLM client using the benchmark's api module."""
        from api import get_client_from_config
        self._client = get_client_from_config(self._judge_model)

    @staticmethod
    def _parse_response(text: str) -> Dict[str, Any]:
        """Parse LLM response into {"label": ..., "confidence": ...}."""
        text_clean = text.strip()
        text_clean = re.sub(r"^```(?:json)?\s*", "", text_clean)
        text_clean = re.sub(r"\s*```$", "", text_clean)

        try:
            obj = json.loads(text_clean)
            label = obj.get("label", "neutral").lower().strip()
            confidence = float(obj.get("confidence", 0.5))
            return {"label": label, "confidence": confidence}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        text_lower = text.lower()
        if "entailment" in text_lower:
            label = "entailment"
        elif "contradiction" in text_lower:
            label = "contradiction"
        else:
            label = "neutral"
        return {"label": label, "confidence": 0.5}

    @staticmethod
    def _to_distribution(label: str, confidence: float) -> Dict[str, float]:
        """Convert a label + confidence into a probability distribution."""
        confidence = max(0.0, min(1.0, confidence))
        remaining = 1.0 - confidence
        dist = {
            "entailment": remaining / 2,
            "neutral": remaining / 2,
            "contradiction": remaining / 2,
        }
        dist[label] = confidence
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()}

    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Query the LLM for an NLI judgement."""
        prompt = self._PROMPT_TEMPLATE.format(premise=premise, hypothesis=hypothesis)
        try:
            response = self._client.generate(prompt, temperature=0.01, max_tokens=256)
            parsed = self._parse_response(response)
            return self._to_distribution(parsed["label"], parsed["confidence"])
        except Exception as e:
            logger.warning("LLM NLI call failed: %s. Returning uniform distribution.", e)
            return {"entailment": 1 / 3, "neutral": 1 / 3, "contradiction": 1 / 3}


# ---------------------------------------------------------------------------
# 5. NLI factory
# ---------------------------------------------------------------------------

_NLI_MAP = {
    "huggingface": HuggingFaceNLI,
    "llm": LLMNLIBackend,
}


def build_nli(name: Optional[str], **kwargs) -> Optional[NLIBackend]:
    """Build an NLI backend by name.

    Args:
        name: Backend name ("huggingface" or "llm"), or None to skip NLI.
        **kwargs: Arguments forwarded to the backend constructor.

    Returns:
        NLI backend instance, or None if name is None.
    """
    if name is None:
        return None
    if name not in _NLI_MAP:
        raise ValueError(f"Unknown NLI backend: {name!r}. Choose from {list(_NLI_MAP)}")
    return _NLI_MAP[name](**kwargs)


# ---------------------------------------------------------------------------
# 6. Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_action_logic(
    gt: str,
    generated: str,
    nli_backend: Optional[NLIBackend] = None,
) -> dict:
    """Evaluate a generated logic chain against ground-truth.

    Both inputs are JSON strings containing a single logic chain. The first
    chain in each list is used for comparison.

    Returns:
        Dictionary with:
        - ``action_f1_score``: float in [0, 1], F1 using bigram soft matching.
        - ``logic_f1_score``: float in [0, 1], F1 of TokenF1+ROUGE-L text similarity.
        - ``logic_nli_distribution``: dict or None, mean {entailment, neutral, contradiction}.
        - ``logic_entailment_rate``: float or None.
        - ``num_gt_events``, ``num_gen_events``: counts.
    """
    gt_chain = parse_chains(gt)[0]
    gen_chain = parse_chains(generated)[0]

    gt_actions = [e.action for e in gt_chain.events]
    gen_actions = [e.action for e in gen_chain.events]

    # --- action_f1_score: bigram F1 soft matching ---
    action_f1_score_val = 0.0
    f1_aligned_pairs: List[Tuple[int, int]] = []
    if gt_actions and gen_actions:
        f1_sim_mat = action_f1_score_matrix(gt_actions, gen_actions)
        f1_total, f1_aligned_pairs = dtw_max_alignment(f1_sim_mat)
        f1_recall = f1_total / len(gt_actions)
        f1_precision = f1_total / len(gen_actions)
        action_f1_score_val = _safe_f1(f1_precision, f1_recall)

    # --- logic_f1_score: text similarity F1 on action_f1 aligned pairs ---
    logic_f1_score_val = 0.0
    if gt_actions and gen_actions and f1_aligned_pairs:
        logic_total = 0.0
        for i, j in f1_aligned_pairs:
            mixed, _, _ = _logic_similarity(
                gt_chain.events[i].logic,
                gen_chain.events[j].logic,
            )
            logic_total += mixed
        logic_recall = logic_total / len(gt_actions)
        logic_precision = logic_total / len(gen_actions)
        logic_f1_score_val = _safe_f1(logic_precision, logic_recall)

    # --- NLI on action_f1 aligned logic pairs ---
    logic_nli_distribution: Optional[Dict[str, float]] = None
    logic_entailment_rate: Optional[float] = None
    if nli_backend is not None and f1_aligned_pairs:
        nli_pairs = [
            (gt_chain.events[i].logic, gen_chain.events[j].logic)
            for i, j in f1_aligned_pairs
        ]
        nli_results = nli_backend.predict_batch(nli_pairs)
        keys = ["entailment", "neutral", "contradiction"]
        logic_nli_distribution = {
            k: sum(d.get(k, 0.0) for d in nli_results) / len(nli_results)
            for k in keys
        }
        logic_entailment_rate = logic_nli_distribution["entailment"]

    return {
        "action_f1_score": action_f1_score_val,
        "logic_f1_score": logic_f1_score_val,
        "logic_nli_distribution": logic_nli_distribution,
        "logic_entailment_rate": logic_entailment_rate,
        "num_gt_events": len(gt_actions),
        "num_gen_events": len(gen_actions),
    }
