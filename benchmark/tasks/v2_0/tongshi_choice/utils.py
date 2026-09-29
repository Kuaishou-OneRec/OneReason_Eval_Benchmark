"""
Tongshi Choice Utilities

Functions for answer parsing (中文优化), accuracy computation (set 相等),
and debug information.

支持单选与多选：
- 单选答案存为 letters 字符串如 "A"
- 多选答案存为 letters 字符串如 "ABC"（升序）
- 评估按集合相等比对
"""

import re
from typing import Dict, Any, List


# Letter to index (兼容样例 JSON 中以索引存 answer 的写法)
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def _zh_choice_patterns(options: str) -> list:
    """
    生成中文 + 英文的答案模式列表。
    options = 形如 "ABCD" 的字母集合（每个一字符）。
    捕获组 1 必须是 letters，可能是单字母（A），也可能是连字符串（ABC）或带分隔（A、B、C）。

    LETTERS 子模式同时允许：
      - 单字母："A"
      - 紧凑多字母："ABC"
      - 带分隔多字母："A,B,C" / "A、B、C" / "A B C" / "A/B/C" / "A和B" / "A与B"
    """
    # 紧凑多字母 OR 带显式分隔符多字母
    letters_re = (
        f"([{options}](?:[{options}]|\\s*[、,，/和与]\\s*[{options}]|\\s+[{options}])*)"
    )
    L = letters_re

    return [
        # 中文强模式（参考 gpqa 英文版翻译）
        rf"(?:正确)?答案(?:应该)?(?:是|为|应为|应当是)\s*[：:]?\s*[\(（]?{L}[\)）]?",
        rf"最佳答案(?:是|为)\s*[：:]?\s*[\(（]?{L}[\)）]?",
        rf"正确(?:的)?(?:选项|答案)(?:是|为)\s*[：:]?\s*[\(（]?{L}[\)）]?",
        rf"故(?:本题)?(?:选|答案[选为是])\s*[：:]?\s*[\(（]?{L}[\)）]?",
        rf"(?:应|该|要)选\s*[\(（]?{L}[\)）]?",
        rf"答案[：:]\s*[\(（]?{L}[\)）]?",
        rf"本题(?:答案)?[选为是]\s*[\(（]?{L}[\)）]?",
        rf"【答案】\s*[\(（]?{L}[\)）]?",
        rf"综上(?:所述)?[，,]?\s*(?:正确答案|应选|选|答案)(?:是|为)?\s*[：:]?\s*[\(（]?{L}[\)）]?",
        rf"选择\s*[\(（]?{L}[\)）]?",
        rf"答\s*[：:]\s*[\(（]?{L}[\)）]?",

        # 英文模式（兼容混合语料）
        rf"(?i)ANSWER\s*[:：]\s*[\(（]?{L}[\)）]?",
        rf"[Tt]he\s+(?:correct\s+)?answer\s+is(?:\s+option)?\s*[:：]?\s*\(?{L}\)?",
        rf"[Tt]he\s+correct\s+answer\s+is\s*[:：]?.*?\\boxed{{{L}}}",
        rf"\\boxed{{{L}}}",
        rf"\*\*Final\s+Answer\*\*\s*[:：]\s*{L}",
        rf"Final\s+Answer\s*[:：]\s*{L}",
        rf"<answer>{L}</answer>",
        rf"<result>{L}</result>",
        rf"<final_answer>{L}</final_answer>",
        rf"(?i)answer\s+is\s+{L}",
        rf"(?i)correct\s+answer\s+is\s+{L}",
        rf"Answer\s*[:：]\s*{L}",

        # 兜底位置模式
        rf"(?:\s|^)[\(（]?{L}[\)）]?[\s。，,：:\.\$、]",
    ]


def _cushion_patterns(options: str) -> list:
    letters_re = (
        f"([{options}](?:[{options}]|\\s*[、,，/和与]\\s*[{options}]|\\s+[{options}])*)"
    )
    return [
        rf"{letters_re}\s*[:：]",
        rf"{letters_re}",
    ]


def _normalize_letters(raw: str, options: str) -> str:
    """从捕获到的字符串中提取字母 + 升序去重，返回字符串。如 'A、B、 C' -> 'ABC'。"""
    if not raw:
        return ""
    seen = set()
    keep = []
    for ch in raw.upper():
        if ch in options and ch not in seen:
            seen.add(ch)
            keep.append(ch)
    return "".join(sorted(keep))


def first_option_postprocess(text: str, options: str = "ABCD", cushion: bool = True) -> str:
    """
    解析模型回复中的选项字母（单选/多选都支持）。

    返回升序字母字符串，如 "A" 或 "ABC"；解析失败返回 ""。
    """
    if not text:
        return ""

    patterns = _zh_choice_patterns(options)
    if cushion:
        patterns = patterns + _cushion_patterns(options)

    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            captured = m.group(1) if m.group(1) is not None else m.group(0)
            letters = _normalize_letters(captured, options)
            if letters:
                return letters
    return ""


def parse_answer_from_response(response_str: str) -> str:
    """
    Parse answer letters (single or multi) from model response.

    Returns:
        Sorted unique letter string, e.g. "A" or "ABC"; empty on failure.
    """
    if not response_str:
        return ""
    return first_option_postprocess(response_str, options="ABCD", cushion=True)


def parse_ground_truth(raw: Any) -> str:
    """
    Normalize ground truth into sorted letter string.

    Accepts:
      - "A" / "ABC"
      - "A、B、C" / "A,B,C" / "A B C"
      - int index (0..3) -> single letter
      - list ["A","B"] -> "AB"
    """
    if raw is None:
        return ""
    # int / int-like string
    if isinstance(raw, int) or (isinstance(raw, str) and raw.isdigit()):
        return INDEX_TO_LETTER.get(int(raw), "")
    if isinstance(raw, (list, tuple, set)):
        joined = "".join(str(x) for x in raw)
        return _normalize_letters(joined, "ABCD")
    return _normalize_letters(str(raw), "ABCD")


def compute_accuracy(predicted: str, ground_truth: str) -> bool:
    """集合相等比较。"""
    if not predicted or not ground_truth:
        return False
    return set(predicted.upper()) == set(ground_truth.upper())


def compute_pass_at_k(predicted_list: List[str], ground_truth: str, k: int) -> bool:
    """top-k 中任一与 ground_truth 集合相等即通过。"""
    if not predicted_list or not ground_truth:
        return False
    return any(compute_accuracy(p, ground_truth) for p in predicted_list[:k])


def convert_index_to_letter(idx: Any) -> str:
    if idx is None:
        return ""
    try:
        return INDEX_TO_LETTER.get(int(idx), "")
    except (ValueError, TypeError):
        return ""


def get_debug_info(
    item: Dict[str, Any],
    predicted_answer: str,
    ground_truth_answer: str,
    is_correct: bool,
) -> Dict[str, Any]:
    return {
        "uuid": item.get("uuid", "unknown"),
        "predicted_answer": predicted_answer,
        "ground_truth_answer": ground_truth_answer,
        "is_correct": is_correct,
        "response": item.get("prediction", ""),
        "prompt": item.get("prompt", ""),
    }
