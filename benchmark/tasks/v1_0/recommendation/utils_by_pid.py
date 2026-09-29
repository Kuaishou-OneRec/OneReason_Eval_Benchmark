"""
Recommendation Task Utilities (PID-based)

Functions for PID extraction and recommendation metrics computation using PIDs.
"""

import re
import json
import random
from typing import Set, Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import Counter


# Encoding constant: each code is in range [0, 8192]
BASE = 8192


def load_pid_mapping(mapping_path: str) -> Dict:
    """
    Load SID to PID mapping from JSON file.
    Auto-detects key encoding format:
      - dot_join: keys stay as str (e.g., "123.456.789")
      - multiply: keys converted to int (e.g., 8454660)

    Args:
        mapping_path: Path to the JSON file containing SID to PID mapping

    Returns:
        Dictionary mapping encoded SID (int or str) to list of PID info dictionaries
        Format: {encoded_sid: [{"pid": pid1, "count": count1, "count_after_downsample": count2}, ...]}
        PIDs are sorted by original count in descending order
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"PID mapping file not found: {mapping_path}")

    with open(mapping_path, 'r') as f:
        sid_to_pid_json = json.load(f)

    encoding = detect_sid_encoding(sid_to_pid_json.keys())

    if encoding == "dot_join":
        code_to_pid = sid_to_pid_json  # keys stay as strings "123.456.789"
    else:
        code_to_pid = {int(k): v for k, v in sid_to_pid_json.items()}

    print(f"[INFO] Loaded {len(code_to_pid)} SID mappings (encoding={encoding}) from {mapping_path}")
    return code_to_pid


def encode_sid(*codes) -> int:
    """
    Encode SID codes into a single integer key.
    Supports any number of codes (3-token or 8-token).

    Examples:
        encode_sid(c1, c2, c3)          # 3-token
        encode_sid(*eight_codes)         # 8-token

    Returns:
        Encoded integer key
    """
    sid = 0
    for c in codes:
        sid = sid * BASE + int(c)
    return sid


def encode_sid_dot(*codes) -> str:
    """
    Encode SID codes into a dot-joined string key.
    Examples:
        encode_sid_dot(123, 456, 789) -> "123.456.789"
    """
    return ".".join(str(int(c)) for c in codes)


def detect_sid_encoding(mapping_keys) -> str:
    """
    Auto-detect the SID encoding format from mapping file keys.
    Returns:
        "dot_join" if keys contain dots (e.g., "123.456.789")
        "multiply" if keys are pure integers (e.g., "8454660")
    """
    for key in mapping_keys:
        if "." in str(key):
            return "dot_join"
        return "multiply"
    return "multiply"


def encode_sid_auto(codes, encoding):
    """
    Encode SID codes using the specified encoding method.
    Args:
        codes: Tuple/list of SID code values
        encoding: "multiply", "dot_join", or a callable
    Returns:
        int (for multiply), str (for dot_join), whatever the callable returns,
        or None if a generated SID cannot be encoded by the selected scheme
    """
    try:
        if callable(encoding):
            return encoding(codes)
        if encoding == "dot_join":
            return encode_sid_dot(*codes)
        return encode_sid(*codes)
    except (IndexError, TypeError, ValueError):
        return None


def extract_sid_codes_from_text(text: str, sid_pattern: str) -> Optional[Tuple[int, ...]]:
    """
    Extract SID codes from text using regex pattern.
    The sid_pattern must be provided and its capture groups define the SID codes.

    Args:
        text: Text containing SID tokens
        sid_pattern: Regex pattern whose capture groups form the SID tuple

    Returns:
        Tuple of extracted SID codes, or None if not found
    """
    match = re.search(sid_pattern, text)
    if not match or not match.groups():
        return None
    try:
        return tuple(int(x) for x in match.groups())
    except (TypeError, ValueError):
        return None


def _get_id_from_info(info: Dict[str, int]) -> int:
    """
    Extract ID from info dict, supporting both 'pid' and 'iid' keys.

    Args:
        info: Dictionary containing either 'pid' or 'iid' key

    Returns:
        The ID value (int)
    """
    return info.get("pid", info.get("iid", 0))


def apply_sid_to_pid_strategy(pid_info_list: List[Dict[str, int]], strategy: str) -> int:
    """
    Apply strategy to select a single PID from a list

    Args:
        pid_info_list: List of PID info dictionaries
                      Format: [{"pid": pid1, "count": count1, "count_after_downsample": count2}, ...]
                      or [{"iid": iid1, "count": count1, "count_after_downsample": count2}, ...] for product
        strategy: One of "most_popular_originally", "most_popular_after_downsampling", "random", or "max_pid"

    Returns:
        Selected PID/IID (int), or 0 if list is empty

    Strategies:
        - "most_popular_originally": Return the PID with highest original count (already sorted)
        - "most_popular_after_downsampling": Return the PID with highest downsampled count (random if tie)
        - "random": Randomly select one PID from the list
        - "max_pid": Return the PID with the largest numeric value
    """
    if not pid_info_list:
        return 0

    if strategy == "most_popular_originally":
        # Return the first PID/IID (highest original count, already sorted)
        return _get_id_from_info(pid_info_list[0])
    elif strategy == "most_popular_after_downsampling":
        # Find max downsampled count
        max_count = max(info["count_after_downsample"] for info in pid_info_list)
        # Get all PIDs/IIDs with max downsampled count
        max_pids = [_get_id_from_info(info) for info in pid_info_list if info["count_after_downsample"] == max_count]
        # Randomly select one if there are ties
        return random.choice(max_pids)
    elif strategy == "random":
        # Randomly select a PID/IID
        return random.choice([_get_id_from_info(info) for info in pid_info_list])
    elif strategy == "max_pid":
        # Return the PID/IID with the largest numeric value
        return max(_get_id_from_info(info) for info in pid_info_list)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'most_popular_originally', 'most_popular_after_downsampling', or 'random'")


def extract_ids_from_answer(answer: List[int]) -> Set[int]:
    """
    Extract all PIDs from answer field (metadata["answer_pid"]) or (metadata["answer_iid"])

    Examples:
        >>> extract_ids_from_answer([123, 456, 789])
        {123, 456, 789}
    """
    return set([pid for pid in answer if pid != 0])


def extract_first_id_from_answer(answer: List[int]) -> int:
    """
    Extract the first PID from answer field

    Examples:
        >>> extract_first_id_from_answer([123, 456, 789])
        123
    """
    valid_pids = [pid for pid in answer if pid != 0]
    return valid_pids[0] if valid_pids else 0


def extract_id_from_generation(
    generation: str,
    code_to_pid: Dict,
    strategy: str = "most_popular_originally",
    encoding: str = "multiply",
    sid_pattern: Optional[str] = None
) -> int:
    """
    Extract PID from model generation

    The generation may contain:
    - SID wrapped in tags: "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"
    - With thinking: "<think>...</think>\\n<|sid_begin|><s_a_1>..."

    Args:
        generation: Model generation string (contains exactly one SID)
        code_to_pid: Mapping dictionary {encoded_sid: [{"pid": pid, "count": ..., "count_after_downsample": ...}, ...]}
        strategy: Strategy for selecting PID ("most_popular_originally", "most_popular_after_downsampling", or "random")
        encoding: SID encoding method ("multiply" or "dot_join")
        sid_pattern: Optional custom regex whose capture groups form the SID tuple

    Returns:
        Extracted PID (int), or 0 if not found

    Examples:
        >>> extract_id_from_generation("<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>", code_to_pid)
        12345  # Assuming this SID maps to PID 12345
    """
    generation = generation.strip()

    # If generation contains </think>, only process content after it
    if '</think>' in generation:
        generation = generation.split('</think>')[-1].strip()

    # Extract SID codes from the generation (should be exactly one)
    sid_codes = extract_sid_codes_from_text(generation, sid_pattern=sid_pattern)

    if sid_codes is None:
        return 0

    # Encode SID and look up PID list
    encoded = encode_sid_auto(sid_codes, encoding)
    if encoded is None:
        return 0
    pid_freq_list = code_to_pid.get(encoded, [])

    # Apply strategy to select PID
    return apply_sid_to_pid_strategy(pid_freq_list, strategy)




def compute_pass_at_k(
    predicted_ids: List[int],
    ground_truth_ids: Set[int],
    k: int
) -> bool:
    """
    Compute Pass@k for a single sample using PIDs

    Pass@k definition:
    - Take the first k candidate PIDs from predictions
    - If any of these k PIDs appears in the ground truth PIDs, return True

    Args:
        predicted_ids: List of predicted PIDs (already extracted from generations)
        ground_truth_ids: Set of ground truth PIDs
        k: Number of top predictions to consider

    Returns:
        True if any of the top-k predictions match ground truth, False otherwise
    """
    if not predicted_ids or not ground_truth_ids:
        return False

    # Take first k predicted PIDs
    top_k_ids = predicted_ids[:k]

    # Check if any matches ground truth
    for pid in top_k_ids:
        if pid != 0 and pid in ground_truth_ids:
            return True

    return False


def compute_position1_pass_at_k(
    predicted_ids: List[int],
    first_ground_truth_id: int,
    k: int
) -> bool:
    """
    Compute Position1_Pass@k for a single sample using PIDs

    Position1_Pass@k definition:
    - Take the first k candidate PIDs from predictions
    - Only consider the first PID in the ground truth
    - If any of these k PIDs matches the first ground truth, return True

    Args:
        predicted_ids: List of predicted PIDs (already extracted from generations)
        first_ground_truth_id: The first ground truth PID
        k: Number of top predictions to consider

    Returns:
        True if any of the top-k predictions match the first ground truth, False otherwise
    """
    if not predicted_ids or not first_ground_truth_id or first_ground_truth_id == 0:
        return False

    # Take first k predicted PIDs
    top_k_ids = predicted_ids[:k]

    # Check if any matches the first ground truth
    for pid in top_k_ids:
        if pid != 0 and pid == first_ground_truth_id:
            return True

    return False


def compute_recall_at_k(
    predicted_ids: List[int],
    ground_truth_ids: Set[int],
    k: int
) -> float:
    """
    Compute Recall@k for a single sample using PIDs

    Recall@k definition:
    - Take the first k candidate PIDs from predictions
    - Count how many unique ground truth PIDs are hit by these k PIDs
    - Return the ratio: hit_count / total_ground_truth_count

    Args:
        predicted_ids: List of predicted PIDs (already extracted from generations)
        ground_truth_ids: Set of ground truth PIDs
        k: Number of top predictions to consider

    Returns:
        Recall@k score (0.0 to 1.0)

    Examples:
        >>> predicted_ids = [123, 456, 999, 888]
        >>> ground_truth_ids = {123, 456, 789}
        >>> compute_recall_at_k(predicted_ids, ground_truth_ids, k=2)
        0.6667  # Hit 2 out of 3 ground truth PIDs
    """
    if not predicted_ids or not ground_truth_ids:
        return 0.0

    # Take first k predicted PIDs
    top_k_ids = predicted_ids[:k]

    # Convert to set and filter out zeros
    predicted_ids_set = set(pid for pid in top_k_ids if pid != 0)

    # Count how many ground truth PIDs are hit
    hit_count = len(predicted_ids_set & ground_truth_ids)  # Set intersection

    # Calculate recall
    recall = hit_count / len(ground_truth_ids)

    return recall


def get_unique_generations(
    generations: List[str],
    max_count: int,
    code_to_pid: Dict,
    strategy: str = "most_popular_originally",
    logprobs: List[float] = None,
    exclude_ids: Set[int] = None,
    sources: List[str] = None,
    encoding: str = "multiply"
):
    """
    Get first N unique PIDs from generations, optionally sorted by logprobs

    This function extracts unique PIDs, optionally sorting by logprobs first.
    Useful for merging results from multiple generation runs.

    Args:
        generations: List of model generation strings containing SID patterns
        max_count: Maximum number of unique PIDs to return
        code_to_pid: Mapping dictionary {encoded_sid: [{"pid": pid, "count": ..., "count_after_downsample": ...}, ...]}
        strategy: Strategy for selecting PID ("most_popular_originally", "most_popular_after_downsampling", or "random")
        logprobs: Optional list of log probabilities (same length as generations)
        exclude_ids: Optional set of PIDs to exclude from results
        sources: Optional list of source labels (same length as generations)

    Returns:
        List of unique PIDs (up to max_count), sorted by logprobs if provided
        If sources provided, returns tuple (List[int], List[str]) of (unique_pids, corresponding_sources)
    """
    # Track sources if provided
    track_sources = sources is not None and len(sources) == len(generations)

    # If logprobs provided, sort generations by logprobs (descending)
    if logprobs is not None and len(logprobs) == len(generations):
        # Create tuples and sort by logprob (descending)
        if track_sources:
            gen_data = list(zip(generations, logprobs, sources))
            gen_data.sort(key=lambda x: x[1], reverse=True)
            sorted_generations = [gen for gen, _, _ in gen_data]
            sorted_sources = [src for _, _, src in gen_data]
        else:
            gen_logprob_pairs = list(zip(generations, logprobs))
            gen_logprob_pairs.sort(key=lambda x: x[1], reverse=True)
            sorted_generations = [gen for gen, _ in gen_logprob_pairs]
            sorted_sources = None
    else:
        sorted_generations = generations
        sorted_sources = sources if track_sources else None

    seen = set()
    unique_pids = []
    unique_sources = [] if track_sources else None
    exclude = exclude_ids or set()

    for i, gen in enumerate(sorted_generations):
        # Skip empty strings
        if not gen or not gen.strip():
            continue

        # Extract PID from generation text
        pid = extract_id_from_generation(gen, code_to_pid, strategy, encoding=encoding)

        # Skip if PID is 0 (not found), already seen, or in exclude list
        if pid == 0 or pid in seen or pid in exclude:
            continue

        unique_pids.append(pid)
        seen.add(pid)

        if track_sources:
            unique_sources.append(sorted_sources[i])

        # Stop if we've collected enough unique PIDs
        if len(unique_pids) >= max_count:
            break

    if track_sources:
        return unique_pids, unique_sources
    return unique_pids


def compute_sid_validity(
    generations: List[str],
    valid_sid_set: Set,
    k: int,
    encoding: str = "multiply",
    sid_pattern: Optional[str] = None
) -> float:
    """
    Compute the validity rate of top-k generated SIDs

    Args:
        generations: List of model generation texts (already sorted by strategy)
        valid_sid_set: Set of valid encoded SIDs (all keys from mapping file)
        k: Number of top generations to check
        encoding: SID encoding method ("multiply" or "dot_join")
        sid_pattern: Optional custom regex whose capture groups form the SID tuple

    Returns:
        Validity rate, 0.0 if no generations
    """
    if not generations:
        return 0.0

    top_k_generations = generations[:k]
    total_count = len(top_k_generations)
    valid_count = 0

    for gen in top_k_generations:
        gen = gen.strip()
        # Handle </think> tag
        if '</think>' in gen:
            gen = gen.split('</think>')[-1].strip()

        # Extract SID codes
        sid_codes = extract_sid_codes_from_text(gen, sid_pattern=sid_pattern)
        if sid_codes is None:
            continue  # Cannot extract SID, treat as invalid

        # Encode and check if in valid set
        encoded_sid = encode_sid_auto(sid_codes, encoding)
        if encoded_sid is None:
            continue
        if encoded_sid in valid_sid_set:
            valid_count += 1

    return valid_count / total_count if total_count > 0 else 0.0


def get_debug_info(
    sample_id: str,
    generations: List[str],
    ground_truth: List[int],
    pass_results: Dict[str, bool],
    position1_pass_results: Dict[str, bool],
    code_to_pid: Dict[int, List[Dict[str, int]]],
    strategy: str = "most_popular_originally",
    raw_prompt: str = "",
    encoding: str = "multiply",
    sid_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prepare debug information for a sample (PID-based)

    Args:
        sample_id: Sample ID
        generations: List of generated SIDs
        ground_truth: Ground truth answer string
        pass_results: Pass@k results for this sample
        position1_pass_results: Position1_Pass@k results for this sample
        code_to_pid: Mapping dictionary {encoded_sid: [{"pid": pid, "count": ..., "count_after_downsample": ...}, ...]}
        strategy: Strategy for selecting PID ("most_popular_originally", "most_popular_after_downsampling", or "random")
        raw_prompt: Raw prompt (optional)
        sid_pattern: Optional custom regex whose capture groups form the SID tuple

    Returns:
        Debug information dictionary
    """
    ground_truth_ids = extract_ids_from_answer(ground_truth)
    first_ground_truth_id = extract_first_id_from_answer(ground_truth)

    # Extract top-k generated PIDs
    top_k_ids = [
        extract_id_from_generation(
            gen,
            code_to_pid,
            strategy,
            encoding=encoding,
            sid_pattern=sid_pattern,
        )
        for gen in generations[:10]
    ]

    debug_item = {
        "sample_id": sample_id,
        "ground_truth_pids": list(ground_truth_ids),
        "first_ground_truth_pid": first_ground_truth_id,
        "top_10_generations": top_k_ids,
        "pass_results": pass_results,
        "position1_pass_results": position1_pass_results,
    }

    if raw_prompt:
        debug_item["raw_prompt_snippet"] = raw_prompt[:200] + "..." if len(raw_prompt) > 200 else raw_prompt

    return debug_item
