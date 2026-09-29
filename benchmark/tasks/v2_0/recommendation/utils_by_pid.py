"""
Recommendation Task Utilities (PID-based)

Functions for PID extraction and recommendation metrics computation using PIDs.
Uses pid_lookup_client + CityHash64 for SID -> PID conversion.
"""

import numpy as np
from typing import Set, Dict, List, Any

from benchmark.tasks.v2_0.recommendation.utils_8sid2emb import extract_sid_codes


def init_pid_lookup_client(pid_lookup_config: dict):
    """Legacy extension point; use a local SID-to-PID mapping in public runs."""
    raise RuntimeError("Remote PID lookup is not bundled; configure sid_to_pid_strategy and mapping_filename")


def compute_semantic_hash(sid_list: List[int]) -> int:
    """
    Compute CityHash64 of SID components.

    Joins the 8 values with '.' and hashes the resulting string.

    Args:
        sid_list: List of 8 SID component values

    Returns:
        uint64 hash value
    """
    semantic_id_str = '.'.join(str(sid) for sid in sid_list[:8])
    import clickhouse_cityhash.cityhash as ch_cityhash
    hash_value_raw = ch_cityhash.CityHash64(semantic_id_str)
    return int(hash_value_raw) & 0xFFFFFFFFFFFFFFFF


def lookup_pid(client, hash_value: int) -> int:
    """
    Look up a single PID from pid_lookup_client using semantic hash.

    Args:
        client: Initialized PIDLookupClient instance
        hash_value: uint64 semantic hash

    Returns:
        PID (int), or 0 if lookup fails
    """
    result = client.fetch_number([hash_value])
    if result:
        try:
            pid = int(np.array(result[0], dtype=np.uint64).flat[0])
            return pid if pid != 0 else 0
        except (ValueError, TypeError, IndexError):
            return 0
    return 0


def batch_lookup_pids(client, hash_values: List[int]) -> List[int]:
    """
    Batch look up PIDs from pid_lookup_client.

    Args:
        client: Initialized PIDLookupClient instance
        hash_values: List of uint64 semantic hashes

    Returns:
        List of PIDs (same length as input), 0 for failed lookups
    """
    if not hash_values:
        return []
    result = client.fetch_number(hash_values)
    if result:
        pids = []
        for r in result:
            try:
                # Each result element may be a scalar or a sequence
                val = int(np.array(r, dtype=np.uint64).flat[0])
                pids.append(val if val != 0 else 0)
            except (ValueError, TypeError, IndexError):
                pids.append(0)
        return pids
    return [0] * len(hash_values)


def extract_id_from_generation(generation: str, client) -> int:
    """
    Extract PID from a single model generation via pid_lookup_client lookup.

    Flow: extract SID codes -> CityHash64 -> remote PID lookup query -> PID

    Args:
        generation: Model generation string containing SID pattern
        client: Initialized PIDLookupClient instance

    Returns:
        Extracted PID (int), or 0 if not found
    """
    sid_components = extract_sid_codes(generation)
    if sid_components is None or len(sid_components) < 8:
        return 0

    hash_value = compute_semantic_hash(sid_components)
    return lookup_pid(client, hash_value)


def extract_ids_from_generations(generations: List[str], client) -> List[int]:
    """
    Batch extract PIDs from multiple generations via pid_lookup_client.

    More efficient than calling extract_id_from_generation individually
    as it batches the remote PID lookup queries.

    Args:
        generations: List of model generation strings
        client: Initialized PIDLookupClient instance

    Returns:
        List of PIDs (same length as input), 0 for failed extractions
    """
    hash_values = []
    hash_indices = []

    for i, gen in enumerate(generations):
        sid_components = extract_sid_codes(gen)
        if sid_components is None or len(sid_components) < 8:
            continue
        hash_val = compute_semantic_hash(sid_components)
        hash_values.append(hash_val)
        hash_indices.append(i)

    # Batch query
    if hash_values:
        pids = batch_lookup_pids(client, hash_values)
    else:
        pids = []

    # Map back to generation order
    result = [0] * len(generations)
    for idx, pid in zip(hash_indices, pids):
        result[idx] = pid

    return result


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
    client,
    logprobs: List[float] = None,
    exclude_ids: Set[int] = None,
    sources: List[str] = None
):
    """
    Get first N unique PIDs from generations, optionally sorted by logprobs

    This function extracts unique PIDs, optionally sorting by logprobs first.
    Useful for merging results from multiple generation runs.

    Args:
        generations: List of model generation strings containing SID patterns
        max_count: Maximum number of unique PIDs to return
        client: Initialized PIDLookupClient instance
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

    # Batch extract all PIDs first
    all_pids = extract_ids_from_generations(sorted_generations, client)

    seen = set()
    unique_pids = []
    unique_sources = [] if track_sources else None
    exclude = exclude_ids or set()

    for i, pid in enumerate(all_pids):
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
    predicted_pids: List[int],
    k: int
) -> float:
    """
    Compute the validity rate of top-k predicted PIDs.

    A prediction is "valid" if the remote PID lookup lookup returned a non-zero PID,
    meaning the generated SID maps to a known item.

    Args:
        predicted_pids: List of predicted PIDs (0 means invalid/unknown SID)
        k: Number of top predictions to check

    Returns:
        Validity rate (0.0 to 1.0), 0.0 if no predictions
    """
    if not predicted_pids:
        return 0.0

    top_k_pids = predicted_pids[:k]
    total_count = len(top_k_pids)
    valid_count = sum(1 for pid in top_k_pids if pid != 0)

    return valid_count / total_count if total_count > 0 else 0.0


def get_debug_info(
    sample_id: str,
    generations: List[str],
    ground_truth: List[int],
    pass_results: Dict[str, bool],
    position1_pass_results: Dict[str, bool],
    predicted_pids: List[int] = None,
    client=None,
    raw_prompt: str = ""
) -> Dict[str, Any]:
    """
    Prepare debug information for a sample (PID-based)

    Args:
        sample_id: Sample ID
        generations: List of generated SIDs
        ground_truth: Ground truth PIDs list
        pass_results: Pass@k results for this sample
        position1_pass_results: Position1_Pass@k results for this sample
        predicted_pids: Pre-computed predicted PIDs (preferred, avoids re-lookup)
        client: PIDLookupClient instance (used only if predicted_pids not provided)
        raw_prompt: Raw prompt (optional)

    Returns:
        Debug information dictionary
    """
    ground_truth_ids = extract_ids_from_answer(ground_truth)
    first_ground_truth_id = extract_first_id_from_answer(ground_truth)

    # Use pre-computed PIDs if available, otherwise extract
    if predicted_pids is not None:
        top_k_ids = predicted_pids[:10]
    elif client is not None:
        top_k_ids = extract_ids_from_generations(generations[:10], client)
    else:
        top_k_ids = [0] * min(10, len(generations))

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


def compute_weighted_pid_recall_from_candidate_set(
    candidate_pids,
    answer_pid: List[int],
    incremental_value: List[float],
) -> float:
    """Compute incremental_value weighted recall against an arbitrary candidate PID set.

    numerator   = sum of incremental_value[i] where answer_pid[i] in candidate_pids (deduped by pid)
    denominator = sum of all incremental_value (deduped by pid)

    Args:
        candidate_pids: Iterable of candidate PIDs (Set/List/Tuple). Will be coerced to set.
        answer_pid:      Ground truth PID list, positionally aligned with incremental_value.
        incremental_value: Weight for each ground truth PID, same length as answer_pid.

    Returns:
        Weighted recall score in [0.0, 1.0].
    """
    if not answer_pid or not incremental_value:
        return 0.0

    seen_pids: set = set()
    deduped_pairs = []
    for pid, weight in zip(answer_pid, incremental_value):
        if pid is not None and pid != 0 and pid not in seen_pids:
            seen_pids.add(pid)
            deduped_pairs.append((pid, weight))

    if not deduped_pairs:
        return 0.0

    denominator = sum(float(w) for _, w in deduped_pairs if w is not None)
    if denominator == 0.0:
        return 0.0

    candidate_set = candidate_pids if isinstance(candidate_pids, set) else set(candidate_pids or [])

    numerator = sum(
        float(w) for pid, w in deduped_pairs
        if w is not None and pid in candidate_set
    )
    return numerator / denominator


def compute_weighted_pid_recall_at_k(
    pid_generations: List[int],
    answer_pid: List[int],
    incremental_value: List[float],
    k: int,
) -> float:
    """Compute incremental_value_weighted_pid_recall@k for a single sample.

    numerator   = sum of incremental_value[i] where answer_pid[i] in top-k pid_generations (deduped)
    denominator = sum of all incremental_value (deduped by pid)

    Args:
        pid_generations: List of predicted PIDs.
        answer_pid:      Ground truth PID list, positionally aligned with incremental_value.
        incremental_value: Weight for each ground truth PID, same length as answer_pid.
        k:               Number of top pid_generations to consider.

    Returns:
        Weighted recall score in [0.0, 1.0].
    """
    if not pid_generations:
        return 0.0

    top_k_pids = {
        pid for pid in pid_generations[:k]
        if pid is not None and pid != 0
    }
    return compute_weighted_pid_recall_from_candidate_set(
        candidate_pids=top_k_pids,
        answer_pid=answer_pid,
        incremental_value=incremental_value,
    )
