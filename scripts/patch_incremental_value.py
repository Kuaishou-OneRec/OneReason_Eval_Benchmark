"""
Patch incremental_value field into metadata of *_generated.json and *_generated.json.debug files,
and compute incremental_value_weighted_pid_recall@k metrics.

Reads uid -> incremental_value mapping from {data_dir}/{task_name}/{task_name}_{split}.parquet,
then:
  1. Patches sample["metadata"]["incremental_value"] for all matching samples.
  2. Computes incremental_value_weighted_pid_recall@k using pid_generations and answer_pid,
     writes per-sample result into sample and aggregates into eval_results.json.

weighted_pid_recall@k(sample):
    numerator   = sum of incremental_value[i] where answer_pid[i] in top-k pid_generations
    denominator = sum of all incremental_value
    result      = numerator / denominator  (0.0 if denominator == 0)

weighted_pid_recall@k(global) = mean over all samples

Usage:
    # Single output_dir (one checkpoint)
    python scripts/patch_incremental_value.py \\
        --output_dir /path/to/results \\
        --version v6.0 \\
        --task_types video_v2.1

    # Batch mode: result_dir contains multiple output_dir subdirectories
    python scripts/patch_incremental_value.py \\
        --result_dir /path/to/all_results \\
        --version v6.0

    # Dry run (print only, no writes)
    python scripts/patch_incremental_value.py \\
        --output_dir /path/to/results \\
        --version v6.0 \\
        --dry-run
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from benchmark.tasks.v2_0.recommendation.utils_by_pid import compute_weighted_pid_recall_at_k

INCREMENTAL_VALUE_FIELD = "incremental_value"
BASE_DATA_DIR = os.environ.get("BENCHMARK_WORK_DIR", "workspace")
K_VALUES = [1, 4, 8, 16, 32, 64]


def get_args():
    parser = argparse.ArgumentParser(
        description="Patch incremental_value into metadata of *_generated.json files "
                    "and compute incremental_value_weighted_pid_recall@k metrics."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--output_dir",
        type=str, default=None,
        help="Single output directory (same as eval_dev_results.py).",
    )
    group.add_argument(
        "--result_dir",
        type=str, default=None,
        help="Parent directory whose subdirectories are each treated as an output_dir.",
    )
    parser.add_argument(
        "--version",
        type=str, default=None,
        help="Benchmark version (e.g., v6.0), auto-computes data_dir.",
    )
    parser.add_argument(
        "--data_dir",
        type=str, default=None,
        help="Explicit data directory (overrides --version).",
    )
    parser.add_argument(
        "--task_types",
        type=str, nargs="+", default=None,
        help="Task name whitelist (default: all tasks found).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only, do not write any files.",
    )
    return parser.parse_args()


def build_uid_map(data_dir: str, task_name: str, split: str) -> dict:
    """Build uid -> incremental_value mapping from parquet file.

    incremental_value is read from inside the 'metadata' JSON column.
    """
    parquet_path = Path(data_dir) / task_name / f"{task_name}_{split}.parquet"
    if not parquet_path.exists():
        print(f"      [WARN] Parquet not found: {parquet_path}, skipping uid map build")
        return {}

    pf = pq.ParquetFile(str(parquet_path))
    schema_names = pf.schema_arrow.names
    if "uid" not in schema_names:
        print(f"      [WARN] 'uid' field not found in {parquet_path}, skipping")
        return {}
    if "metadata" not in schema_names:
        print(f"      [WARN] 'metadata' field not found in {parquet_path}, skipping")
        return {}

    uid_map = {}
    missing_field = 0
    for batch in pf.iter_batches(batch_size=4096, columns=["uid", "metadata"]):
        for row in batch.to_pylist():
            uid_key = str(row["uid"]) if row["uid"] is not None else ""
            meta_raw = row.get("metadata")
            if not meta_raw:
                missing_field += 1
                continue
            try:
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            except (json.JSONDecodeError, TypeError):
                missing_field += 1
                continue
            if INCREMENTAL_VALUE_FIELD not in meta:
                missing_field += 1
                continue
            uid_map[uid_key] = meta[INCREMENTAL_VALUE_FIELD]

    if missing_field:
        print(f"      [WARN] {missing_field} rows missing '{INCREMENTAL_VALUE_FIELD}' in metadata")
    print(f"      [PARQUET] Loaded {len(uid_map)} uid mappings from {parquet_path}")
    return uid_map


def patch_and_compute_metrics(
    generation_file: str,
    uid_map: dict,
    *,
    dry_run: bool = False,
) -> dict:
    """Patch incremental_value into metadata and compute weighted_pid_recall@k.

    Args:
        generation_file: Path to *_generated.json (main file, not .debug).
        uid_map:         uid (str) -> incremental_value (list).
        dry_run:         If True, print stats only without writing.

    Returns:
        dict with keys:
            patch_stats: {total, patched, missing_uid}
            metrics:     {incremental_value_weighted_pid_recall@k: float, ...}
    """
    with open(generation_file, "r", encoding="utf-8") as f:
        gen_data = json.load(f)

    samples = gen_data.get("samples", {})
    patch_stats = {"total": len(samples), "patched": 0, "missing_uid": 0}
    counters = defaultdict(float)
    sample_scores: dict = {}
    metric_samples = 0

    if not samples:
        print(f"      [WARN] No samples in {generation_file}, skipping")
        return {"patch_stats": patch_stats, "metrics": {}, "sample_scores": {}}

    for sample_id, sample in tqdm(
        samples.items(),
        desc=f"      {Path(generation_file).name}",
        total=patch_stats["total"],
    ):
        metadata = sample.get("metadata", {})
        uid_val = metadata.get("uid")
        uid_key = str(uid_val) if uid_val is not None else ""

        incremental_value = None
        if uid_key in uid_map:
            incremental_value = uid_map[uid_key]
            metadata[INCREMENTAL_VALUE_FIELD] = incremental_value
            sample["metadata"] = metadata
            patch_stats["patched"] += 1
        else:
            patch_stats["missing_uid"] += 1

        answer_pid = metadata.get("answer_pid") or metadata.get("answer_iid") or []
        pid_generations = sample.get("pid_generations", [])

        if incremental_value and answer_pid and pid_generations:
            metric_samples += 1
            scores = {}
            for k in K_VALUES:
                score = compute_weighted_pid_recall_at_k(
                    pid_generations=pid_generations,
                    answer_pid=answer_pid,
                    incremental_value=incremental_value,
                    k=k,
                )
                scores[f"incremental_value_weighted_pid_recall@{k}"] = score
                counters[f"incremental_value_weighted_pid_recall@{k}"] += score
            sample.update(scores)
            sample_scores[sample_id] = scores

    metrics = {}
    if metric_samples > 0:
        for key, total in counters.items():
            metrics[key] = total / metric_samples

    if dry_run:
        print(
            f"      [DRY-RUN] {Path(generation_file).name}: "
            f"total={patch_stats['total']} patched={patch_stats['patched']} "
            f"missing={patch_stats['missing_uid']} metric_samples={metric_samples}"
        )
        for k, v in sorted(metrics.items()):
            print(f"        {k} = {v:.4f}")
        return {"patch_stats": patch_stats, "metrics": metrics, "sample_scores": sample_scores}

    with open(generation_file, "w", encoding="utf-8") as f:
        json.dump(gen_data, f, indent=2, ensure_ascii=False)

    print(
        f"      [PATCH] {Path(generation_file).name}: "
        f"total={patch_stats['total']} patched={patch_stats['patched']} "
        f"missing={patch_stats['missing_uid']} metric_samples={metric_samples}"
    )
    for k, v in sorted(metrics.items()):
        print(f"        {k} = {v:.4f}")

    return {"patch_stats": patch_stats, "metrics": metrics, "sample_scores": sample_scores}


def patch_debug_file(
    debug_file: str,
    uid_map: dict,
    sample_scores: dict,
    *,
    dry_run: bool = False,
) -> dict:
    """Patch incremental_value into metadata and write per-sample metric scores
    into a *_generated.json.debug file.

    Args:
        sample_scores: {sample_id: {metric_name: value}} from patch_and_compute_metrics.

    Returns:
        dict: patch_stats with keys total, patched, missing_uid.
    """
    with open(debug_file, "r", encoding="utf-8") as f:
        gen_data = json.load(f)

    samples = gen_data.get("samples", {})
    stats = {"total": len(samples), "patched": 0, "missing_uid": 0}

    if not samples:
        print(f"      [WARN] No samples in {debug_file}, skipping")
        return stats

    for sample_id, sample in tqdm(
        samples.items(),
        desc=f"      {Path(debug_file).name}",
        total=stats["total"],
    ):
        metadata = sample.get("metadata", {})
        uid_val = metadata.get("uid")
        uid_key = str(uid_val) if uid_val is not None else ""

        if uid_key in uid_map:
            metadata[INCREMENTAL_VALUE_FIELD] = uid_map[uid_key]
            sample["metadata"] = metadata
            stats["patched"] += 1
        else:
            stats["missing_uid"] += 1

        if sample_id in sample_scores:
            sample.update(sample_scores[sample_id])

    if dry_run:
        print(
            f"      [DRY-RUN] {Path(debug_file).name}: "
            f"total={stats['total']} patched={stats['patched']} missing={stats['missing_uid']}"
        )
        return stats

    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(gen_data, f, indent=2, ensure_ascii=False)

    print(
        f"      [PATCH] {Path(debug_file).name}: "
        f"total={stats['total']} patched={stats['patched']} missing={stats['missing_uid']}"
    )
    return stats


def collect_task_splits(output_dir: Path, task_types: list) -> dict:
    """Scan output_dir to collect {(task_name, split): ...} needed for uid map building.

    Returns:
        dict: {(task_name, split): None} — ordered unique pairs.
    """
    pairs = {}
    for model_name in sorted(os.listdir(output_dir)):
        model_dir = output_dir / model_name
        if not model_dir.is_dir():
            continue
        all_tasks = [t for t in os.listdir(model_dir) if (model_dir / t).is_dir()]
        if task_types:
            all_tasks = [t for t in all_tasks if t in task_types]
        for task_name in all_tasks:
            task_dir = model_dir / task_name
            for filename in sorted(os.listdir(task_dir)):
                if not filename.endswith("_generated.json") or filename.endswith(".debug"):
                    continue
                split = filename.replace("_generated.json", "")
                pairs[(task_name, split)] = None
    return pairs


def process_output_dir(
    output_dir: Path,
    data_dir: str,
    task_types: list,
    uid_map_cache: dict,
    dry_run: bool,
):
    """Process a single output_dir: patch metadata and compute weighted_pid_recall@k."""
    if not output_dir.exists():
        print(f"  [WARN] output_dir does not exist: {output_dir}, skipping")
        return

    # Build uid maps once upfront for all (task_name, split) pairs in this output_dir
    needed = collect_task_splits(output_dir, task_types)
    for (task_name, split) in needed:
        if (task_name, split) not in uid_map_cache:
            uid_map_cache[(task_name, split)] = build_uid_map(data_dir, task_name, split)

    eval_results_path = output_dir / "eval_results.json"
    eval_results = {}
    if eval_results_path.exists():
        with open(eval_results_path, "r", encoding="utf-8") as f:
            eval_results = json.load(f)

    for model_name in sorted(os.listdir(output_dir)):
        model_dir = output_dir / model_name
        if not model_dir.is_dir():
            continue

        print(f"\n  Model: {model_name}")

        all_tasks = [t for t in os.listdir(model_dir) if (model_dir / t).is_dir()]
        if task_types:
            all_tasks = [t for t in all_tasks if t in task_types]

        if not all_tasks:
            print("    No tasks found, skipping")
            continue

        for task_name in sorted(all_tasks):
            task_dir = model_dir / task_name
            print(f"    Task: {task_name}")

            for filename in sorted(os.listdir(task_dir)):
                if not filename.endswith("_generated.json") or filename.endswith(".debug"):
                    continue

                split = filename.replace("_generated.json", "")
                uid_map = uid_map_cache.get((task_name, split), {})

                if not uid_map:
                    print(f"      [SKIP] No uid map for {task_name}/{split}, skipping")
                    continue

                main_file = task_dir / filename
                debug_file = task_dir / f"{filename}.debug"

                print(f"      File: {main_file.name}")
                result = patch_and_compute_metrics(str(main_file), uid_map, dry_run=dry_run)

                if debug_file.exists():
                    print(f"      File: {debug_file.name}")
                    patch_debug_file(str(debug_file), uid_map, result.get("sample_scores", {}), dry_run=dry_run)

                metrics = result.get("metrics", {})
                if metrics and not dry_run:
                    eval_results.setdefault(model_name, {}).setdefault(task_name, {}).setdefault(split, {}).update(metrics)

    if eval_results and not dry_run:
        with open(eval_results_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print(f"  Saved updated eval_results.json to {eval_results_path}")


def main():
    args = get_args()

    data_dir = args.data_dir if args.data_dir else f"{BASE_DATA_DIR}data_{args.version}"

    uid_map_cache = {}

    if args.result_dir:
        result_dir = Path(args.result_dir)
        if not result_dir.exists():
            print(f"Error: result_dir does not exist: {result_dir}")
            return

        output_dirs = sorted([
            d for d in result_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        print(f"Batch mode: found {len(output_dirs)} subdirectories in {result_dir}")

        for i, output_dir in enumerate(output_dirs, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(output_dirs)}] output_dir: {output_dir}")
            print(f"{'='*60}")
            process_output_dir(
                output_dir=output_dir,
                data_dir=data_dir,
                task_types=args.task_types,
                uid_map_cache=uid_map_cache,
                dry_run=args.dry_run,
            )
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"Error: output_dir does not exist: {output_dir}")
            return

        print(f"Single mode: processing {output_dir}")
        process_output_dir(
            output_dir=output_dir,
            data_dir=data_dir,
            task_types=args.task_types,
            uid_map_cache=uid_map_cache,
            dry_run=args.dry_run,
        )

    print(f"\nDone. Parquet uid maps loaded {len(uid_map_cache)} time(s) total.")
    if args.dry_run:
        print("DRY-RUN: no files were modified.")


if __name__ == "__main__":
    main()
