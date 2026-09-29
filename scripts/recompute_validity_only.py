"""
Recompute only sid_validity metrics without modifying other metrics.

Uses pid_lookup_client for SID->PID lookup to determine validity.

Usage:
    python scripts/recompute_validity_only.py \
        --results_dir /path/to/resource
"""
import os
import argparse
import json
from pathlib import Path

from benchmark.tasks.v2_0.recommendation.utils_by_pid import (
    init_pid_lookup_client, extract_ids_from_generations, compute_sid_validity
)

# Standalone remote PID lookup config for this script (video/ad domain, slot=3)
_PID_LOOKUP_CONFIG = {}

TASK_TYPES = ['ad', 'video', 'product', 'label_cond', 'interactive']
K_VALUES = [1, 32, 128]

# Task name fallback mapping: new_name -> [old_names]
TASK_NAME_FALLBACKS = {
    'product': ['goods'],
    'video': ['sid_user_doc'],
}


def find_task_dir(model_dir, task_name, debug=False):
    """Find task directory with fallback support for old names.

    Searches in model_dir directly and all its subdirectories.

    Returns:
        Tuple of (task_dir_path, actual_dir_name) or (None, None) if not found
    """
    # Search locations: direct and all subdirectories
    search_bases = [model_dir]
    for subdir in model_dir.iterdir():
        if subdir.is_dir():
            search_bases.append(subdir)

    if debug:
        print(f"      [DEBUG] Search bases for {task_name}:")
        for base in search_bases:
            print(f"        - {base}")

    for base in search_bases:
        # Try current name first
        task_dir = base / task_name
        if debug:
            print(f"      [DEBUG] Checking: {task_dir} -> exists={task_dir.exists()}")
        if task_dir.exists():
            return task_dir, task_name

        # Try fallback names
        fallbacks = TASK_NAME_FALLBACKS.get(task_name, [])
        for old_name in fallbacks:
            task_dir = base / old_name
            if debug:
                print(f"      [DEBUG] Checking fallback: {task_dir} -> exists={task_dir.exists()}")
            if task_dir.exists():
                return task_dir, old_name

    return None, None


def recompute_validity_for_task(predictions_path, pid_client, task_name, k_values):
    """Compute sid_validity metrics for a single task using pid_lookup_client"""
    with open(predictions_path, 'r') as f:
        pred_data = json.load(f)

    samples = pred_data.get('samples', {})
    if not samples:
        return {}

    # Compute validity for each sample
    validity_sums = {k: 0.0 for k in k_values}
    total_samples = 0

    for sample_id, sample in samples.items():
        generations = sample.get('generations', [])
        if not generations:
            continue
        total_samples += 1

        # Batch extract PIDs
        pid_predictions = extract_ids_from_generations(generations, pid_client)

        for k in k_values:
            validity_rate = compute_sid_validity(pid_predictions, k)
            validity_sums[k] += validity_rate

    # Compute average
    metrics = {}
    for k in k_values:
        avg = validity_sums[k] / total_samples if total_samples > 0 else 0.0
        metrics[f'sid_validity@{k}'] = avg

    print(f"    {task_name}: {total_samples} samples, validity@32={metrics.get('sid_validity@32', 0):.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=os.environ.get("BENCHMARK_RESULTS_DIR", "results"))
    parser.add_argument('--model_dir', type=str, default=os.environ.get("BENCHMARK_RESULTS_DIR", "results"),
                        help="Debug: process only this specific model directory (full path)")
    parser.add_argument('--debug', action='store_true', help="Enable debug output")
    parser.add_argument('--k_values', type=int, nargs='+', default=K_VALUES)
    args = parser.parse_args()

    # Initialize pid_lookup_client
    print("Initializing pid_lookup_client...")
    pid_client = init_pid_lookup_client(_PID_LOOKUP_CONFIG)
    print(f"  pid_lookup_client initialized successfully")

    # Determine which directories to process
    if args.model_dir:
        # Debug mode: process single directory
        model_dirs = [Path(args.model_dir)]
        print(f"Debug mode: processing single directory {args.model_dir}")
    else:
        # Normal mode: process all results_* directories
        results_dir = Path(args.results_dir)
        model_dirs = sorted(results_dir.glob('results_*'))

    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue

        print(f"Processing: {model_dir.name}")
        eval_results_path = model_dir / 'eval_results.json'

        if not eval_results_path.exists():
            print("  No eval_results.json, skipping")
            continue

        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)

        if not eval_results:
            print("  eval_results.json is empty, skipping")
            continue

        model_name = list(eval_results.keys())[0]

        tasks_processed = 0
        for task_name in TASK_TYPES:
            task_dir, actual_dir_name = find_task_dir(model_dir, task_name, debug=args.debug)
            if task_dir is None:
                print(f"    {task_name}: directory not found")
                continue

            predictions_path = task_dir / 'test_generated.json'
            if not predictions_path.exists():
                print(f"    {task_name}: test_generated.json not found in {actual_dir_name}/")
                continue

            # Skip if sid_validity@32 already exists
            eval_task_name = actual_dir_name if actual_dir_name in eval_results[model_name] else task_name
            if eval_task_name in eval_results[model_name]:
                test_metrics = eval_results[model_name][eval_task_name].get('test', {})
                if 'sid_validity@32' in test_metrics:
                    print(f"    {task_name}: sid_validity@32 already exists, skipping")
                    continue

            validity_metrics = recompute_validity_for_task(
                predictions_path, pid_client, task_name, args.k_values
            )

            # Update metrics
            if eval_task_name in eval_results[model_name]:
                if 'test' in eval_results[model_name][eval_task_name]:
                    eval_results[model_name][eval_task_name]['test'].update(validity_metrics)
                    tasks_processed += 1

        if tasks_processed > 0:
            with open(eval_results_path, 'w') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)
            print(f"  Done ({tasks_processed} tasks updated)\n")
        else:
            print("  No tasks found, skipping\n")

    print("All directories processed.")


if __name__ == '__main__':
    main()
