"""
Plot pid_pass@k vs temperature for each task.

Usage:
    python3 scripts/ray-vllm/plot_temperature_pid_pass.py [options]

Directory layout expected:
    <base_output_dir>/
        temperature_0_6/eval_results.json
        temperature_1/eval_results.json
        temperature_1_0/eval_results.json
        ...

eval_results.json structure:
    {model: {task: {split: {metric: value, ...}, ...}, ...}, ...}
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Optional

matplotlib_import_ok = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib_import_ok = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_temperature(dir_name: str):
    """
    Parse temperature value from directory name.

    Examples:
        temperature_0_6   -> 0.6
        temperature_0.6   -> 0.6
        temperature_1     -> 1.0
        temperature_1_0   -> 1.0
    """
    # Strip leading 'temperature_'
    suffix = re.sub(r"^temperature_", "", dir_name)
    if not suffix:
        return None

    # Replace underscores used as decimal separators: 0_6 -> 0.6
    # Strategy: if there is exactly one underscore and both sides are digits, treat as decimal.
    # Multiple underscores: treat each as decimal separator sequentially is ambiguous;
    # handle the common case: one underscore -> decimal point.
    parts = suffix.split("_")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        candidate = f"{parts[0]}.{parts[1]}"
    elif len(parts) == 1:
        candidate = parts[0]
    else:
        # e.g. temperature_1_0_5 (rare); join all parts and hope for the best
        candidate = ".".join(parts)

    try:
        return float(candidate)
    except ValueError:
        return None


# Regex: matches pid_pass@<digits> but NOT pid_position1_pass@k or pid_pass_random@k
_PID_PASS_RE = re.compile(r"^pid_pass@(\d+)$")


def extract_pid_pass_metrics(metrics: dict) -> dict:
    """Return {k_int: value} for all pid_pass@k metrics."""
    result = {}
    for metric_name, value in metrics.items():
        m = _PID_PASS_RE.match(metric_name)
        if m:
            k = int(m.group(1))
            result[k] = float(value)
    return result


def safe_filename(name: str) -> str:
    """Replace characters unsafe for filenames with underscores."""
    return re.sub(r"[^\w\-]", "_", name)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def collect_data(base_output_dir: str, split: str, model_filter: Optional[str]):
    """
    Returns list of dicts with keys: temperature, model, task, split, k, value.
    """
    records = []

    if not os.path.isdir(base_output_dir):
        print(f"ERROR: base_output_dir does not exist: {base_output_dir}", file=sys.stderr)
        sys.exit(1)

    found_any_json = False
    found_any_metric = False

    for entry in sorted(os.listdir(base_output_dir)):
        if not entry.startswith("temperature_"):
            continue
        temp_dir = os.path.join(base_output_dir, entry)
        if not os.path.isdir(temp_dir):
            continue

        temperature = parse_temperature(entry)
        if temperature is None:
            print(f"WARNING: could not parse temperature from '{entry}', skipping.", file=sys.stderr)
            continue

        eval_path = os.path.join(temp_dir, "eval_results.json")
        if not os.path.isfile(eval_path):
            print(f"WARNING: no eval_results.json in {temp_dir}, skipping.", file=sys.stderr)
            continue

        found_any_json = True

        with open(eval_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"WARNING: failed to parse {eval_path}: {e}", file=sys.stderr)
                continue

        # Structure: model -> task -> split -> metrics
        for model_name, tasks in data.items():
            if model_name.startswith("_"):
                continue  # skip summary fields
            if model_filter and model_name != model_filter:
                continue
            if not isinstance(tasks, dict):
                continue

            for task_name, splits in tasks.items():
                if task_name.startswith("_"):
                    continue
                if not isinstance(splits, dict):
                    continue

                split_data = splits.get(split)
                if split_data is None:
                    continue
                if not isinstance(split_data, dict):
                    continue

                pid_pass = extract_pid_pass_metrics(split_data)
                for k, value in pid_pass.items():
                    found_any_metric = True
                    records.append(
                        {
                            "temperature": temperature,
                            "model": model_name,
                            "task": task_name,
                            "split": split,
                            "k": k,
                            "value": value,
                        }
                    )

    if not found_any_json:
        print(
            f"ERROR: No eval_results.json files found under {base_output_dir}/temperature_*/",
            file=sys.stderr,
        )
        sys.exit(1)

    if not found_any_metric:
        print(
            f"ERROR: Found eval_results.json files but no pid_pass@k metrics "
            f"(split='{split}') in {base_output_dir}.",
            file=sys.stderr,
        )
        sys.exit(1)

    return records


def aggregate_records(records: list) -> dict:
    """
    Aggregate over models (mean) when no model filter is set.

    Returns nested dict: task -> k -> temperature -> mean_value
    """
    # Accumulate: (task, k, temperature) -> list of values
    acc: dict = defaultdict(list)
    for r in records:
        key = (r["task"], r["k"], r["temperature"])
        acc[key].append(r["value"])

    # Build result structure
    result: dict = defaultdict(lambda: defaultdict(dict))
    for (task, k, temp), values in acc.items():
        result[task][k][temp] = sum(values) / len(values)

    return result


def write_csv(records: list, output_path: str):
    fieldnames = ["task", "split", "k", "temperature", "value", "model"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({fn: r[fn] for fn in fieldnames})
    print(f"CSV written: {output_path}")


def plot_all_tasks_sum(agg: dict, output_path: str, show: bool):
    """
    agg: task -> k -> temperature -> mean_value  (from aggregate_records)

    For each (k, temperature), sum values across all tasks that have that pair.
    Missing task/temperature/k combinations are simply skipped (no zero-fill).
    """
    if not matplotlib_import_ok:
        print("ERROR: matplotlib is not installed; cannot plot.", file=sys.stderr)
        sys.exit(1)

    # Build: k -> temperature -> sum_value
    k_temp_sum: dict = defaultdict(lambda: defaultdict(float))
    for task_data in agg.values():
        for k, temp_val in task_data.items():
            for temp, val in temp_val.items():
                k_temp_sum[k][temp] += val

    fig, ax = plt.subplots(figsize=(8, 5))

    for k in sorted(k_temp_sum.keys()):
        temp_val = k_temp_sum[k]
        temps = sorted(temp_val.keys())
        values = [temp_val[t] for t in temps]
        line, = ax.plot(temps, values, marker="o", label=f"pid_pass@{k}")
        best_idx = values.index(max(values))
        ax.scatter(temps[best_idx], values[best_idx], marker="*", s=160,
                   color=line.get_color(), zorder=5)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Sum of pid_pass@k across tasks")
    ax.set_title("pid_pass@k by Temperature\nSum across all tasks")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_task(task_name: str, k_temp_value: dict, output_path: str, show: bool):
    """
    k_temp_value: {k: {temperature: value}}
    """
    if not matplotlib_import_ok:
        print("ERROR: matplotlib is not installed; cannot plot.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for k in sorted(k_temp_value.keys()):
        temp_val = k_temp_value[k]
        temps = sorted(temp_val.keys())
        values = [temp_val[t] for t in temps]
        line, = ax.plot(temps, values, marker="o", label=f"pid_pass@{k}")
        best_idx = values.index(max(values))
        ax.scatter(temps[best_idx], values[best_idx], marker="*", s=160,
                   color=line.get_color(), zorder=5)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("pid_pass@k")
    ax.set_title(f"pid_pass@k by Temperature\nTask: {task_name}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_BASE = (
    os.environ.get("BENCHMARK_RESULTS_DIR", "results")
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot pid_pass@k vs temperature for each task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_output_dir",
        default=DEFAULT_BASE,
        help="Root directory containing temperature_*/ subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save plots and CSV. Defaults to <base_output_dir>/plots.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to read from eval_results.json.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name to filter. If omitted, all models are aggregated "
            "(mean across models for each task/split/k/temperature)."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display plots interactively (requires a display; not recommended on servers).",
    )

    args = parser.parse_args()

    base_output_dir = args.base_output_dir
    output_dir = args.output_dir if args.output_dir else os.path.join(base_output_dir, "plots")

    os.makedirs(output_dir, exist_ok=True)

    # Collect raw records
    records = collect_data(base_output_dir, args.split, args.model)

    # Write CSV summary
    csv_path = os.path.join(output_dir, "temperature_pid_pass_summary.csv")
    write_csv(records, csv_path)

    # Aggregate and plot
    agg = aggregate_records(records)

    for task_name, k_temp_value in sorted(agg.items()):
        fname = safe_filename(task_name) + "_pid_pass_by_temperature.png"
        out_path = os.path.join(output_dir, fname)
        plot_task(task_name, k_temp_value, out_path, args.show)

    # Summary plot: sum pid_pass@k across all tasks
    sum_plot_path = os.path.join(output_dir, "all_tasks_sum_pid_pass_by_temperature.png")
    plot_all_tasks_sum(agg, sum_plot_path, args.show)

    print(f"\nDone. {len(agg)} task plot(s) + 1 summary plot written to: {output_dir}")


if __name__ == "__main__":
    main()
