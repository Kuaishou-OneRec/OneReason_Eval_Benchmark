"""
Speed Test handlers: task submission, result parsing, resource estimation, theoretical calculation.
"""
import json
import csv
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from config import SPEEDTEST_QUEUE_DIRS, SPEEDTEST_OUTPUT_DIR


# ---------------------------------------------------------------------------
# Task submission
# ---------------------------------------------------------------------------

def submit_speed_task(
    model_path: str,
    model_tag: str,
    tasks: str,
    beams: str,
    batches: str,
    mode: str,
    gpu_mode: str,
    gpu_ids: str,
) -> str:
    """Write a speed-test task JSON into the incoming queue directory."""
    if not model_path.strip():
        return "❌ 请填写模型路径"
    if not tasks.strip():
        return "❌ 请选择至少一个任务"

    task_id = f"speed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    task_data = {
        "task_id": task_id,
        "model_path": model_path.strip(),
        "model_tag": model_tag.strip() or Path(model_path.strip()).name,
        "tasks": tasks.strip(),
        "beams": beams.strip() or "1",
        "batches": batches.strip() or "128",
        "mode": mode,
        "gpu_mode": gpu_mode,
        "gpu_ids": gpu_ids.strip() or "0",
        "submitted_at": datetime.now().isoformat(),
    }

    incoming_dir = SPEEDTEST_QUEUE_DIRS["incoming"]
    incoming_dir.mkdir(parents=True, exist_ok=True)
    task_file = incoming_dir / f"{task_id}.json"
    task_file.write_text(json.dumps(task_data, indent=2, ensure_ascii=False))
    return f"✅ 测速任务已提交: {task_id}"


# ---------------------------------------------------------------------------
# Queue status
# ---------------------------------------------------------------------------

def get_speedtest_queue_status() -> Tuple[str, str, str]:
    """Return (incoming_html, processing_html, done_html) for queue display."""
    sections = []
    for label, key in [("等待中", "incoming"), ("运行中", "processing"), ("已完成", "done"), ("失败", "failed")]:
        d = SPEEDTEST_QUEUE_DIRS[key]
        if not d.exists():
            sections.append(f"**{label}**: 0")
            continue
        count = len(list(d.glob("*.json")))
        sections.append(f"**{label}**: {count}")
    return " | ".join(sections)


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def scan_speed_results() -> List[Dict[str, Any]]:
    """Scan SPEEDTEST_OUTPUT_DIR for completed sweep results.csv files."""
    results = []
    if not SPEEDTEST_OUTPUT_DIR.exists():
        return results

    for sweep_dir in sorted(SPEEDTEST_OUTPUT_DIR.iterdir(), reverse=True):
        if not sweep_dir.is_dir():
            continue
        csv_path = sweep_dir / "results.csv"
        if not csv_path.exists():
            continue
        rows = _parse_results_csv(csv_path)
        for row in rows:
            row["sweep_dir"] = str(sweep_dir)
            row["sweep_name"] = sweep_dir.name
        results.extend(rows)
    return results


def _parse_results_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Parse a results.csv from compare_think_nothink.sh."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "model_tag": row.get("model_tag", ""),
                "task": row.get("task", ""),
                "mode": row.get("mode", ""),
                "beam": int(row.get("beam", 0)),
                "batch": int(row.get("batch", 0)),
                "result": row.get("result", ""),
                "wall_time_s": _safe_float(row.get("wall_time_s")),
                "infer_time_s": _safe_float(row.get("infer_time_s")),
                "input_tokens": _safe_int(row.get("input_tokens")),
                "output_tokens": _safe_int(row.get("output_tokens")),
                "decode_tok_per_s": _safe_float(row.get("decode_tok_per_s")),
                "peak_gpu_mem_mb": _safe_float(row.get("peak_gpu_mem_mb")),
            }
            rows.append(parsed)
    return rows


def parse_generated_json(json_path: Path) -> Dict[str, Any]:
    """Extract mfu_stats_aggregate and metadata from a generated.json file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    mfu = data.get("mfu_stats_aggregate", {})
    return {
        "num_params": data.get("num_params", 0),
        "total_time": data.get("total_time", 0),
        "avg_time_per_sample": data.get("avg_time_per_sample", 0),
        "hardware_info": data.get("hardware_info", {}),
        "num_samples": len(data.get("samples", [])),
        "mfu_total_input_tokens": mfu.get("total_input_tokens", []),
        "mfu_total_output_tokens": mfu.get("total_output_tokens", []),
        "mfu_total_time": mfu.get("total_time", []),
    }


# ---------------------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------------------

def estimate_resources(
    sample_count: int,
    wall_time_s: float,
    infer_time_s: float,
    target_count: int,
    gpu_counts: List[int],
) -> List[Dict[str, Any]]:
    """
    Linear extrapolation from benchmark results to target data volume.

    Returns list of dicts with {num_gpus, wall_est_s, infer_est_s, wall_est_display, infer_est_display}.
    """
    if sample_count <= 0:
        return []

    per_sample_wall = wall_time_s / sample_count
    per_sample_infer = infer_time_s / sample_count

    results = []
    for n in gpu_counts:
        wall_est = per_sample_wall * target_count / n
        infer_est = per_sample_infer * target_count / n
        results.append({
            "num_gpus": n,
            "wall_est_s": wall_est,
            "infer_est_s": infer_est,
            "wall_est_display": _format_duration(wall_est),
            "infer_est_display": _format_duration(infer_est),
        })
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float:
    try:
        return float(val) if val else 0.0
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val) -> int:
    try:
        return int(val) if val else 0
    except (ValueError, TypeError):
        return 0


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}天"
