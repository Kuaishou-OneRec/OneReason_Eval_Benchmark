"""
Sentinel worker for speed-test tasks.

Completely independent from the evaluation sentinel — separate queue, separate process.
Usage:
    python3 -m auto_eval.speedtest.sentinel_speedtest
"""
import os
import sys
import time
import json
import subprocess
import shutil
import signal
import atexit
from pathlib import Path
from datetime import datetime

from config import SPEEDTEST_QUEUE_DIRS, SPEEDTEST_OUTPUT_DIR

POLL_INTERVAL = 10
SCRIPT_DIR = Path(__file__).parent.parent / "executor"
COMPARE_SCRIPT = SCRIPT_DIR / "compare_think_nothink.sh"


_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    print(f"\n[sentinel_speedtest] Received signal {signum}, shutting down gracefully...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def find_next_job(directory: Path):
    """Find oldest JSON file in directory (FIFO)."""
    try:
        files = [f for f in directory.glob("*.json") if f.is_file()]
        if not files:
            return None
        files.sort(key=lambda f: f.stat().st_mtime)
        return files[0]
    except FileNotFoundError:
        return None


def run_speed_task(task_data: dict, task_file: Path):
    """Execute a single speed-test task by calling compare_think_nothink.sh."""
    task_id = task_data["task_id"]
    model_path = task_data["model_path"]
    model_tag = task_data.get("model_tag", Path(model_path).name)
    tasks = task_data["tasks"]
    beams = task_data.get("beams", "1")
    batches = task_data.get("batches", "128")
    mode = task_data.get("mode", "nothink")
    gpu_mode = task_data.get("gpu_mode", "single")
    gpu_ids = task_data.get("gpu_ids", "0")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = SPEEDTEST_OUTPUT_DIR / f"sweep_{model_tag}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MODEL_SPECS"] = f"{model_tag}={model_path}"
    env["TASKS"] = tasks
    env["BEAMS"] = beams
    env["BATCHES"] = batches
    env["MODES"] = mode
    env["OUTPUT_ROOT"] = str(output_dir)
    env["CSV_PATH"] = str(output_dir / "results.csv")

    if gpu_mode == "multi":
        env["PARALLEL"] = "1"
        env["PARALLEL_GPUS"] = gpu_ids
        env["RAY_ADDRESS"] = "auto"
    else:
        env["PARALLEL"] = "0"
        env["GPU_ID"] = gpu_ids.split()[0]

    print(f"[sentinel_speedtest] Running task {task_id}: model={model_tag} tasks={tasks} beams={beams} gpu_mode={gpu_mode}")

    try:
        result = subprocess.run(
            ["bash", str(COMPARE_SCRIPT)],
            env=env,
            cwd=str(SCRIPT_DIR.parent.parent),
            capture_output=True,
            text=True,
        )

        log_file = output_dir / "run.log"
        log_file.write_text(result.stdout + "\n" + result.stderr)

        if result.returncode != 0:
            print(f"[sentinel_speedtest] Task {task_id} FAILED (rc={result.returncode})")
            task_data["error_log"] = result.stderr[-2000:] if result.stderr else ""
            return False

        print(f"[sentinel_speedtest] Task {task_id} completed. Results: {output_dir / 'results.csv'}")
        task_data["output_dir"] = str(output_dir)
        return True

    except Exception as e:
        print(f"[sentinel_speedtest] Task {task_id} exception: {e}")
        task_data["error_log"] = str(e)
        return False


def move_task(task_file: Path, dest_dir: Path, task_data: dict):
    """Move task JSON to destination directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / task_file.name
    task_file.write_text(json.dumps(task_data, indent=2, ensure_ascii=False))
    shutil.move(str(task_file), str(dest_file))


def main():
    for d in SPEEDTEST_QUEUE_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    SPEEDTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[sentinel_speedtest] Started. Polling {SPEEDTEST_QUEUE_DIRS['incoming']} every {POLL_INTERVAL}s")
    print(f"[sentinel_speedtest] Results will be stored in {SPEEDTEST_OUTPUT_DIR}")

    while not _shutdown_requested:
        task_file = find_next_job(SPEEDTEST_QUEUE_DIRS["incoming"])

        if task_file is None:
            time.sleep(POLL_INTERVAL)
            continue

        try:
            task_data = json.loads(task_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[sentinel_speedtest] Failed to read {task_file}: {e}")
            move_task(task_file, SPEEDTEST_QUEUE_DIRS["failed"], {"error": str(e)})
            continue

        # incoming -> processing
        move_task(task_file, SPEEDTEST_QUEUE_DIRS["processing"], task_data)
        processing_file = SPEEDTEST_QUEUE_DIRS["processing"] / task_file.name

        success = run_speed_task(task_data, processing_file)

        if success:
            move_task(processing_file, SPEEDTEST_QUEUE_DIRS["done"], task_data)
        else:
            move_task(processing_file, SPEEDTEST_QUEUE_DIRS["failed"], task_data)

    print("[sentinel_speedtest] Shutdown complete.")


if __name__ == "__main__":
    main()
