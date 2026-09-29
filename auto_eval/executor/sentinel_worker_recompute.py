import sys
import time
import json
import subprocess
import shutil
import signal
import atexit
from pathlib import Path
from datetime import datetime
from benchmark.console import console, success_style, err_style, warning_style, dim_style, head_style, subhead_style, row_style
from auto_eval.config import WORKER_ID, QUEUE_BASE_DIR, LOG_BASE_DIR, DATA_DIR

import os

QUEUE_DIRS = {
        "incoming": QUEUE_BASE_DIR / "incoming",  # Shared queue (Web UI submissions)
        "incoming_worker": QUEUE_BASE_DIR / f"incoming-{WORKER_ID}",  # Worker-specific queue (manual high-priority tasks)
        "processing": QUEUE_BASE_DIR / f"processing-{WORKER_ID}",  # Worker-specific processing
        "done": QUEUE_BASE_DIR / "done",  # Shared
        "failed": QUEUE_BASE_DIR / "failed"  # Shared
    }

# ---------------------------------------------
# 1. Configuration paths
# ---------------------------------------------
# Queue directories (from config)
INCOMING_DIR = QUEUE_DIRS["incoming"]
INCOMING_WORKER_DIR = QUEUE_DIRS.get("incoming_worker", None)  # None if no WORKER_ID
PROCESSING_DIR = QUEUE_DIRS["processing"]
DONE_DIR = QUEUE_DIRS["done"]
FAILED_DIR = QUEUE_DIRS["failed"]

# Path of our evaluation script (eval_script.py) - using relative path
EVAL_SCRIPT_PATH = Path(__file__).parent / "eval_script_recompute.py"

# Polling interval
POLL_INTERVAL = 5 # (seconds)
# ---------------------------------------------

def find_next_job(directory):
    """
    Find the oldest job file in the directory (FIFO).
    Returns Path object, or None if not found.
    """
    try:
        files = [f for f in directory.glob('*.json') if f.is_file()]
        if not files:
            return None

        files.sort(key=lambda f: f.stat().st_mtime) # Sort by modification time
        return files[0] # Return the oldest file
    except FileNotFoundError:
        return None

def find_next_job_dual_queue():
    """
    Find the next job from dual queues with priority (if WORKER_ID is set).
    Priority order:
    1. Worker-specific queue (incoming-{WORKER_ID}/) - for manual high-priority tasks (if exists)

    Returns:
        Tuple of (job_file_path, is_from_shared_queue)
        - job_file_path: Path object or None
        - is_from_shared_queue: True if from shared queue, False if from worker-specific queue
    """
    # 1. First check worker-specific queue (high priority) - only if WORKER_ID is set
    if INCOMING_WORKER_DIR is not None:
        job_file = find_next_job(INCOMING_WORKER_DIR)
        if job_file:
            console.print(f"[Sentinel] Found job from worker-specific queue: {job_file.name}", style=dim_style)
            return job_file, False

    return None, False

def process_job(job_file_path):
    """
    Process a single job file, including retry logic.
    """
    job_filename = job_file_path.name
    processing_job_path = PROCESSING_DIR / job_filename

    console.print(f"\n[Sentinel] Found new job: {job_filename}\n", style=head_style)

    # 1. "Lock" the job (move to processing directory)
    try:
        shutil.move(str(job_file_path), str(processing_job_path))
        console.print(f"[Sentinel] Moved job {job_filename} -> processing/")
    except Exception as e:
        console.print(f"✗ [Sentinel] Unable to lock job {job_filename}: {e}. May have been preempted by another process, skipping.", style=err_style)
        return

    # 2. Read job configuration to get retry count
    try:
        with open(processing_job_path, 'r') as f:
            task_config = json.load(f)

        max_retries = task_config.get('max_retries', 3)
        task_id = task_config.get('task_id', job_filename)
        retry_count = task_config.get('retry_count', 0)

        # Add started_at timestamp when first moving to processing
        if 'started_at' not in task_config:
            task_config['started_at'] = datetime.now().isoformat()
            with open(processing_job_path, 'w') as f:
                json.dump(task_config, f, indent=4)

    except Exception as e:
        console.print(f"✗ [Sentinel] Unable to read JSON content of {job_filename}: {e}", style=err_style)
        shutil.move(str(processing_job_path), str(FAILED_DIR / job_filename))
        return

    # 3. Execute job (invoke eval_script.py), with retry
    for attempt in range(retry_count, max_retries):
        console.print(f"[Sentinel] Starting to execute job {task_id} (attempt {attempt + 1}/{max_retries})...", style=dim_style)

        try:
            # Print timestamp before evaluation
            start_time = datetime.now()
            print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting evaluation for job: {task_id}")

            # This is where we invoke our eval_script.py executor
            subprocess.run(
                ["python3", str(EVAL_SCRIPT_PATH), str(processing_job_path)],
                check=True, # Critical! If eval_script.py exits with non-zero, raise exception
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            # Print timestamp after evaluation
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Completed evaluation for job: {task_id} (Duration: {duration:.2f}s)")

            # 4. Success: move to done
            console.print(f"✓ [Sentinel] Job {task_id} succeeded.", style=success_style)

            # Add finished_at timestamp
            task_config['finished_at'] = datetime.now().isoformat()
            with open(processing_job_path, 'w') as f:
                json.dump(task_config, f, indent=4)

            shutil.move(str(processing_job_path), str(DONE_DIR / job_filename))
            return # Job succeeded, exit function

        except subprocess.CalledProcessError as e:
            # 5. Failed: log and prepare retry
            console.print(f"✗ [Sentinel] Job {task_id} (attempt {attempt + 1}) execution failed!", style=err_style)
            console.print(f"\n✗ [Sentinel] eval_script.py Stderr:\n{e.stderr}\n", style=err_style)

            # Update retry count in JSON and save error info from sentinel level
            task_config['retry_count'] = attempt + 1

            # If error_log not already set by eval_script.py, add it here
            if 'error_log' not in task_config:
                task_config['error_log'] = {
                    'sentinel_error': {
                        'exit_code': e.returncode,
                        'stdout': e.stdout,
                        'stderr': e.stderr,
                        'attempt': attempt + 1
                    }
                }

            try:
                with open(processing_job_path, 'w') as f:
                    json.dump(task_config, f, indent=4)
            except Exception as write_e:
                console.print(f"✗ [Sentinel] Unable to update retry count of {job_filename}: {write_e}", style=err_style)

            if attempt < max_retries - 1:
                console.print(f"[Sentinel] Waiting 60 seconds before retrying...", style=warning_style)
                time.sleep(60) # Wait 1 minute
            else:
                # 6. Reached max retries: move to failed
                console.print(f"✗ [Sentinel] Job {task_id} reached max retries ({max_retries}), moving to failed/", style=err_style)

                # Add finished_at timestamp for failed task
                task_config['finished_at'] = datetime.now().isoformat()
                with open(processing_job_path, 'w') as f:
                    json.dump(task_config, f, indent=4)

                shutil.move(str(processing_job_path), str(FAILED_DIR / job_filename))
                return # Completely failed, exit function
                
        except Exception as e:
            # Other unknown Python errors
            console.print(f"✗ [Sentinel] Critical error occurred, unable to execute {EVAL_SCRIPT_PATH}: {e}", style=err_style)

            # Add finished_at timestamp for critical error
            try:
                task_config['finished_at'] = datetime.now().isoformat()
                with open(processing_job_path, 'w') as f:
                    json.dump(task_config, f, indent=4)
            except:
                pass

            # Also treated as failure
            shutil.move(str(processing_job_path), str(FAILED_DIR / job_filename))
            return

def is_processing_busy():
    """
    Check if there is already a job being processed.
    Returns True if processing directory has any JSON files.
    """
    try:
        files = list(PROCESSING_DIR.glob('*.json'))
        return len(files) > 0
    except Exception:
        return False

def cleanup_directories():
    """
    Clean up worker-specific directories on exit (if WORKER_ID is set).
    Move remaining tasks to failed directory before removing.
    Note: Does NOT touch the shared incoming/ directory.
    """
    try:
        if WORKER_ID:
            console.print("\n\n[Sentinel] Cleaning up worker-specific directories...", style=warning_style)

            # Move tasks from worker-specific incoming directory to failed
            if INCOMING_WORKER_DIR is not None and INCOMING_WORKER_DIR.exists():
                incoming_files = list(INCOMING_WORKER_DIR.glob('*.json'))
                for file in incoming_files:
                    try:
                        shutil.move(str(file), str(FAILED_DIR / file.name))
                        console.print(f"[Sentinel] Moved {file.name} from {INCOMING_WORKER_DIR.name}/ to failed/", style=dim_style)
                    except Exception as e:
                        console.print(f"[Sentinel] Failed to move {file.name}: {e}", style=err_style)

                shutil.rmtree(INCOMING_WORKER_DIR)
                console.print(f"[Sentinel] Removed: {INCOMING_WORKER_DIR}", style=dim_style)

            # Move tasks from worker-specific processing directory to failed
            if PROCESSING_DIR.exists() and WORKER_ID in str(PROCESSING_DIR):
                processing_files = list(PROCESSING_DIR.glob('*.json'))
                for file in processing_files:
                    try:
                        shutil.move(str(file), str(FAILED_DIR / file.name))
                        console.print(f"[Sentinel] Moved {file.name} from {PROCESSING_DIR.name}/ to failed/", style=dim_style)
                    except Exception as e:
                        console.print(f"[Sentinel] Failed to move {file.name}: {e}", style=err_style)

                shutil.rmtree(PROCESSING_DIR)
                console.print(f"[Sentinel] Removed: {PROCESSING_DIR}", style=dim_style)

            console.print("[Sentinel] Cleanup completed. Shared incoming/ directory preserved.\n", style=success_style)
        else:
            console.print("\n\n[Sentinel] Single-worker mode: skipping cleanup (shared directories preserved).", style=warning_style)
    except Exception as e:
        console.print(f"[Sentinel] Error during cleanup: {e}", style=err_style)

def signal_handler(signum, frame):
    """
    Handle Ctrl+C (SIGINT) signal.
    """
    console.print("\n\n[Sentinel] Received interrupt signal (Ctrl+C)", style=warning_style)
    cleanup_directories()
    sys.exit(0)

def main_loop():
    console.print("\n\nSentinel Worker Started\n\n", style=head_style, justify="center")
    if WORKER_ID:
        console.print(f"Worker ID: {WORKER_ID}", style=row_style, justify="center")
        console.print(f"Worker-specific {INCOMING_WORKER_DIR}", style=row_style, justify="center")
    else:
        console.print("Mode: Single-worker (no WORKER_ID)", style=row_style, justify="center")
        console.print(f"Monitoring: Shared {INCOMING_DIR}", style=row_style, justify="center")

    # Ensure all directories exist
    dirs_to_create = [INCOMING_DIR, PROCESSING_DIR, DONE_DIR, FAILED_DIR]
    if INCOMING_WORKER_DIR is not None:
        dirs_to_create.append(INCOMING_WORKER_DIR)

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    while True:
        # Check if processing directory is busy
        if is_processing_busy():
            console.print(f"... (Processing busy, waiting @ {time.strftime('%H:%M:%S')}) ...", style=dim_style, end="\r")
            time.sleep(POLL_INTERVAL)
            continue

        job_file, is_from_shared = find_next_job_dual_queue()

        if job_file:
            process_job(job_file)
        else:
            # No jobs, take a break
            console.print(f"... (Waiting @ {time.strftime('%H:%M:%S')}) ...", style=dim_style, end="\r")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    # Also register cleanup for normal exit
    atexit.register(cleanup_directories)

    main_loop()