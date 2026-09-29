"""
Sentinel Worker for Generation Tasks

This worker monitors the incoming directory for generation task JSON files,
processes them using vLLM inference, and writes results back to the JSON files.

Usage:
    python sentinel_worker.py

The worker will:
1. Monitor the incoming directory for new .json files
2. Move found files to processing directory
3. Execute generation_script.py for each task
4. Move completed tasks to done/ or failed/ directory
"""
import sys
import time
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

from benchmark.console import console, success_style, err_style, warning_style, dim_style, head_style, subhead_style, row_style
from auto_eval.config import GENERATION_DIRS

# ---------------------------------------------
# Configuration
# ---------------------------------------------
INCOMING_DIR = GENERATION_DIRS["incoming"]
PROCESSING_DIR = GENERATION_DIRS["processing"]
DONE_DIR = GENERATION_DIRS["done"]
FAILED_DIR = GENERATION_DIRS["failed"]

# Path to generation script
GENERATION_SCRIPT_PATH = Path(__file__).parent / "generation_script.py"

# Polling interval (seconds)
POLL_INTERVAL = 5

# Maximum retries for a task
MAX_RETRIES = 3
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

        files.sort(key=lambda f: f.stat().st_mtime)  # Sort by modification time
        return files[0]  # Return the oldest file
    except FileNotFoundError:
        return None


def process_job(job_file_path):
    """
    Process a single generation job file.
    """
    job_filename = job_file_path.name
    processing_job_path = PROCESSING_DIR / job_filename

    console.print(f"\n[Generation Sentinel] Found new job: {job_filename}\n", style=head_style)

    # 1. "Lock" the job (move to processing directory)
    try:
        shutil.move(str(job_file_path), str(processing_job_path))
        console.print(f"[Generation Sentinel] Moved job {job_filename} -> processing/")
    except Exception as e:
        console.print(f"[Generation Sentinel] Unable to lock job {job_filename}: {e}. Skipping.", style=err_style)
        return

    # 2. Read job configuration
    try:
        with open(processing_job_path, 'r', encoding='utf-8') as f:
            task_config = json.load(f)

        task_id = task_config.get('task_id', job_filename)
        max_retries = task_config.get('max_retries', MAX_RETRIES)
        retry_count = task_config.get('retry_count', 0)

        # Add started_at timestamp
        if 'started_at' not in task_config:
            task_config['started_at'] = datetime.now().isoformat()
            task_config['status'] = 'processing'
            with open(processing_job_path, 'w', encoding='utf-8') as f:
                json.dump(task_config, f, ensure_ascii=False, indent=4)

    except Exception as e:
        console.print(f"[Generation Sentinel] Unable to read JSON content of {job_filename}: {e}", style=err_style)
        shutil.move(str(processing_job_path), str(FAILED_DIR / job_filename))
        return

    # 3. Execute generation script with retry
    for attempt in range(retry_count, max_retries):
        console.print(f"[Generation Sentinel] Executing job {task_id} (attempt {attempt + 1}/{max_retries})...", style=dim_style)

        try:
            result = subprocess.run(
                ["python3", str(GENERATION_SCRIPT_PATH), str(processing_job_path)],
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            # 4. Success
            console.print(f"[Generation Sentinel] Job {task_id} succeeded.", style=success_style)

            # Read updated task config to check status
            with open(processing_job_path, 'r', encoding='utf-8') as f:
                task_config = json.load(f)

            # Ensure finished_at is set
            if 'finished_at' not in task_config:
                task_config['finished_at'] = datetime.now().isoformat()
                with open(processing_job_path, 'w', encoding='utf-8') as f:
                    json.dump(task_config, f, ensure_ascii=False, indent=4)

            shutil.move(str(processing_job_path), str(DONE_DIR / job_filename))
            return

        except subprocess.CalledProcessError as e:
            console.print(f"[Generation Sentinel] Job {task_id} (attempt {attempt + 1}) failed!", style=err_style)
            console.print(f"\n[Generation Sentinel] Stderr:\n{e.stderr}\n", style=err_style)

            # Update retry count
            task_config['retry_count'] = attempt + 1

            if 'error_log' not in task_config:
                task_config['error_log'] = {
                    'sentinel_error': {
                        'exit_code': e.returncode,
                        'stdout': e.stdout[:2000] if e.stdout else '',  # Truncate long output
                        'stderr': e.stderr[:2000] if e.stderr else '',
                        'attempt': attempt + 1
                    }
                }

            try:
                with open(processing_job_path, 'w', encoding='utf-8') as f:
                    json.dump(task_config, f, ensure_ascii=False, indent=4)
            except Exception as write_e:
                console.print(f"[Generation Sentinel] Unable to update retry count: {write_e}", style=err_style)

            if attempt < max_retries - 1:
                console.print(f"[Generation Sentinel] Waiting 30 seconds before retrying...", style=warning_style)
                time.sleep(30)
            else:
                # Max retries reached
                console.print(f"[Generation Sentinel] Job {task_id} reached max retries, moving to failed/", style=err_style)

                task_config['status'] = 'failed'
                task_config['finished_at'] = datetime.now().isoformat()
                with open(processing_job_path, 'w', encoding='utf-8') as f:
                    json.dump(task_config, f, ensure_ascii=False, indent=4)

                shutil.move(str(processing_job_path), str(FAILED_DIR / job_filename))
                return

        except Exception as e:
            console.print(f"[Generation Sentinel] Critical error: {e}", style=err_style)

            try:
                task_config['status'] = 'failed'
                task_config['finished_at'] = datetime.now().isoformat()
                task_config['error_log'] = {
                    'exception': str(e),
                    'exception_type': type(e).__name__
                }
                with open(processing_job_path, 'w', encoding='utf-8') as f:
                    json.dump(task_config, f, ensure_ascii=False, indent=4)
            except:
                pass

            shutil.move(str(processing_job_path), str(FAILED_DIR / job_filename))
            return


def is_processing_busy():
    """
    Check if there is already a job being processed.
    """
    try:
        files = list(PROCESSING_DIR.glob('*.json'))
        return len(files) > 0
    except Exception:
        return False


def main_loop():
    console.print("\n\nGeneration Sentinel Worker Started\n\n", style=head_style, justify="center")
    console.print(f"Monitoring: {INCOMING_DIR}", style=row_style, justify="center")
    console.print(f"Processing: {PROCESSING_DIR}", style=row_style, justify="center")
    console.print(f"Done: {DONE_DIR}", style=row_style, justify="center")
    console.print(f"Failed: {FAILED_DIR}", style=row_style, justify="center")
    console.print("")

    # Ensure all directories exist
    for d in [INCOMING_DIR, PROCESSING_DIR, DONE_DIR, FAILED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    while True:
        # Check if processing directory is busy
        if is_processing_busy():
            console.print(f"... (Processing busy, waiting @ {time.strftime('%H:%M:%S')}) ...", style=dim_style, end="\r")
            time.sleep(POLL_INTERVAL)
            continue

        job_file = find_next_job(INCOMING_DIR)

        if job_file:
            process_job(job_file)
        else:
            # No jobs, take a break
            console.print(f"... (Waiting for generation tasks @ {time.strftime('%H:%M:%S')}) ...", style=dim_style, end="\r")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
