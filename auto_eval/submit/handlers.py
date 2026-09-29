"""
Business logic for task submission
"""
import os
import html
import json
import datetime
import random
import string
import time
import pandas as pd
import gradio as gr
from pathlib import Path
from typing import Tuple, List

from auto_eval.config import (
    QUEUE_DIRS, CPU_QUEUE_DIRS, QUEUE_BASE_DIR, MAX_RETRIES, EMAIL_DOMAIN, MAX_RECENT_TASKS,
    get_all_worker_dirs,
    RESULTS_BASE_DIR, LOG_BASE_DIR, VERSION
)
from auto_eval.utils import validate_email as validate_email_util, build_email_from_username as build_email_util
from auto_eval.tasks_meta import get_tasks_metadata
from auto_eval.submit.core import build_eval_task, submit_eval_task, generate_output_suffix


def _gr_warn(msg: str) -> None:
    try:
        gr.Warning(msg, duration=3)
    except TypeError:
        gr.Warning(msg)


def _gr_info(msg: str) -> None:
    try:
        gr.Info(msg, duration=3)
    except TypeError:
        gr.Info(msg)


# Queue directories
INCOMING_DIR = QUEUE_DIRS["incoming"]
INCOMING_CPU_DIR = CPU_QUEUE_DIRS["incoming"]
PROCESSING_DIR = QUEUE_DIRS["processing"]
DONE_DIR = QUEUE_DIRS["done"]
FAILED_DIR = QUEUE_DIRS["failed"]
DONE_CPU_DIR = CPU_QUEUE_DIRS["done"]
FAILED_CPU_DIR = CPU_QUEUE_DIRS["failed"]


def validate_email(email: str) -> bool:
    """Validate email format"""
    return validate_email_util(email)


def build_email_from_username(username: str) -> str:
    """Build email from username"""
    return build_email_util(username, EMAIL_DOMAIN)


def get_think_display(enable_thinking: bool) -> str:
    """
    Get the display string for think status.

    Args:
        enable_thinking: Whether thinking is enabled

    Returns:
        Display string: "Think" or "Non Think"
    """
    return "Think" if enable_thinking else "Non Think"


def get_tasks_display(selected_tasks: list, all_tasks: list) -> str:
    """
    Get the display string for selected tasks.

    Args:
        selected_tasks: List of selected tasks
        all_tasks: List of all available tasks

    Returns:
        "All" if all tasks selected, otherwise comma-separated list
    """
    if set(selected_tasks) == set(all_tasks):
        return "All"
    return ",".join(selected_tasks)


# Dynamic task metadata from registry
DEFAULT_VERSION = "v3.1"
DEFAULT_TASKS = get_tasks_metadata(DEFAULT_VERSION)["all_tasks"]
DEFAULT_SELECTED_TASKS = []


def parse_paths_to_preview(email: str, model_paths_text: str, enable_thinking: bool = False, rollout_mode: str = "Race", selected_tasks: list = None, version: str = "v3.1", seed: int = 42) -> pd.DataFrame:
    """
    Parse model paths and generate preview DataFrame.

    Args:
        email: User email (not used currently, kept for compatibility)
        model_paths_text: Multi-line model paths
        enable_thinking: Enable thinking mode
        rollout_mode: Rollout mode
        selected_tasks: List of selected evaluation tasks

    Returns:
        DataFrame with columns: Model Path, Think, Output Suffix, Task, Status
    """
    print(f"[DEBUG] parse_paths_to_preview called...")
    print(f"[DEBUG] enable_thinking: {enable_thinking}")
    print(f"[DEBUG] selected_tasks: {selected_tasks}")

    if selected_tasks is None:
        selected_tasks = get_tasks_metadata(version)["default_selected"].copy()

    preview_cols = ["Model Path", "Think", "Output Suffix", "Task", "Status"]

    if not model_paths_text or not model_paths_text.strip():
        print("[DEBUG] Empty model_paths_text")
        return pd.DataFrame(columns=preview_cols)

    paths = model_paths_text.strip().split('\n')
    preview_data = []

    # Calculate think display and tasks display (same for all rows)
    think_display = get_think_display(enable_thinking)
    version_tasks = get_tasks_metadata(version)["all_tasks"]
    tasks_display = get_tasks_display(selected_tasks, version_tasks)

    for path in paths:
        path = path.strip()
        if not path:
            continue

        print(f"[DEBUG] Processing path: {path}")

        # Validate path exists
        status = "✅ 模型路径存在"
        try:
            if not os.path.exists(path):
                status = "❌ 模型路径不存在"
        except Exception as e:
            print(f"[DEBUG] Error checking path existence: {e}")
            status = f"⚠️ 检查失败: {e}"

        # Generate output_suffix
        suffix = generate_output_suffix(path, enable_thinking, rollout_mode, version, seed)
        print(f"[DEBUG] Generated suffix: {suffix}")

        preview_data.append([path, think_display, suffix, tasks_display, status])

    print(f"[DEBUG] Total preview rows: {len(preview_data)}")
    df = pd.DataFrame(preview_data, columns=preview_cols)
    return df


def update_derived_paths(df: pd.DataFrame, model_paths_text: str, version: str) -> pd.DataFrame:
    """
    Update derived paths when user edits Output Suffix column.

    Args:
        df: User-edited DataFrame
        model_paths_text: Original model paths text
        version: Version string passed from JSON config

    Returns:
        Updated DataFrame with recalculated derived paths
    """
    print("[DEBUG] update_derived_paths called...")
    try:
        # Re-parse original paths to get source data
        original_df = parse_paths_to_preview(None, model_paths_text)

        # Extract user-edited Output Suffix column
        user_suffixes = df["Output Suffix"].tolist()

        updated_data = []

        # Iterate through source data
        for index, row in original_df.iterrows():
            if index < len(user_suffixes):
                # Accept user-modified Suffix
                new_suffix = user_suffixes[index]

                # Recalculate derived columns
                new_output_dir = f"{RESULTS_BASE_DIR}/{version}/results_{new_suffix}/"
                new_log_path = f"{LOG_BASE_DIR}/{version}/{new_suffix}.log"

                # Use original Model Path and Status, with new Suffix and derived columns
                updated_data.append([
                    row["Model Path"],  # Restore original
                    new_suffix,         # Accept user edit
                    new_output_dir,     # Force recalculate
                    new_log_path,       # Force recalculate
                    row["Status"]       # Restore original
                ])

        return pd.DataFrame(updated_data, columns=["Model Path", "Output Suffix", "Output Dir", "Log Path", "Status"])
    except Exception as e:
        print(f"[ERROR] Exception in update_derived_paths: {e}")
        return df  # Return original on error


def submit_tasks(username: str, preview_df: pd.DataFrame, enable_thinking: bool = False, rollout_mode: str = "Race", selected_tasks: list = None, overwrite: bool = False, version: str = "v3.1", seed: int = 42, compute_cot_metrics: bool = False, tensor_parallel_size: int = 1) -> Tuple:
    """
    Submit tasks to incoming queue.

    Args:
        username: Username or email
        preview_df: Preview DataFrame with task information
        enable_thinking: Enable thinking mode
        rollout_mode: Rollout mode
        selected_tasks: List of selected evaluation tasks
        compute_cot_metrics: Whether to compute CoT quality metrics (delta_ll/
            ehr/cov_rate/sid_valid_rate). Only meaningful under
            rollout_mode=Default with enable_thinking=True; caller is expected
            to have already gated on those conditions.
        tensor_parallel_size: Number of GPUs used by each model replica.

    Returns:
        Tuple of (submit_area_visible, new_task_btn_visible, *queue_status_updates)
    """
    # Failure return: 6 gr.update()
    fail_return = (
        gr.update(visible=True),   # submit_area (keep visible)
        gr.update(visible=False),  # new_task_btn (keep hidden)
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )

    if selected_tasks is None:
        selected_tasks = DEFAULT_SELECTED_TASKS.copy()

    # Build complete email from username
    email = build_email_from_username(username)

    if not validate_email(email):
        print(f"[ERROR] Invalid email: {email}")
        _gr_warn(f"❌ 邮箱格式无效: {email}")
        return fail_return

    if preview_df is None or preview_df.empty:
        print("[ERROR] Preview DataFrame is empty")
        _gr_warn("❌ 请先输入模型路径")
        return fail_return

    # Check for invalid paths
    if "❌ 模型路径不存在" in preview_df["Status"].values:
        print("[ERROR] Invalid model paths detected")
        _gr_warn("❌ 存在无效的模型路径,请检查后重试")
        return fail_return

    # Check that at least one task is selected
    if not selected_tasks:
        print("[ERROR] No tasks selected")
        _gr_warn("❌ 请至少选择一个评估任务")
        return fail_return

    submitted_count = 0

    for index, row in preview_df.iterrows():
        model_path = row["Model Path"]
        output_suffix = row["Output Suffix"]

        task_data = build_eval_task(
            model_path=model_path,
            output_suffix=output_suffix,
            submitter_email=email,
            version=version,
            enable_thinking=enable_thinking,
            rollout_mode=rollout_mode,
            seed=seed,
            overwrite=overwrite,
            tasks=selected_tasks,
            compute_cot_metrics=compute_cot_metrics,
            tensor_parallel_size=tensor_parallel_size,
        )
        task_id = task_data["task_id"]

        # Submit to shared incoming queue (workers will compete for tasks)
        print(f"[INFO] Submitting task to shared incoming queue: {INCOMING_DIR}")

        try:
            json_filepath = submit_eval_task(task_data, queue_dir=INCOMING_DIR)
            submitted_count += 1
            print(f"[SUCCESS] Task submitted to shared queue: {json_filepath}")
        except Exception as e:
            json_filename = f"{task_id}.json"
            print(f"[ERROR] Failed to write task file {json_filename}: {e}")
            _gr_warn(f"❌ 提交失败: 无法写入任务文件 {json_filename}\n错误: {str(e)}")
            return fail_return

        # Sleep between task submissions to prevent race conditions
        # This ensures the file system is synced and load balancing is accurate
        if index < len(preview_df) - 1:  # Don't sleep after the last task
            time.sleep(1)  # 1 second delay between submissions

    # After successful submission, hide submit area and refresh queue status
    status_updates = get_queue_status()

    # Show success message
    print(f"[SUCCESS] Total {submitted_count} task(s) submitted successfully")
    _gr_info(f"✅ 成功提交 {submitted_count} 个任务!")

    # Success return: 6 values
    return (
        gr.update(visible=False),  # submit_area (hide)
        gr.update(visible=True),   # new_task_btn (show)
        status_updates[0], status_updates[1], status_updates[2], status_updates[3]
    )


def parse_api_model_to_preview(api_model: str, selected_tasks: list = None, version: str = "v3.1", seed: int = 42) -> pd.DataFrame:
    """Generate preview DataFrame for a closed-source API model."""
    preview_cols = ["Model Path", "Think", "Output Suffix", "Task", "Status"]

    if not api_model:
        return pd.DataFrame(columns=preview_cols)

    if selected_tasks is None:
        selected_tasks = get_tasks_metadata(version)["default_selected"].copy()

    version_tasks = get_tasks_metadata(version)["all_tasks"]
    tasks_display = get_tasks_display(selected_tasks, version_tasks)

    suffix = f"{version}_{api_model}"
    if seed != 42:
        suffix += f"_seed{seed}"

    preview_data = [[f"[API] {api_model}", "N/A", suffix, tasks_display, "✅ API 模型"]]
    return pd.DataFrame(preview_data, columns=preview_cols)


def submit_api_tasks(username: str, api_model: str, selected_tasks: list = None,
                     overwrite: bool = False, version: str = "v3.1", seed: int = 42) -> Tuple:
    """
    Submit a closed-source model evaluation task to the CPU queue (incoming_cpu/).
    A CPU sentinel worker will pick it up and execute via API calls.
    """
    fail_return = (
        gr.update(visible=True),
        gr.update(visible=False),
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )

    if selected_tasks is None:
        selected_tasks = get_tasks_metadata(version)["default_selected"].copy()

    email = build_email_from_username(username)
    if not validate_email(email):
        _gr_warn(f"❌ 邮箱格式无效: {email}")
        return fail_return

    if not api_model:
        _gr_warn("❌ 请选择闭源模型")
        return fail_return

    if not selected_tasks:
        _gr_warn("❌ 请至少选择一个评估任务")
        return fail_return

    output_suffix = f"{version}_{api_model}"
    if seed != 42:
        output_suffix += f"_seed{seed}"

    email_user = email.split('@')[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    task_id = f"{timestamp}_{email_user}_{output_suffix}_{rand_suffix}"

    log_path = f"{LOG_BASE_DIR}/{version}/{output_suffix}.log"
    task_data = {
        "task_id": task_id,
        "model_type": api_model,
        "model_path": "",
        "log_path": str(log_path),
        "output_suffix": output_suffix,
        "version": version,
        "tasks": selected_tasks,
        "seed": seed,
        "overwrite": overwrite,
        "submitter_email": email,
        "submitted_at": datetime.datetime.now().isoformat(),
        "max_retries": MAX_RETRIES,
        "retry_count": 0,
    }

    # Write to CPU queue (separate from GPU incoming/)
    json_filename = f"{task_id}.json"
    json_filepath = INCOMING_CPU_DIR / json_filename

    try:
        INCOMING_CPU_DIR.mkdir(parents=True, exist_ok=True)
        with open(json_filepath, 'w') as f:
            json.dump(task_data, f, indent=4)
        print(f"[SUCCESS] API task submitted to CPU queue: {json_filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to write task file {json_filepath}: {e}")
        _gr_warn(f"❌ 提交失败: 无法写入任务文件\n错误: {str(e)}")
        return fail_return

    _gr_info(f"✅ 闭源模型评测任务已提交到 CPU 队列 ({api_model})")

    status_updates = get_queue_status()
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        status_updates[0], status_updates[1], status_updates[2], status_updates[3]
    )


def format_timestamp(timestamp_str: str) -> str:
    """
    Format ISO timestamp to readable format.

    Args:
        timestamp_str: ISO format timestamp

    Returns:
        Formatted timestamp (YYYY-MM-DD HH:MM:SS)
    """
    try:
        dt = datetime.datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "N/A"


def format_submitter(email: str) -> str:
    """
    Format submitter email by removing @example.org suffix.

    Args:
        email: Submitter email

    Returns:
        Username without domain
    """
    if '@' in email:
        return email.split('@')[0]
    return email


def extract_task_id_hash(task_json_name: str) -> str:
    """
    Extract the hash part from task JSON filename.

    Args:
        task_json_name: Full task JSON filename (e.g., "20251123_012017_xxx_suffix_rj7s.json")

    Returns:
        Hash part (e.g., "rj7s")
    """
    try:
        # Remove .json extension and get the last part after underscore
        base_name = task_json_name.replace('.json', '')
        parts = base_name.split('_')
        if len(parts) >= 1:
            return parts[-1]  # Return last part (the hash)
        return task_json_name
    except Exception:
        return task_json_name


def get_task_info_pending(json_path: Path) -> List[str]:
    """
    Get task information for pending queue.

    Returns:
        List of [model_path, think_display, output_dir, log_path, task_id, task_display, submit_time, submitter]
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        model_path = data.get('model_path', 'N/A')
        model_type = data.get('model_type', '')
        if model_type and not model_path:
            model_path = f"[API] {model_type}"
        output_suffix = data.get('output_suffix', 'N/A')
        version = data.get('version', 'v3.1')
        enable_thinking = data.get('enable_thinking', False)
        all_tasks = get_tasks_metadata(version)["all_tasks"]
        tasks = data.get('tasks', all_tasks)
        submitter_email = data.get('submitter_email', 'N/A')
        submitted_at = data.get('submitted_at', 'N/A')

        # Generate output_dir and log_path
        output_dir = f"{RESULTS_BASE_DIR}/{version}/results_{output_suffix}/"
        log_path = f"{LOG_BASE_DIR}/{version}/{output_suffix}.log"

        # Get Task ID (hash only)
        task_id = extract_task_id_hash(json_path.name)

        # Format submit time and submitter
        submit_time = format_timestamp(submitted_at)
        submitter = format_submitter(submitter_email)

        # Format think display
        think_display = "API" if model_type else get_think_display(enable_thinking)

        # Format tasks display
        tasks_display = get_tasks_display(tasks, all_tasks)

        return [model_path, think_display, output_dir, log_path, task_id, tasks_display, submit_time, submitter]
    except Exception:
        return ["N/A", "Think", "N/A", "N/A", "N/A", "All", "N/A", "N/A"]


def get_task_info_processing(json_path: Path) -> List[str]:
    """
    Get task information for processing queue.

    Returns:
        List of [model_path, think_display, output_dir, log_path, task_id, tasks_display, start_time, submitter]
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        model_path = data.get('model_path', 'N/A')
        model_type = data.get('model_type', '')
        if model_type and not model_path:
            model_path = f"[API] {model_type}"
        output_suffix = data.get('output_suffix', 'N/A')
        version = data.get('version', 'v3.1')
        enable_thinking = data.get('enable_thinking', False)
        all_tasks = get_tasks_metadata(version)["all_tasks"]
        tasks = data.get('tasks', all_tasks)
        submitter_email = data.get('submitter_email', 'N/A')
        started_at = data.get('started_at', 'N/A')

        # Generate output_dir and log_path
        output_dir = f"{RESULTS_BASE_DIR}/{version}/results_{output_suffix}/"
        log_path = f"{LOG_BASE_DIR}/{version}/{output_suffix}.log"

        # Get Task ID (hash only)
        task_id = extract_task_id_hash(json_path.name)

        # Format start time and submitter
        start_time = format_timestamp(started_at)
        submitter = format_submitter(submitter_email)

        # Format think display
        think_display = "API" if model_type else get_think_display(enable_thinking)

        # Format tasks display
        tasks_display = get_tasks_display(tasks, all_tasks)

        return [model_path, think_display, output_dir, log_path, task_id, tasks_display, start_time, submitter]
    except Exception:
        return ["N/A", "Think", "N/A", "N/A", "N/A", "All", "N/A", "N/A"]


def get_task_info_done_or_failed(json_path: Path) -> List[str]:
    """
    Get task information for done/failed queue.

    Returns:
        List of [model_path, think_display, output_dir, log_path, task_id, tasks_display, start_time, end_time, submitter]
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        model_path = data.get('model_path', 'N/A')
        model_type = data.get('model_type', '')
        if model_type and not model_path:
            model_path = f"[API] {model_type}"
        output_suffix = data.get('output_suffix', 'N/A')
        version = data.get('version', 'v3.1')
        enable_thinking = data.get('enable_thinking', False)
        all_tasks = get_tasks_metadata(version)["all_tasks"]
        tasks = data.get('tasks', all_tasks)
        submitter_email = data.get('submitter_email', 'N/A')
        started_at = data.get('started_at', 'N/A')
        finished_at = data.get('finished_at', 'N/A')

        # Generate output_dir and log_path
        output_dir = f"{RESULTS_BASE_DIR}/{version}/results_{output_suffix}/"
        log_path = f"{LOG_BASE_DIR}/{version}/{output_suffix}.log"

        # Get Task ID (hash only)
        task_id = extract_task_id_hash(json_path.name)

        # Format times and submitter
        start_time = format_timestamp(started_at)
        end_time = format_timestamp(finished_at)
        submitter = format_submitter(submitter_email)

        # Format think display
        think_display = "API" if model_type else get_think_display(enable_thinking)

        # Format tasks display
        tasks_display = get_tasks_display(tasks, all_tasks)

        return [model_path, think_display, output_dir, log_path, task_id, tasks_display, start_time, end_time, submitter]
    except Exception:
        return ["N/A", "Think", "N/A", "N/A", "N/A", "All", "N/A", "N/A", "N/A"]


def dataframe_to_html(data: List[List[str]], headers: List[str], table_id: str, is_deletable: bool = False) -> str:
    """
    Convert data to HTML table string.

    Args:
        data: List of rows, each row is a list of strings
        headers: List of header strings
        table_id: ID for the table
        is_deletable: Whether to add checkbox column

    Returns:
        HTML string
    """
    safe_table_id = html.escape(str(table_id))

    if not data:
        # Return empty table with headers
        buf = f'<div class="table-wrap"><table id="{safe_table_id}" class="custom-queue-table">'
        buf += '<thead><tr>'
        for h in headers:
            buf += f'<th>{html.escape(str(h))}</th>'
        buf += '</tr></thead><tbody>'
        buf += f'<tr><td colspan="{len(headers)}" style="text-align: center; padding: 20px;"></td></tr>'
        buf += '</tbody></table></div>'
        return buf

    buf = f'<div class="table-wrap"><table id="{safe_table_id}" class="custom-queue-table">'

    # Header
    buf += '<thead><tr>'
    for i, h in enumerate(headers):
        # First column width for deletable tables
        style = 'style="width: 40px; min-width: 40px; max-width: 40px;"' if is_deletable and i == 0 else ''
        buf += f'<th {style}>{html.escape(str(h))}</th>'
    buf += '</tr></thead>'

    # Body
    buf += '<tbody>'
    for row_idx, row in enumerate(data):
        buf += '<tr>'
        for col_idx, cell in enumerate(row):
            # First column checkbox for deletable tables
            if is_deletable and col_idx == 0:
                buf += f'<td style="width: 40px; min-width: 40px; max-width: 40px; text-align: center;">'
                buf += f'<input type="checkbox" class="del-checkbox" data-row="{html.escape(str(row_idx))}">'
                buf += '</td>'
            else:
                buf += f'<td>{html.escape(str(cell))}</td>'
        buf += '</tr>'
    buf += '</tbody></table></div>'

    return buf


def get_queue_status() -> Tuple[str, str, str, str]:
    """
    Scan queue directories and return status HTML strings.
    Aggregates tasks from all workers for incoming and processing queues.

    Returns:
        Tuple of (pending_html, processing_html, done_html, failed_html)
    """
    def get_files(directory: Path) -> List[Path]:
        try:
            if not directory.exists():
                return []
            files = [f for f in directory.glob('*.json') if f.is_file()]
            return files
        except Exception as e:
            print(f"[ERROR] Error reading directory {directory}: {e}")
            return []

    try:
        # Get all worker directories
        all_worker_dirs = get_all_worker_dirs()

        # Aggregate pending (incoming) tasks from shared incoming and all workers
        pending_files = []
        # Always scan shared incoming directory (GPU)
        pending_files.extend(get_files(INCOMING_DIR))
        # Scan CPU incoming directory
        pending_files.extend(get_files(INCOMING_CPU_DIR))
        # Also scan all worker-specific incoming directories
        if all_worker_dirs:
            for worker_id, dirs in all_worker_dirs.items():
                pending_files.extend(get_files(dirs['incoming']))

        # Aggregate processing tasks from all workers (GPU + CPU)
        processing_files = []
        if all_worker_dirs:
            for worker_id, dirs in all_worker_dirs.items():
                processing_files.extend(get_files(dirs['processing']))
        else:
            # Fall back to legacy mode
            processing_files = get_files(PROCESSING_DIR)
        # Also scan all CPU processing directories (processing_cpu*)
        for d in QUEUE_BASE_DIR.glob("processing_cpu*"):
            if d.is_dir():
                processing_files.extend(get_files(d))

        # Done and failed are shared across all workers (GPU + CPU)
        done_files = get_files(DONE_DIR) + get_files(DONE_CPU_DIR)
        failed_files = get_files(FAILED_DIR) + get_files(FAILED_CPU_DIR)

        # Sort by modification time (newest first)
        pending_files = sorted(pending_files, key=lambda f: f.stat().st_mtime, reverse=True)
        processing_files = sorted(processing_files, key=lambda f: f.stat().st_mtime, reverse=True)
        done_files = sorted(done_files, key=lambda f: f.stat().st_mtime, reverse=True)
        failed_files = sorted(failed_files, key=lambda f: f.stat().st_mtime, reverse=True)

        # Extract information
        pending_tasks = [get_task_info_pending(f) for f in pending_files]
        processing_tasks = [get_task_info_processing(f) for f in processing_files]
        done_tasks = [get_task_info_done_or_failed(f) for f in done_files[:MAX_RECENT_TASKS]]
        failed_tasks = [get_task_info_done_or_failed(f) for f in failed_files[:MAX_RECENT_TASKS]]

        # Define headers
        headers_pending = ["", "Model Path", "Think", "Output Dir", "Log Path", "Task ID", "Task", "Submit Time", "Submitter"]
        headers_processing = ["Model Path", "Think", "Output Dir", "Log Path", "Task ID", "Task", "Start Time", "Submitter"]
        headers_done_failed = ["", "Model Path", "Think", "Output Dir", "Log Path", "Task ID", "Task", "Start Time", "End Time", "Submitter"]

        # Generate HTML
        # Pending: Add empty string for first column (checkbox placeholder)
        pending_data = [[""] + task for task in pending_tasks]
        html_pending = dataframe_to_html(pending_data, headers_pending, "pending-table", is_deletable=True)

        # Processing: No checkbox
        html_processing = dataframe_to_html(processing_tasks, headers_processing, "processing-table", is_deletable=False)

        # Done: Add empty string for first column
        done_data = [[""] + task for task in done_tasks]
        html_done = dataframe_to_html(done_data, headers_done_failed, "done-table", is_deletable=True)

        # Failed: Add empty string for first column
        failed_data = [[""] + task for task in failed_tasks]
        html_failed = dataframe_to_html(failed_data, headers_done_failed, "failed-table", is_deletable=True)

        return html_pending, html_processing, html_done, html_failed

    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in get_queue_status: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return (
            "<div>Error loading pending queue</div>",
            "<div>Error loading processing queue</div>",
            "<div>Error loading done queue</div>",
            "<div>Error loading failed queue</div>"
        )


def delete_selected_tasks(data_json: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Delete selected tasks based on data from frontend.
    Searches across all worker directories for pending tasks.

    Args:
        data_json: JSON string with format: {"taskJsons": [...], "queueType": "pending"|"done"|"failed"}

    Returns:
        Updated queue status DataFrames
    """
    import json as json_lib

    try:
        print(f"[DEBUG] Raw input: '{data_json}'")
        print(f"[DEBUG] Input type: {type(data_json)}")

        data = json_lib.loads(data_json) if data_json else {}
        task_jsons = data.get('taskJsons', [])
        queue_type = data.get('queueType', '')

        print(f"[DEBUG] Parsed task_jsons: {task_jsons}")
        print(f"[DEBUG] Queue type: {queue_type}")
        print(f"[DEBUG] Task JSON files count: {len(task_jsons)}")

        if not task_jsons:
            _gr_warn("⚠️ 请先选择要删除的任务")
            return get_queue_status()

        deleted_count = 0

        def find_file_by_task_id(directory: Path, task_id: str):
            """Find a JSON file in directory that ends with the given task_id hash."""
            if not directory.exists():
                return None
            # Task ID is the hash at the end of filename: *_{task_id}.json
            for json_file in directory.glob(f"*_{task_id}.json"):
                return json_file
            return None

        for task_id in task_jsons:
            if not task_id or task_id == "N/A":
                continue

            # For pending tasks, search across shared incoming and all worker directories
            if queue_type == 'pending':
                all_worker_dirs = get_all_worker_dirs()
                found = False

                # First, try shared incoming directory
                json_filepath = find_file_by_task_id(INCOMING_DIR, task_id)
                if json_filepath and json_filepath.exists():
                    try:
                        json_filepath.unlink()
                        deleted_count += 1
                        found = True
                        print(f"[SUCCESS] Deleted task from shared incoming: {json_filepath}")
                    except Exception as e:
                        print(f"[ERROR] Failed to delete {json_filepath}: {e}")
                        _gr_warn(f"❌ 删除失败: {task_id}\n错误: {str(e)}")

                # If not found in shared incoming, search CPU incoming
                if not found:
                    json_filepath = find_file_by_task_id(INCOMING_CPU_DIR, task_id)
                    if json_filepath and json_filepath.exists():
                        try:
                            json_filepath.unlink()
                            deleted_count += 1
                            found = True
                            print(f"[SUCCESS] Deleted task from CPU incoming: {json_filepath}")
                        except Exception as e:
                            print(f"[ERROR] Failed to delete {json_filepath}: {e}")

                # If not found, search all worker-specific incoming directories
                if not found and all_worker_dirs:
                    for worker_id, dirs in all_worker_dirs.items():
                        json_filepath = find_file_by_task_id(dirs['incoming'], task_id)
                        if json_filepath and json_filepath.exists():
                            try:
                                json_filepath.unlink()
                                deleted_count += 1
                                found = True
                                print(f"[SUCCESS] Deleted task from worker {worker_id}: {json_filepath}")
                                break
                            except Exception as e:
                                print(f"[ERROR] Failed to delete {json_filepath}: {e}")
                                _gr_warn(f"❌ 删除失败: {task_id}\n错误: {str(e)}")

                if not found:
                    print(f"[WARNING] File not found for task_id: {task_id}")

            # For done/failed tasks, search GPU and CPU directories
            elif queue_type in ['done', 'failed']:
                if queue_type == 'done':
                    directories = [DONE_DIR, DONE_CPU_DIR]
                else:
                    directories = [FAILED_DIR, FAILED_CPU_DIR]

                found_done = False
                for directory in directories:
                    json_filepath = find_file_by_task_id(directory, task_id)
                    if json_filepath and json_filepath.exists():
                        try:
                            json_filepath.unlink()
                            deleted_count += 1
                            found_done = True
                            print(f"[SUCCESS] Deleted task: {json_filepath}")
                        except Exception as e:
                            print(f"[ERROR] Failed to delete {json_filepath}: {e}")
                            _gr_warn(f"❌ 删除失败: {task_id}\n错误: {str(e)}")
                        break

                if not found_done:
                    print(f"[WARNING] File not found for task_id: {task_id}")

            else:
                _gr_warn(f"❌ 无效的队列类型: {queue_type}")
                continue

        if deleted_count > 0:
            _gr_info(f"✅ 成功删除 {deleted_count} 个任务")
        else:
            _gr_warn("⚠️ 未找到需要删除的任务文件")

        # Refresh queue status
        return get_queue_status()
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in delete_selected_tasks: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        _gr_warn(f"❌ 删除操作失败: {str(e)}")
        return get_queue_status()
