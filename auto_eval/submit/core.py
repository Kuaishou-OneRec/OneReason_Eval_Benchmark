"""
Core module for building and submitting GPU evaluation tasks.
Used by both the Web UI (handlers.py) and the CLI (submit_cli.py).
"""
import json
import re
import datetime
import random
import string
from pathlib import Path
from typing import List, Optional, Union

from auto_eval.config import QUEUE_DIRS, LOG_BASE_DIR, MAX_RETRIES
from auto_eval.tasks_meta import get_tasks_metadata


def generate_output_suffix(model_path: str, enable_thinking: bool = False, rollout_mode: str = "Race", version: str = "v3.1", seed: int = 42) -> str:
    """
    Generate output_suffix from model path.

    Format 1 (verl/actor): Extract directory before 'global_step' + step number
    Example: .../label_cond_reco_1112.../global_step_1000/actor/...
    Output: label_cond_reco_1112..._step1000

    Format 2 (pretrain): Use original logic - extract 3 directories before 'global_step'
    Example: .../sft/v0.0.1_from_stg2_v0.1.1_8b/step20/global_step20/...
    Output: sft_v0.0.1_from_stg2_v0.1.1_8b_step20

    Args:
        model_path: Model path string
        enable_thinking: If True, append "_think" suffix
        rollout_mode: Rollout mode

    Returns:
        Output suffix string
    """
    try:
        # Strip known prefixes from model_path
        prefixes_to_strip = []
        stripped_path = model_path.strip()
        for prefix in prefixes_to_strip:
            if stripped_path.startswith(prefix):
                stripped_path = stripped_path[len(prefix):]
                break

        # Split and filter out empty strings (caused by double slashes or trailing slashes)
        parts = [p for p in stripped_path.split('/') if p]

        # Search backward for 'global_step'
        idx = -1
        global_step_num = ""
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].startswith("global_step"):
                idx = i
                # Extract step number from global_step_XXX
                global_step_num = parts[i].replace("global_step_", "").replace("global_step", "")
                break

        # Build suffix from all parts before global_step (or all parts if not found)
        suffix = ""
        if idx != -1:
            # Use all parts before global_step, then append _stepXXX
            # Filter out pure "stepXXX" directories to avoid duplication with global_step number
            suffix_parts = [p for p in parts[:idx] if not re.fullmatch(r'step\d+', p)]
            suffix = "_".join(suffix_parts) if suffix_parts else ""
            if global_step_num:
                suffix += f"_step{global_step_num}" if suffix else f"step{global_step_num}"
        else:
            # No global_step found: use all parts
            suffix = "_".join(parts)
        suffix = suffix.replace("_converted", "")

        # Append thinking suffix
        if rollout_mode == "Race":
            suffix += "_race"
        elif enable_thinking:
            suffix += "_think"
        else:
            suffix += "_nonthink"

        # Append rollout mode suffix
        if rollout_mode == "Think 64":
            suffix += "_think64"
        elif rollout_mode == "Think 1 Then Beam 32":
            suffix += "_think1bs32"
        elif rollout_mode == "Think 1 Then Beam 1":
            suffix += "_think1bs1"
        elif rollout_mode == "Think 32 Then Beam 1":
            suffix += "_think32bs1"
        elif rollout_mode == "Think 64 Then Beam 1":
            suffix += "_think64bs1"
        elif rollout_mode == "Think 1 Then Beam 32 (temp=1, top_k=-1, top_p=1)":
            suffix += "_think1bs32_temp1_topk-1_topp1"
        elif rollout_mode == "Think 1 Then Beam 1 (temp=1, top_k=-1, top_p=1)":
            suffix += "_think1bs1_temp1_topk-1_topp1"
        elif rollout_mode == "Think 32 Then Beam 1 (temp=1, top_k=-1, top_p=1)":
            suffix += "_think32bs1_temp1_topk-1_topp1"
        elif rollout_mode == "Final":
            suffix += "_final"
        elif rollout_mode == "Pretrain No Chat":
            suffix += "_pretrain_no_chat"
        elif rollout_mode == "No System":
            suffix += "_nosystem"
        elif rollout_mode == "No System No Think":
            suffix += "_nosystem_nothink"
        elif rollout_mode == "No System No Think One N":
            suffix += "_nosystem_nothink_one_n"
        elif rollout_mode == "Think 1 Then Beam 256":
            suffix += "_think1bs256"
        elif rollout_mode == "Think 8 Then Beam 64":
            suffix += "_think8bs64"
        elif rollout_mode == "Think 1 Then Beam 128":
            suffix += "_think1bs128"
        elif rollout_mode == "Beam 4":
            suffix += "_beam4"
        elif rollout_mode == "Beam 32":
            suffix += "_beam32"
        elif rollout_mode == "Beam 512":
            suffix += "_beam512"
        elif rollout_mode == "Greedy Sample":
            suffix += "_greedy_sample"

        # Append seed suffix (only when not default)
        if seed != 42:
            suffix += f"_seed{seed}"

        suffix = f'{version}_' + suffix

        return suffix
    except Exception:
        return "ERROR_PARSING_PATH"


def generate_task_id(output_suffix: str, email: str) -> str:
    email_user = email.split('@')[0] if '@' in email else email
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{timestamp}_{email_user}_{output_suffix}_{rand_suffix}"


def normalize_tasks(tasks: Union[None, str, List[str]], version: str = "v3.1") -> List[str]:
    """
    Normalize tasks input to a list.

    - None / "" / "all" / [] -> [] (executor interprets empty list as all tasks via 'all')
    - "task1,task2" -> ["task1", "task2"]
    - ["task1", "task2"] -> ["task1", "task2"]

    The executor (eval_script.py) converts [] to the string 'all' when calling the shell script,
    which matches the Web UI behavior of passing selected_tasks directly.
    """
    if tasks is None:
        return []
    if isinstance(tasks, list):
        if not tasks:
            return []
        # filter and strip
        result = [t.strip() for t in tasks if t and t.strip()]
        if not result or result == ["all"]:
            return []
        return result
    # string
    s = tasks.strip()
    if not s or s.lower() == "all":
        return []
    return [t.strip() for t in s.split(',') if t.strip()]


def build_eval_task(
    model_path: str,
    output_suffix: str,
    submitter_email: str,
    version: str = "v3.1",
    enable_thinking: bool = False,
    rollout_mode: str = "Race",
    seed: int = 42,
    overwrite: bool = False,
    tasks: Union[None, str, List[str]] = None,
    compute_cot_metrics: bool = False,
    tensor_parallel_size: int = 1,
) -> dict:
    """
    Build the task JSON dict compatible with eval_script.py and the Web UI.
    """
    from benchmark.tasks.tasks import check_benchmark_version, check_task_types
    version = check_benchmark_version(version)
    normalized_tasks = normalize_tasks(tasks, version)
    if normalized_tasks:
        normalized_tasks = check_task_types(normalized_tasks, version)
    log_path = f"{LOG_BASE_DIR}/{version}/{output_suffix}.log"
    task_id = generate_task_id(output_suffix, submitter_email)

    task = {
        "task_id": task_id,
        "model_path": model_path,
        "log_path": str(log_path),
        "output_suffix": output_suffix,
        "version": version,
        "enable_thinking": enable_thinking,
        "rollout_mode": rollout_mode,
        "seed": seed,
        "overwrite": overwrite,
        "tasks": normalized_tasks,
        "tensor_parallel_size": int(tensor_parallel_size),
        "submitter_email": submitter_email,
        "submitted_at": datetime.datetime.now().isoformat(),
        "max_retries": MAX_RETRIES,
        "retry_count": 0,
    }
    if compute_cot_metrics:
        task["compute_cot_metrics"] = True
    return task


def submit_eval_task(task_data: dict, queue_dir: Optional[Path] = None) -> Path:
    """
    Write task_data as a JSON file into the specified queue directory.
    Returns the path of the written file.
    Raises OSError on write failure.
    """
    if queue_dir is None:
        queue_dir = QUEUE_DIRS["incoming"]
    queue_dir = Path(queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)

    task_id = task_data["task_id"]
    json_filename = f"{task_id}.json"
    json_filepath = queue_dir / json_filename

    with open(json_filepath, 'w') as f:
        json.dump(task_data, f, indent=4)

    return json_filepath
