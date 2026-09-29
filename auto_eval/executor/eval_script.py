import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from auto_eval.config import DATA_DIR, DATA_RESULTS_BASE_DIR, QUEUE_LOG_BASE_DIR
from auto_eval.tasks_meta import get_executor_task_groups


# --- Configuration ---
# eval_script.sh must run in this directory
# We assume this script (eval_script.py) and eval_script.sh are in the same directory
SCRIPT_DIR = Path(__file__).parent.resolve()
if os.environ.get('PARALLEL_MODE', 'false') == 'true':
    SHELL_SCRIPT_PATH = SCRIPT_DIR / "eval_parallel_wrapper.sh"
else:
    SHELL_SCRIPT_PATH = SCRIPT_DIR / "eval_script.sh"
# ----------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python3 eval_script.py <path_to_task_json>")
        sys.exit(1)

    task_json_path = Path(sys.argv[1])

    # 1. Read and parse task JSON
    try:
        with open(task_json_path, 'r') as f:
            task_config = json.load(f)

        model_path = task_config['model_path']
        output_suffix = task_config['output_suffix']
        version = task_config.get('version', 'v3.1')
        enable_thinking = task_config.get('enable_thinking', False)
        rollout_mode = task_config.get('rollout_mode', "Race")
        overwrite = task_config.get('overwrite', False)
        seed = task_config.get('seed', 42)
        tasks = task_config.get('tasks', [])
        compute_cot_metrics = task_config.get('compute_cot_metrics', False)
        task_id = task_config['task_id']

        # Build extra params from task_config (tensor_parallel_size etc.)
        extparams_parts = []
        tp_size = task_config.get('tensor_parallel_size')
        if tp_size is not None:
            extparams_parts.append(f"--tensor_parallel_size {tp_size}")
        extparams = ' '.join(extparams_parts)

        # Convert tasks list to comma-separated string
        tasks_str = ','.join(tasks) if tasks else 'all'

    except Exception as e:
        logging.error(f"Failed to read or parse JSON task {task_json_path}: {e}")
        # If JSON cannot be read, retrying is useless, exit directly
        sys.exit(1)

    logging.info(f"--- [Task {task_id} started] ---")
    logging.info(f"Version: {version}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Output suffix: {output_suffix}")
    logging.info(f"Enable thinking: {enable_thinking}")
    logging.info(f"Rollout mode: {rollout_mode}")
    logging.info(f"Overwrite: {overwrite}")
    logging.info(f"Seed: {seed}")
    logging.info(f"Tasks: {tasks_str}")
    logging.info(f"Compute CoT metrics: {compute_cot_metrics}")
    logging.info(f"Extparams: {extparams}")
    logging.info(f"Parallel mode: {os.environ.get('PARALLEL_MODE', 'false')}")
    logging.info(f"CWD (working directory): {SCRIPT_DIR}")
    logging.info(f"Executing shell script: {SHELL_SCRIPT_PATH}")
    logging.info(f"DATA_RESULTS_BASE_DIR: {DATA_RESULTS_BASE_DIR}")
    logging.info(f"DATA_DIR: {DATA_DIR}")

    # Construct rollout arguments
    rollout_args = ""
    if rollout_mode in ["Think 64", "ZJH 64"]:  # Support both new and old names for backward compatibility
        rollout_args = "--num_beams 0 --top_k -1 --temperature 1.1 --top_p 1.0 --select_k top_k_by_logprobs --num_return_sequences 64 --num_return_thinking_sequences 64"
    elif rollout_mode in ["Think 1 Then Beam 32", "ZJH 32"]:  # Support both new and old names
        rollout_args = "--num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1"
    elif rollout_mode in ["Think 1 Then Beam 1", "ZJH 1"]:  # Support both new and old names
        rollout_args = "--num_beams 1 --num_return_sequences 1 --num_return_thinking_sequences 1"
    elif rollout_mode == "Think 32 Then Beam 1":
        rollout_args = "--num_beams 1 --num_return_sequences 32 --num_return_thinking_sequences 32"
    elif rollout_mode == "Think 64 Then Beam 1":
        rollout_args = "--num_beams 1 --num_return_sequences 64 --num_return_thinking_sequences 64 --select_k top_k_by_logprobs"

    elif rollout_mode == "Think 1 Then Beam 32 (temp=1, top_k=-1, top_p=1)":
        rollout_args = "--num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1 --temperature 1.0 --top_k -1 --top_p 1.0"
    elif rollout_mode == "Think 1 Then Beam 1 (temp=1, top_k=-1, top_p=1)":
        rollout_args = "--num_beams 1 --num_return_sequences 1 --num_return_thinking_sequences 1 --temperature 1.0 --top_k -1 --top_p 1.0"
    elif rollout_mode == "Think 32 Then Beam 1 (temp=1, top_k=-1, top_p=1)":
        rollout_args = "--num_beams 1 --num_return_sequences 32 --num_return_thinking_sequences 32 --temperature 1.0 --top_k -1 --top_p 1.0"

    elif rollout_mode == "Final":
        rollout_args = "--num_beams 32 --num_return_sequences 32 --num_return_thinking_sequences 1"
    elif rollout_mode == "Pretrain No Chat":
        rollout_args = "--custom_chat_template qwen3_pretrain_no_chat.jinja2"
    elif rollout_mode == "No System":
        rollout_args = "--custom_chat_template qwen3_no_system.jinja2"
    elif rollout_mode == "No System No Think":
        rollout_args = "--custom_chat_template qwen3_no_system_no_think.jinja2"
    elif rollout_mode == "No System No Think One N":
        rollout_args = "--custom_chat_template qwen3_no_system_no_think_one_n.jinja2"
    elif rollout_mode == "Think 1 Then Beam 256":
        rollout_args = "--num_beams 256 --num_return_sequences 256 --num_return_thinking_sequences 1 --max_logprobs 512"
    elif rollout_mode == "Think 8 Then Beam 64":
        rollout_args = "--num_beams 64 --num_return_sequences 512 --num_return_thinking_sequences 8"
    elif rollout_mode == "Think 1 Then Beam 128":
        rollout_args = "--num_beams 128 --num_return_sequences 128 --num_return_thinking_sequences 1"
    elif rollout_mode == "Beam 4":
        rollout_args = "--num_beams 4 --num_return_sequences 4"
    elif rollout_mode == "Beam 32":
        rollout_args = "--num_beams 32 --num_return_sequences 32"
    elif rollout_mode == "Beam 512":
        rollout_args = "--num_beams 512 --num_return_sequences 512 --max_logprobs 1024 --worker_batch_size 25"
    elif rollout_mode == "Greedy Sample":
        rollout_args = "--num_beams 0 --temperature 0.01 --num_return_sequences 1 --sample_size 100000"
    elif rollout_mode == "Race":
        rollout_args = "--race_mode"

    # Append seed argument
    rollout_args += f" --seed {seed}"

    # Append CoT-quality metrics flag (feat/cot-metrics). Requires
    # num_return_thinking_sequences=1 in the chosen rollout_mode.
    if compute_cot_metrics:
        rollout_args += " --compute_cot_metrics true"


    # 2. Execute evaluation
    try:
        # Prepare environment variables with configuration paths
        env = os.environ.copy()
        from auto_eval.config import DATA_DIR
        env['BENCHMARK_TASK_DATA_DIR'] = str(DATA_DIR)
        env['BENCHMARK_DATA_DIR'] = str(DATA_RESULTS_BASE_DIR)
        env['BENCHMARK_LOG_DIR'] = str(QUEUE_LOG_BASE_DIR)
        env['VERSION'] = version
        env['DATA_VERSION'] = version

        # Pass seed via environment variable (used by eval_end2end.sh)
        env['SEED'] = str(seed)

        shell_script = SHELL_SCRIPT_PATH
        # Pass executor task groups from registry to shell scripts (per-category)
        task_groups = get_executor_task_groups(version)
        env['EXECUTOR_CATEGORIES'] = ' '.join(task_groups.keys())
        for cat, tasks in task_groups.items():
            env[f'EXECUTOR_TASKS_{cat}'] = ' '.join(tasks)

        logging.info(f"Rollout args: {rollout_args}")
        logging.info(f"Executor categories: {list(task_groups.keys())}")
        for cat, tasks in task_groups.items():
            logging.info(f"  [{cat}] ({len(tasks)} tasks): {' '.join(tasks)}")

        # Unified subprocess call — both scripts accept the same positional args
        # $1 (model_path), $2 (output_suffix), $3 (enable_thinking), $4 (tasks),
        # $5 (rollout_args), $6 (rollout_mode), $7 (overwrite), $8 (extparams)
        result = subprocess.run(
            [
                "/bin/bash",
                str(shell_script),
                model_path,
                output_suffix,
                "true" if enable_thinking else "false",
                tasks_str,
                rollout_args,
                rollout_mode,
                "true" if overwrite else "false",
                extparams,
            ],
            env=env,
            cwd=SCRIPT_DIR,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        # 3. Success
        logging.info(f"Shell script stdout:\n{result.stdout}")
        logging.info(f"--- [Task {task_id} succeeded] ---")
        # Exit with status code 0 on success
        sys.exit(0)

    except subprocess.CalledProcessError as e:
        # 4. Failure
        logging.error(f"!!! [Task {task_id} execution failed] !!!")
        logging.error(f"Exit code: {e.returncode}")
        logging.error(f"Stdout (if any):\n{e.stdout}")
        logging.error(f"Stderr:\n{e.stderr}")

        # Save error log to task JSON file (as a new field, not appending text)
        try:
            task_config['error_log'] = {
                'exit_code': e.returncode,
                'stdout': e.stdout,
                'stderr': e.stderr
            }
            with open(task_json_path, "w") as f:
                json.dump(task_config, f, indent=4)
            logging.info(f"Error log saved to {task_json_path}")
        except Exception as log_e:
            logging.error(f"Failed to write error log back to {task_json_path}: {log_e}")

        # Critical: exit with non-zero status code
        # This way the sentinel script knows it failed
        sys.exit(1)
    except Exception as e:
        # 5. Other Python errors
        logging.error(f"!!! [Task {task_id} encountered Python exception] !!!: {e}")

        # Save error log to task JSON file
        try:
            task_config['error_log'] = {
                'exception': str(e),
                'exception_type': type(e).__name__
            }
            with open(task_json_path, "w") as f:
                json.dump(task_config, f, indent=4)
            logging.info(f"Error log saved to {task_json_path}")
        except Exception as log_e:
            logging.error(f"Failed to write error log back to {task_json_path}: {log_e}")

        sys.exit(1)

if __name__ == "__main__":
    main()
