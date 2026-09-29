"""
Generation script - executes vLLM inference for a single task
Similar to eval_script.py but for generation tasks
"""
import os
import sys
import json
import subprocess
import logging
import tempfile
from pathlib import Path
from datetime import datetime

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Navigate to project root

# Path to vLLM run script
VLLM_RUN_SCRIPT = PROJECT_ROOT / "scripts" / "vllm" / "inference" / "run.sh"
VLLM_GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "vllm" / "inference" / "generate.py"
VLLM_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "vllm" / "inference"
# ----------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')


def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python3 generation_script.py <path_to_task_json>")
        sys.exit(1)

    task_json_path = Path(sys.argv[1])

    # 1. Read and parse task JSON
    try:
        with open(task_json_path, 'r', encoding='utf-8') as f:
            task_config = json.load(f)

        task_id = task_config.get('task_id', task_json_path.stem)
        models = task_config['models']  # List of model paths
        prompts = task_config['prompts']  # List of prompts
        params = task_config.get('params', {})

    except KeyError as e:
        logging.error(f"Missing required field in task JSON: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to read or parse JSON task {task_json_path}: {e}")
        sys.exit(1)

    logging.info(f"--- [Generation Task {task_id} started] ---")
    logging.info(f"Models: {models}")
    logging.info(f"Number of prompts: {len(prompts)}")
    logging.info(f"Params: {params}")

    # 2. Create temporary input JSON for vLLM
    try:
        # Build input JSON for run.sh/generate.py
        vllm_input = {
            "models": models,
            "prompts": prompts,
            "params": params
        }

        # Create temporary files in vLLM script directory
        temp_input_path = VLLM_SCRIPT_DIR / f"temp_input_{task_id}.json"
        temp_output_path = VLLM_SCRIPT_DIR / f"temp_output_{task_id}.json"

        with open(temp_input_path, 'w', encoding='utf-8') as f:
            json.dump(vllm_input, f, ensure_ascii=False, indent=2)

        logging.info(f"Created temporary input file: {temp_input_path}")

    except Exception as e:
        logging.error(f"Failed to create temporary input file: {e}")
        sys.exit(1)

    # 3. Execute vLLM generation via run.sh
    try:
        # Use run.sh with input and output file names (relative to script dir)
        result = subprocess.run(
            [
                "/bin/bash",
                str(VLLM_RUN_SCRIPT),
                f"temp_input_{task_id}.json",
                f"temp_output_{task_id}.json"
            ],
            cwd=VLLM_SCRIPT_DIR,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        logging.info(f"vLLM generation completed successfully")
        logging.info(f"stdout:\n{result.stdout}")

    except subprocess.CalledProcessError as e:
        logging.error(f"!!! [Generation Task {task_id} execution failed] !!!")
        logging.error(f"Exit code: {e.returncode}")
        logging.error(f"Stdout:\n{e.stdout}")
        logging.error(f"Stderr:\n{e.stderr}")

        # Save error log to task JSON
        task_config['status'] = 'failed'
        task_config['error_log'] = {
            'exit_code': e.returncode,
            'stdout': e.stdout,
            'stderr': e.stderr
        }
        task_config['finished_at'] = datetime.now().isoformat()

        with open(task_json_path, "w", encoding='utf-8') as f:
            json.dump(task_config, f, ensure_ascii=False, indent=4)

        # Clean up temp files
        _cleanup_temp_files(temp_input_path, temp_output_path)
        sys.exit(1)

    except Exception as e:
        logging.error(f"!!! [Generation Task {task_id} encountered Python exception] !!!: {e}")

        task_config['status'] = 'failed'
        task_config['error_log'] = {
            'exception': str(e),
            'exception_type': type(e).__name__
        }
        task_config['finished_at'] = datetime.now().isoformat()

        with open(task_json_path, "w", encoding='utf-8') as f:
            json.dump(task_config, f, ensure_ascii=False, indent=4)

        _cleanup_temp_files(temp_input_path, temp_output_path)
        sys.exit(1)

    # 4. Read output and update task JSON with results
    try:
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            generation_output = json.load(f)

        # Update task config with results
        task_config['status'] = 'completed'
        task_config['results'] = generation_output.get('results', {})
        task_config['finished_at'] = datetime.now().isoformat()

        # Write back to task JSON
        with open(task_json_path, "w", encoding='utf-8') as f:
            json.dump(task_config, f, ensure_ascii=False, indent=4)

        logging.info(f"--- [Generation Task {task_id} succeeded] ---")
        logging.info(f"Results written to: {task_json_path}")

    except Exception as e:
        logging.error(f"Failed to read output or update task JSON: {e}")

        task_config['status'] = 'failed'
        task_config['error_log'] = {
            'exception': str(e),
            'exception_type': type(e).__name__,
            'phase': 'output_processing'
        }
        task_config['finished_at'] = datetime.now().isoformat()

        with open(task_json_path, "w", encoding='utf-8') as f:
            json.dump(task_config, f, ensure_ascii=False, indent=4)

        _cleanup_temp_files(temp_input_path, temp_output_path)
        sys.exit(1)

    # 5. Clean up temporary files
    _cleanup_temp_files(temp_input_path, temp_output_path)
    sys.exit(0)


def _cleanup_temp_files(*paths):
    """Clean up temporary files"""
    for path in paths:
        try:
            if path and Path(path).exists():
                Path(path).unlink()
                logging.info(f"Cleaned up temp file: {path}")
        except Exception as e:
            logging.warning(f"Failed to clean up temp file {path}: {e}")


if __name__ == "__main__":
    main()
