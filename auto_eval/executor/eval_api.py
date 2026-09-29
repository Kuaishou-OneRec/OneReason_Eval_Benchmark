"""
Evaluation script for closed-source (API-based) models.
Runs as a subprocess, called by the CPU sentinel worker.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auto_eval.config import RESULTS_BASE_DIR, LOG_BASE_DIR, DATA_RESULTS_BASE_DIR

logger = logging.getLogger(__name__)


def _setup_logging(output_suffix: str, version: str):
    """Setup logging to both console and file."""
    log_dir = LOG_BASE_DIR / version
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{output_suffix}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
        root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(str(log_file), mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
    root_logger.addHandler(file_handler)

    return log_file


def run_api_evaluation(task_config: dict):
    """Run evaluation using API-based generator."""
    model_type = task_config['model_type']
    output_suffix = task_config['output_suffix']
    version = task_config.get('version', 'v3.1')
    tasks = task_config.get('tasks', [])
    seed = task_config.get('seed', 42)
    overwrite = task_config.get('overwrite', False)
    task_id = task_config.get('task_id', 'unknown')

    # Empty list means run all tasks for this version
    if not tasks:
        tasks = None

    log_file = _setup_logging(output_suffix, version)

    logger.info(f"--- [API Task {task_id} started] ---")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Version: {version}")
    logger.info(f"Output suffix: {output_suffix}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Seed: {seed}")

    import importlib
    importlib.invalidate_caches()
    from benchmark.benchmark import Benchmark
    from benchmark.api_generator import APIGenerator

    # Create API generator (honour api_max_workers in task JSON, default serial)
    api_max_workers = task_config.get('api_max_workers', 1)
    generator = APIGenerator(model_type=model_type, max_workers=api_max_workers)

    # Data directory matches version (same as eval_script.py)
    data_dir = str(DATA_RESULTS_BASE_DIR / "data" / f"data_{version}")

    # Create benchmark instance (no model_path for API models)
    benchmark = Benchmark(
        model_path=None,
        task_types=tasks,
        data_dir=data_dir,
        seed=seed,
        benchmark_version=version,
    )

    # Output directory
    output_dir = str(RESULTS_BASE_DIR / version / f"results_{output_suffix}")

    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Log file: {log_file}")

    # Run generation
    benchmark.run(
        generator=generator,
        output_dir=output_dir,
        overwrite=overwrite,
    )

    # Run evaluation
    logger.info("Running evaluation on generated results...")
    eval_results_path = os.path.join(output_dir, "eval_results.json")
    Benchmark.evaluate_dev(
        generation_results_dir=output_dir,
        output_path=eval_results_path,
        data_dir=data_dir,
        overwrite=overwrite,
        benchmark_version=version,
    )

    logger.info(f"--- [API Task {task_id} succeeded] ---")
    logger.info(f"Results: {eval_results_path}")


def main():
    """Entry point when run as standalone script."""
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

    if len(sys.argv) != 2:
        logger.error("Usage: python3 eval_api.py <path_to_task_json>")
        sys.exit(1)

    task_json_path = Path(sys.argv[1])

    try:
        with open(task_json_path, 'r') as f:
            task_config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read task JSON {task_json_path}: {e}")
        sys.exit(1)

    try:
        run_api_evaluation(task_config)
        sys.exit(0)
    except Exception as e:
        logger.error(f"API evaluation failed: {e}")
        import traceback
        traceback.print_exc()

        try:
            task_config['error_log'] = {
                'exception': str(e),
                'exception_type': type(e).__name__
            }
            with open(task_json_path, 'w') as f:
                json.dump(task_config, f, indent=4)
        except Exception:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()
