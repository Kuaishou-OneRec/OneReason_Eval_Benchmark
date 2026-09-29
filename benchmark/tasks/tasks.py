"""Public task selection: competition data v3.1 only."""
from typing import List, Optional, Any
from benchmark.tasks.v3_1 import registry
LATEST_BENCHMARK_VERSION = "v3.1"
BenchmarkTable = {LATEST_BENCHMARK_VERSION: registry.TaskTable}
MergedTaskTable = dict(registry.TaskTable)

def get_loader(task_name, data_dir, tokenizer=None, enable_thinking=False, custom_chat_template=None, benchmark_version=None):
    check_benchmark_version(benchmark_version)
    return registry.get_loader(task_name, data_dir, tokenizer, enable_thinking, custom_chat_template)

def get_evaluator(task_name, benchmark_version=None):
    check_benchmark_version(benchmark_version)
    return registry.get_evaluator(task_name)

def get_task_config(task_name, benchmark_version=None):
    check_benchmark_version(benchmark_version)
    return registry.get_task_config(task_name)

def get_available_benchmark_versions():
    return list(BenchmarkTable)

def get_available_task_types(benchmark_version=None):
    check_benchmark_version(benchmark_version)
    return sorted(MergedTaskTable)

def get_available_domains(benchmark_version=None):
    check_benchmark_version(benchmark_version)
    return sorted({r.category for r in registry.TASK_REGISTRY.values()})

def get_available_languages(benchmark_version=None):
    check_benchmark_version(benchmark_version)
    return sorted({r.config["language"] for r in registry.TASK_REGISTRY.values() if r.config.get("language")})

def check_benchmark_version(benchmark_version: Optional[str]) -> str:
    """
    Validate if benchmark version is valid

    Args:
        benchmark_version: Version to validate, returns latest version if None

    Returns:
        str: Valid benchmark version

    Raises:
        ValueError: If version is invalid
    """
    if benchmark_version is None:
        benchmark_version = LATEST_BENCHMARK_VERSION
    else:
        available_benchmark_versions = get_available_benchmark_versions()

        if benchmark_version not in available_benchmark_versions:
            raise ValueError(
                f"Invalid benchmark version: {benchmark_version}. Available versions: {', '.join(available_benchmark_versions)}"
            )

    return benchmark_version


def check_task_types(
    task_types: Optional[List[str]],
    benchmark_version: Optional[str] = None,
) -> List[str]:
    """
    Validate if task types are valid

    Args:
        task_types: List of task types to validate, returns all task types if None
        benchmark_version: Benchmark version, None means search all versions (merged)

    Returns:
        List[str]: Valid task types list

    Raises:
        ValueError: If task type is invalid
    """
    available_task_types = get_available_task_types(benchmark_version)
    if task_types is None:
        task_types = available_task_types
    else:
        if isinstance(task_types, str):
            task_types = [task_types]
        task_types = sorted(list(set(task_types)))
        task_types = [task_type.lower() for task_type in task_types]
        for task_type in task_types:
            if task_type not in available_task_types:
                raise ValueError(
                    f"{benchmark_version} | Invalid task type: {task_type}. Available task types: {', '.join(available_task_types)}"
                )
    return task_types


def check_splits(
    splits: Optional[List[str]],
    benchmark_version: str = LATEST_BENCHMARK_VERSION,
) -> List[str]:
    """
    Validate if dataset splits are valid

    Args:
        splits: List of splits to validate, returns all splits if None
        benchmark_version: Benchmark version

    Returns:
        List[str]: Valid splits list

    Raises:
        ValueError: If split is invalid
    """
    # Only allow test split
    available_splits = ["test"]

    if splits is None:
        splits = available_splits
    else:
        if isinstance(splits, str):
            splits = [splits]
        splits = sorted(list(set(splits)))
        splits = [split.lower() for split in splits]

    for split in splits:
        if split not in available_splits:
            raise ValueError(
                f"{benchmark_version} | Invalid split: {split}. Available splits: {', '.join(available_splits)}"
            )
    return splits
