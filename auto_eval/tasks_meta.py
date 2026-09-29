import importlib
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

_TASKS_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "tasks"


def _discover_versions() -> List[str]:
    """Scan benchmark/tasks/ for v*_* dirs that contain a registry module."""
    versions: List[str] = []
    if not _TASKS_DIR.is_dir():
        return versions
    for d in sorted(_TASKS_DIR.iterdir()):
        if d.is_dir() and re.match(r"^v\d+_\d+$", d.name) and (d / "registry.py").exists():
            version_str = d.name.replace("_", ".", 1)  # v2_0 -> v2.0
            versions.append(version_str)
    return versions


def _load_registry(version: str):
    module_name = f"benchmark.tasks.{version.replace('.', '_')}.registry"
    return importlib.import_module(module_name)


def get_all_versions() -> List[str]:
    return _discover_versions()


def get_tasks_metadata(version: str) -> Dict[str, List[str]]:
    reg = _load_registry(version)
    task_registry = getattr(reg, "TASK_REGISTRY", {})

    all_tasks: List[str] = list(task_registry.keys())
    visible_tasks: List[str] = []
    by_category: Dict[str, List[str]] = {}

    for name, tr in task_registry.items():
        cat = getattr(tr, "category", "")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(name)

        if getattr(tr, "show_in_web_ui", False):
            visible_tasks.append(name)

    default_selected: List[str] = visible_tasks.copy()

    return {
        "all_tasks": all_tasks,
        "visible_tasks": visible_tasks,
        "default_selected": default_selected,
        "by_category": [by_category[k] for k in by_category],  # kept as list-of-lists if needed
    }


def get_executor_task_groups(version: Optional[str] = None) -> Dict[str, List[str]]:
    """Group only the registered competition tasks."""
    from benchmark.tasks.tasks import check_benchmark_version
    reg = _load_registry(check_benchmark_version(version))
    groups: Dict[str, List[str]] = {}
    for name, task in reg.TASK_REGISTRY.items():
        groups.setdefault(task.category, []).append(name)
    return groups


def get_task_plot_metadata(version: str) -> Dict[str, Any]:
    """Return all analysis/plot metadata for tasks in a given version registry."""
    reg = _load_registry(version)
    task_registry = getattr(reg, "TASK_REGISTRY", {})

    task_metrics: Dict[str, List[str]] = {}
    radar_task_metrics: Dict[str, str] = {}
    task_ranges: Dict[str, Dict[str, float]] = {}
    affected_tasks: List[str] = []

    for name, tr in task_registry.items():
        if not getattr(tr, "metrics", None):
            # Still track affected_by_sid_pid for tasks without plot metrics
            if getattr(tr, "affected_by_sid_pid", False):
                affected_tasks.append(name)
            continue
        task_metrics[name] = tr.metrics
        if tr.radar_metric:
            radar_task_metrics[name] = tr.radar_metric
        task_ranges[name] = {"min": tr.radar_range_min, "max": tr.radar_range_max}
        if tr.affected_by_sid_pid:
            affected_tasks.append(name)

    return {
        "task_metrics": task_metrics,
        "radar_task_metrics": radar_task_metrics,
        "task_ranges": task_ranges,
        "affected_tasks": affected_tasks,
    }
