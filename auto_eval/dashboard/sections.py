"""Section specifications for the dashboard.

Sections are dynamically built from the task registry and evaluation configs
so that adding a new task or metric only requires updating the registry.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


DEFAULT_BINDING = "__default__"
GLOBAL_SCOPE = "全域"
SINGLE_SCOPE = "单域"
EPSILON = 1e-12


@dataclass(frozen=True)
class MetricBinding:
    task: str
    metric: str


@dataclass(frozen=True)
class RowSpec:
    labels: tuple[str, ...]
    bindings: dict[str, MetricBinding]


@dataclass(frozen=True)
class SectionSpec:
    key: str
    title: str
    description: str
    left_headers: tuple[str, ...]
    scope_headers: tuple[str, ...]
    rows: tuple[RowSpec, ...]


def metric_filter_key(section_key: str, metric_name: str) -> str:
    return f"{section_key}::{metric_name}"


def task_filter_key(section_key: str, task_name: str) -> str:
    return f"{section_key}::task::{task_name}"


def single_row(labels: tuple[str, ...], task: str, metric: str) -> RowSpec:
    return RowSpec(labels=labels, bindings={DEFAULT_BINDING: MetricBinding(task=task, metric=metric)})


def scoped_row(
    labels: tuple[str, ...],
    global_task: str,
    single_task: str,
    metric: str,
) -> RowSpec:
    return RowSpec(
        labels=labels,
        bindings={
            GLOBAL_SCOPE: MetricBinding(task=global_task, metric=metric),
            SINGLE_SCOPE: MetricBinding(task=single_task, metric=metric),
        },
    )


# ---------------------------------------------------------------------------
# Category → section metadata (lightweight, no task/metric data)
# ---------------------------------------------------------------------------

CATEGORY_SECTION_META: OrderedDict[str, dict[str, Any]] = OrderedDict([
    ("general", {
        "key": "general",
        "title": "推理 Benchmark",
        "description": "通用推理评测主指标。",
        "left_headers": ("Task", "Metric"),
        "scope_headers": (),
    }),
    ("perception", {
        "key": "perception",
        "title": "L0 基础能力",
        "description": "Caption2SID、SID2Caption、QA 等感知任务。",
        "left_headers": ("Task", "Metric"),
        "scope_headers": (),
    }),
    ("derivation", {
        "key": "derivation",
        "title": "L1 i2i 能力",
        "description": "看后搜、tagnex 等推导任务。",
        "left_headers": ("Task", "Metric"),
        "scope_headers": (),
    }),
    ("evolution", {
        "key": "evolution",
        "title": "L2 多跳能力",
        "description": "兴趣演化多跳任务。",
        "left_headers": ("Task", "Metric"),
        "scope_headers": (),
    }),
    ("recommendation", {
        "key": "rec",
        "title": "Rec 推荐能力",
        "description": "推荐任务，按 think/nothink 再拆单域/全域。",
        "left_headers": ("Task", "Metric"),
        "scope_headers": (GLOBAL_SCOPE, SINGLE_SCOPE),
    }),
])

DOMAIN_LABELS = {
    "video": "短视频",
    "product": "电商",
    "ad": "广告",
    "live": "直播",
}

# Pattern: optional "cross_" prefix, domain name, version suffix
_REC_TASK_RE = re.compile(r"^(cross_)?(.+?)_(v\d+\..+)$")

# Metrics that have pid_ variants when evaluation_mode == "both"
_PID_ELIGIBLE = {"pass", "recall", "position1_pass"}


# ---------------------------------------------------------------------------
# Metric expansion
# ---------------------------------------------------------------------------

def expand_metrics(config: dict) -> list[str]:
    """Expand parameterized metrics from a task's evaluation_config.

    e.g. ``["pass@k", "recall@k"]`` with ``k_values: [1, 4, 8]``
    → ``["pass@1", "pass@4", "pass@8", "recall@1", "recall@4", "recall@8"]``

    If ``evaluation_mode == "both"``, also generates ``pid_`` prefixed variants
    for eligible metric families.
    """
    eval_cfg = config.get("evaluation_config", {})
    raw_metrics: list[str] = eval_cfg.get("metrics", [])
    k_values: list[int] = eval_cfg.get("k_values", [])
    eval_mode: str = eval_cfg.get("evaluation_mode", "")

    result: list[str] = []
    for metric in raw_metrics:
        if "@k" in metric and k_values:
            base = metric.replace("@k", "")
            for k in k_values:
                result.append(f"{base}@{k}")
            if eval_mode == "both" and base in _PID_ELIGIBLE:
                for k in k_values:
                    result.append(f"pid_{base}@{k}")
        else:
            result.append(metric)
    return result


# ---------------------------------------------------------------------------
# Dynamic section builder
# ---------------------------------------------------------------------------

def build_section_specs(version: str | None = None) -> tuple[SectionSpec, ...]:
    """Build SECTION_SPECS dynamically from the task registry.

    If ``version`` is None, merge TASK_REGISTRY across all discovered versions
    (later/newer versions override earlier ones on name collisions).
    """
    from auto_eval.tasks_meta import _load_registry, get_all_versions

    task_registry: dict = {}
    if version is None:
        for v in get_all_versions():
            try:
                reg = _load_registry(v)
            except Exception:
                continue
            task_registry.update(getattr(reg, "TASK_REGISTRY", {}))
    else:
        reg = _load_registry(version)
        task_registry = dict(getattr(reg, "TASK_REGISTRY", {}))

    # Group tasks by category, only visible ones
    by_category: dict[str, list] = {}
    for name, tr in task_registry.items():
        if not getattr(tr, "show_in_web_ui", False):
            continue
        cat = getattr(tr, "category", "")
        by_category.setdefault(cat, []).append(tr)

    sections: list[SectionSpec] = []

    for category, meta in CATEGORY_SECTION_META.items():
        tasks = by_category.get(category, [])
        if not tasks:
            continue

        if category == "recommendation":
            rows = _build_rec_rows(tasks)
        else:
            rows = _build_simple_rows(tasks)

        if not rows:
            continue

        sections.append(SectionSpec(
            key=meta["key"],
            title=meta["title"],
            description=meta["description"],
            left_headers=meta["left_headers"],
            scope_headers=meta["scope_headers"],
            rows=tuple(rows),
        ))

    return tuple(sections)


def _build_simple_rows(tasks: list) -> list[RowSpec]:
    """Build rows for non-recommendation categories (one row per task×metric)."""
    rows: list[RowSpec] = []
    for tr in tasks:
        metrics = expand_metrics(tr.config)
        task_label = tr.name
        for metric in metrics:
            rows.append(single_row((task_label, metric), tr.name, metric))
    return rows


def _build_rec_rows(tasks: list) -> list[RowSpec]:
    """Build scoped rows for recommendation, pairing cross/single domain tasks."""
    # Index tasks: single tasks and cross tasks
    single_tasks: dict[str, Any] = {}  # (domain, version) → TaskRegistration
    cross_tasks: dict[str, Any] = {}   # (domain, version) → TaskRegistration
    unpaired: list = []

    for tr in tasks:
        m = _REC_TASK_RE.match(tr.name)
        if not m:
            unpaired.append(tr)
            continue
        is_cross = bool(m.group(1))
        domain = m.group(2)
        version = m.group(3)
        key = (domain, version)
        if is_cross:
            cross_tasks[key] = tr
        else:
            single_tasks[key] = tr

    rows: list[RowSpec] = []

    # Build scoped rows for paired tasks
    all_keys = sorted(set(single_tasks.keys()) | set(cross_tasks.keys()))
    for domain, version in all_keys:
        cross_tr = cross_tasks.get((domain, version))
        single_tr = single_tasks.get((domain, version))
        # Use whichever is available for metric expansion
        ref_tr = cross_tr or single_tr
        if ref_tr is None:
            continue
        metrics = expand_metrics(ref_tr.config)
        domain_label = DOMAIN_LABELS.get(domain, domain)
        task_label = f"{domain_label} {version}"

        if cross_tr and single_tr:
            for metric in metrics:
                rows.append(scoped_row(
                    (task_label, metric),
                    cross_tr.name,
                    single_tr.name,
                    metric,
                ))
        else:
            # Only one side exists, use single_row
            tr = cross_tr or single_tr
            for metric in metrics:
                rows.append(single_row((task_label, metric), tr.name, metric))

    # Unpaired tasks (e.g. memory variants without matching pattern)
    for tr in unpaired:
        metrics = expand_metrics(tr.config)
        for metric in metrics:
            rows.append(single_row((tr.name, metric), tr.name, metric))

    return rows


# Module-level computed value
SECTION_SPECS: tuple[SectionSpec, ...] = build_section_specs()
