"""Data classes and processing logic for the dashboard, extracted from analyze.py."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .sections import (
    DEFAULT_BINDING,
    EPSILON,
    GLOBAL_SCOPE,
    SINGLE_SCOPE,
    MetricBinding,
    RowSpec,
    SectionSpec,
    metric_filter_key,
    task_filter_key,
)


# Competition exposes per-task metrics; aggregate weights are not configured.
RACE_SCORE_COMPONENTS = {}


@dataclass
class ModelSpec:
    key: str
    source_dir: str
    actual_source_dir: str
    archive_dir: str
    version: str
    step: str
    epoch: str
    think_enabled: bool
    kind: str
    group_key: str
    group_title: str
    group_meta: str
    result_dir: Path
    converted: dict


@dataclass(frozen=True)
class LeafColumn:
    key: str
    model_key: str
    group_key: str
    group_title: str
    group_meta: str
    kind: str
    scope: str | None


@dataclass
class RowData:
    spec: RowSpec
    values: dict[str, float | None]
    delta_by_leaf: dict[str, float]
    relative_delta_by_leaf: dict[str, float | None]
    max_value: float | None
    min_value: float | None
    numeric_count: int


@dataclass
class HighlightItem:
    section_key: str
    section_title: str
    row_label: str
    metric_key: str
    group_key: str
    group_title: str
    group_meta: str
    scope: str | None
    delta: float
    think_value: float
    nothink_value: float
    significant: bool


@dataclass
class SectionData:
    spec: SectionSpec
    leaf_columns: list[LeafColumn]
    rows: list[RowData]
    gap_threshold: float
    present_cells: int
    total_cells: int
    significant_up: int
    significant_down: int
    highlights: list[HighlightItem]


@dataclass(frozen=True)
class GroupSpec:
    group_key: str
    version: str
    step: str
    epoch: str
    title: str
    meta: str
    has_think: bool
    has_nothink: bool


@dataclass(frozen=True)
class RaceScoreCard:
    group_key: str
    group_title: str
    group_meta: str
    kind: str
    source_dir: str
    scores: dict[str, float | None]
    contributions: dict[str, list[dict[str, object]]]
    missing: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def clean_text(value: str) -> str:
    return " ".join((value or "").replace("\ufeff", "").strip().split())


def compact_step(step: str) -> str:
    text = clean_text(step)
    if text.isdigit():
        value = int(text)
        if value >= 1000:
            compact = f"{value / 1000:.1f}".rstrip("0").rstrip(".")
            return f"{compact}k step"
        return f"{value} step"
    return text


def compact_epoch(epoch: str) -> str:
    text = clean_text(epoch)
    return text.replace("epoch", "ep").replace("token", "tok")


def build_group_title(version: str) -> str:
    if version == "Pretrain":
        return "Pretrain"
    if version.endswith(" thinkonly"):
        return f'{version.removesuffix(" thinkonly")} T-only'
    if version.endswith(" nothinkonly"):
        return f'{version.removesuffix(" nothinkonly")} N-only'
    if version.endswith(" nothink"):
        return f'{version.removesuffix(" nothink")} N-only'
    if version.endswith(" mix exp"):
        return f'{version.removesuffix(" mix exp")} Mix Exp'
    if version.endswith(" mix"):
        return f'{version.removesuffix(" mix")} Mix'
    return version


def build_group_meta(epoch: str, step: str) -> str:
    return f"{compact_step(step)} · {compact_epoch(epoch)}"


def build_model_key(archive_dir: str, kind: str) -> str:
    return f"{archive_dir}|{kind}"


def format_number(value: float | None) -> str:
    if value is None:
        return "-"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def format_signed(value: float) -> str:
    return f"{value:+.6f}".rstrip("0").rstrip(".")


def format_display_number(value: float | None) -> str:
    if value is None:
        return "-"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def format_display_signed(value: float) -> str:
    return f"{value:+.4f}".rstrip("0").rstrip(".")


def format_display_percent(value: float) -> str:
    text = f"{abs(value) * 100:.1f}".rstrip("0").rstrip(".")
    return text or "0"


def format_delta_badge(delta: float, relative_delta: float | None) -> str:
    arrow = "↑" if delta > EPSILON else "↓" if delta < -EPSILON else "≈"
    if relative_delta is None:
        return f"{arrow}new" if abs(delta) > EPSILON else "≈0%"
    return f"{arrow}{format_display_percent(relative_delta)}%"


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * ratio + 0.999999) - 1))
    return sorted_values[index]


# ---------------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------------

def read_converted(result_dir: Path) -> dict:
    json_path = result_dir / "eval_results.json"
    if not json_path.exists():
        return {}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not payload:
        return {}
    first_key = next(iter(payload))
    converted = payload[first_key]
    return converted if isinstance(converted, dict) else {}


def lookup_metric(model: ModelSpec, binding: MetricBinding) -> float | None:
    task_payload = model.converted.get(binding.task)
    if not isinstance(task_payload, dict):
        return None
    test_payload = task_payload.get("test", {})
    if not isinstance(test_payload, dict):
        return None
    raw_value = test_payload.get(binding.metric)
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    return None


def lookup_metric_value(model: ModelSpec, task_name: str, metric_name: str) -> float | None:
    task_payload = model.converted.get(task_name)
    if not isinstance(task_payload, dict):
        return None
    test_payload = task_payload.get("test", {})
    if not isinstance(test_payload, dict):
        return None
    raw_value = test_payload.get(metric_name)
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    return None


def build_race_score_card_data(models: list[ModelSpec]) -> list[dict]:
    cards: list[dict] = []
    for model in models:
        scores: dict[str, float | None] = {}
        contributions: dict[str, list[dict[str, object]]] = {}
        missing: dict[str, list[str]] = {}

        for race_name, components in RACE_SCORE_COMPONENTS.items():
            total_score = 0.0
            race_contributions: list[dict[str, object]] = []
            missing_dependencies: list[str] = []

            for task_name, metric_name, weight in components:
                metric_value = lookup_metric_value(model, task_name, metric_name)
                contribution_value = None
                if metric_value is None:
                    missing_dependencies.append(f"{task_name}.{metric_name}")
                else:
                    contribution_value = weight * metric_value
                    total_score += contribution_value

                race_contributions.append(
                    {
                        "task": task_name,
                        "metric": metric_name,
                        "weight": weight,
                        "value": contribution_value,
                    }
                )

            contributions[race_name] = race_contributions
            if missing_dependencies:
                scores[race_name] = None
                missing[race_name] = missing_dependencies
            else:
                scores[race_name] = total_score

        cards.append(
            {
                "group_key": model.group_key,
                "group_title": model.group_title,
                "group_meta": model.group_meta,
                "kind": model.kind,
                "source_dir": model.source_dir,
                "scores": scores,
                "contributions": contributions,
                "missing": missing,
            }
        )

    return cards


# ---------------------------------------------------------------------------
# Column / row / group building
# ---------------------------------------------------------------------------

def build_leaf_columns(models: list[ModelSpec], scope_headers: tuple[str, ...]) -> list[LeafColumn]:
    leaves: list[LeafColumn] = []
    scopes = scope_headers if scope_headers else (None,)
    for model in models:
        for scope in scopes:
            scope_key = f"|{scope}" if scope else ""
            leaves.append(
                LeafColumn(
                    key=f"{model.key}{scope_key}",
                    model_key=model.key,
                    group_key=model.group_key,
                    group_title=model.group_title,
                    group_meta=model.group_meta,
                    kind=model.kind,
                    scope=scope,
                )
            )
    return leaves


def build_row_label(row_spec: RowSpec) -> str:
    return " / ".join(label for label in row_spec.labels if label)


def build_task_label(row_spec: RowSpec) -> str:
    labels = row_spec.labels[:-1] if len(row_spec.labels) > 1 else row_spec.labels
    return " / ".join(label for label in labels if label)


def normalize_pair_version(version: str) -> str:
    normalized = clean_text(version).lower()
    for suffix in (" thinkonly", " nothinkonly", " nothink", " think", " nonthink"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized.strip()


def build_group_specs(models: list[ModelSpec]) -> list[GroupSpec]:
    grouped: OrderedDict[str, dict[str, object]] = OrderedDict()
    for model in models:
        state = grouped.setdefault(
            model.group_key,
            {
                "version": model.version,
                "step": model.step,
                "epoch": model.epoch,
                "title": model.group_title,
                "meta": model.group_meta,
                "has_think": False,
                "has_nothink": False,
            },
        )
        if model.kind == "think":
            state["has_think"] = True
        if model.kind == "nothink":
            state["has_nothink"] = True

    specs: list[GroupSpec] = []
    for group_key, info in grouped.items():
        specs.append(
            GroupSpec(
                group_key=group_key,
                version=str(info["version"]),
                step=str(info["step"]),
                epoch=str(info["epoch"]),
                title=str(info["title"]),
                meta=str(info["meta"]),
                has_think=bool(info["has_think"]),
                has_nothink=bool(info["has_nothink"]),
            )
        )
    return specs


def default_pair_group_key(think_group: GroupSpec, nothink_groups: list[GroupSpec]) -> str | None:
    if think_group.has_nothink:
        return think_group.group_key

    normalized_version = normalize_pair_version(think_group.version)
    candidates = [
        group
        for group in nothink_groups
        if normalize_pair_version(group.version) == normalized_version and group.step == think_group.step
    ]
    if not candidates:
        candidates = [
            group for group in nothink_groups if normalize_pair_version(group.version) == normalized_version
        ]
    if not candidates:
        candidates = [group for group in nothink_groups if group.step == think_group.step]
    return candidates[0].group_key if candidates else None


def group_key_to_scope(pair_key: tuple[str, str | None]) -> str | None:
    return pair_key[1]


def compute_rowspans(rows: list[RowData], label_count: int) -> list[list[int]]:
    spans = [[0 for _ in range(label_count)] for _ in rows]
    for column_index in range(label_count):
        row_index = 0
        while row_index < len(rows):
            prefix = rows[row_index].spec.labels[: column_index + 1]
            end_index = row_index + 1
            while end_index < len(rows) and rows[end_index].spec.labels[: column_index + 1] == prefix:
                end_index += 1
            spans[row_index][column_index] = end_index - row_index
            row_index = end_index
    return spans


# ---------------------------------------------------------------------------
# Section data building
# ---------------------------------------------------------------------------

def build_section_data(spec: SectionSpec, models: list[ModelSpec]) -> SectionData:
    model_by_key = {model.key: model for model in models}
    leaf_columns = build_leaf_columns(models, spec.scope_headers)

    pair_lookup: dict[tuple[str, str | None], dict[str, str]] = {}
    for leaf in leaf_columns:
        pair_lookup.setdefault((leaf.group_key, leaf.scope), {})[leaf.kind] = leaf.key

    rows: list[RowData] = []
    total_cells = 0
    present_cells = 0
    raw_highlights: list[HighlightItem] = []
    raw_deltas: list[float] = []

    for row_spec in spec.rows:
        values: dict[str, float | None] = {}
        numeric_values: list[float] = []
        delta_by_leaf: dict[str, float] = {}
        relative_delta_by_leaf: dict[str, float | None] = {}

        for leaf in leaf_columns:
            binding_key = leaf.scope if leaf.scope is not None else DEFAULT_BINDING
            binding = row_spec.bindings.get(binding_key)
            if binding is None:
                value = None
            else:
                value = lookup_metric(model_by_key[leaf.model_key], binding)
            values[leaf.key] = value
            total_cells += 1
            if value is not None:
                present_cells += 1
                numeric_values.append(value)

        for pair_key, pair in pair_lookup.items():
            think_key = pair.get("think")
            nothink_key = pair.get("nothink")
            if not think_key or not nothink_key:
                continue
            think_value = values.get(think_key)
            nothink_value = values.get(nothink_key)
            if think_value is None or nothink_value is None:
                continue
            delta = think_value - nothink_value
            delta_by_leaf[think_key] = delta
            if abs(nothink_value) <= EPSILON:
                relative_delta_by_leaf[think_key] = 0.0 if abs(delta) <= EPSILON else None
            else:
                relative_delta_by_leaf[think_key] = delta / nothink_value
            raw_deltas.append(abs(delta))

            group_leaf = next(leaf for leaf in leaf_columns if leaf.key == think_key)
            raw_highlights.append(
                HighlightItem(
                    section_key=spec.key,
                    section_title=spec.title,
                    row_label=build_row_label(row_spec),
                    metric_key=metric_filter_key(spec.key, row_spec.labels[-1]),
                    group_key=group_leaf.group_key,
                    group_title=group_leaf.group_title,
                    group_meta=group_leaf.group_meta,
                    scope=group_key_to_scope(pair_key),
                    delta=delta,
                    think_value=think_value,
                    nothink_value=nothink_value,
                    significant=False,
                )
            )

        rows.append(
            RowData(
                spec=row_spec,
                values=values,
                delta_by_leaf=delta_by_leaf,
                relative_delta_by_leaf=relative_delta_by_leaf,
                max_value=max(numeric_values) if numeric_values else None,
                min_value=min(numeric_values) if numeric_values else None,
                numeric_count=len(numeric_values),
            )
        )

    gap_threshold = percentile(raw_deltas, 0.8)
    significant_up = 0
    significant_down = 0
    for item in raw_highlights:
        item.significant = gap_threshold > 0 and abs(item.delta) >= gap_threshold
        if item.significant and item.delta > 0:
            significant_up += 1
        if item.significant and item.delta < 0:
            significant_down += 1

    return SectionData(
        spec=spec,
        leaf_columns=leaf_columns,
        rows=rows,
        gap_threshold=gap_threshold,
        present_cells=present_cells,
        total_cells=total_cells,
        significant_up=significant_up,
        significant_down=significant_down,
        highlights=raw_highlights,
    )
