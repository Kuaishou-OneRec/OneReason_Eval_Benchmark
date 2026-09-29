"""
Leaderboard data layer and HTML generator.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from analyze.handlers import get_result_directories, load_eval_results
from tasks_meta import get_task_plot_metadata, get_all_versions


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

_CATEGORY_CHECKS = [
    ("general",    lambda t: any(k in t for k in ("mmlu", "gpqa", "math", "gsm"))),
    ("perception", lambda t: any(k in t for k in (
        "sid_qa", "sid2caption", "caption2sid", "itemic_pattern_caption",
    ))),
    ("derivation", lambda t: any(k in t for k in ("view_then_search", "tagnex"))),
    ("evolution",  lambda t: "evolution" in t),
    ("rec",        lambda t: any(k in t for k in ("video_v2", "product_v2", "ad_v2", "live_v2", "cross_"))),
]

_CATEGORY_LABELS = {
    "general":    "推理",
    "perception": "L0 基础",
    "derivation": "L1 i2i",
    "evolution":  "L2 多跳",
    "rec":        "Rec 推荐",
    "other":      "其他",
}

_TAB_ORDER = ["general", "perception", "derivation", "evolution", "rec"]

_DIR_PATTERN = re.compile(
    r'^results_(.+?)_step(\d+)_(think|nonthink|nothink)(.*)$'
)
_VERSION_TOKEN = re.compile(r'^v\d+(\.\d+)*')


def _categorize(task_name: str) -> str:
    for cat, check in _CATEGORY_CHECKS:
        if check(task_name):
            return cat
    return "other"


def _parse_model_meta(dir_name: str) -> Tuple[str, str]:
    """Parse a results directory name and return (version, think).

    Pattern: results_{version_token}_step{N}_{think|nonthink|nothink}{suffix}

    version: leading v\\d+(\\.\\d+)* token from group 1; fallback to first
             underscore-split token if no version token found.
    think:   "think" if group 3 is "think", else "nonthink".

    Returns ("?", "?") if the directory name doesn't match the pattern.
    """
    match = _DIR_PATTERN.match(dir_name)
    if not match:
        return ("?", "?")

    raw_version = match.group(1)
    think_token = match.group(3)

    vm = _VERSION_TOKEN.match(raw_version)
    if vm:
        version = vm.group(0)
    else:
        parts = raw_version.split("_")
        version = parts[0] if parts else raw_version

    think = "think" if think_token == "think" else "nonthink"
    return (version, think)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def build_leaderboard_data(top_n: int = 10, version_filter: str = "全部", think_filter: str = "全部") -> Dict[str, List[Dict]]:
    """
    Returns {task_name: [{"rank": int, "model": str, "metric": str, "score": float,
                           "version": str, "think": str}, ...]}
    - All versions mixed (v2.0-v4.0).
    - Sorted descending by score, top_n entries.
    - metric: the radar_metric for that task.
    - model: stripped dir name (without "results_" prefix).
    - Skips tasks with no data or unknown metrics.
    """
    radar_metrics: Dict[str, str] = {}
    for version in get_all_versions():
        meta = get_task_plot_metadata(version)
        radar_metrics.update(meta.get("radar_task_metrics", {}))

    dir_names = get_result_directories()
    if not dir_names:
        return {}

    all_results = load_eval_results(dir_names)

    task_scores: Dict[str, Dict[str, float]] = {}

    for dir_name, raw_json in all_results.items():
        for model_key, model_data in raw_json.items():
            if model_key.startswith("_"):
                continue
            if not isinstance(model_data, dict):
                continue
            for task_name, task_data in model_data.items():
                if task_name.startswith("_"):
                    continue
                metric_name = radar_metrics.get(task_name)
                if not metric_name:
                    continue
                try:
                    score = task_data["test"][metric_name]
                    if not isinstance(score, (int, float)):
                        continue
                except (KeyError, TypeError):
                    continue
                task_scores.setdefault(task_name, {})[dir_name] = float(score)

    leaderboard: Dict[str, List[Dict]] = {}
    for task_name, dir_scores in task_scores.items():
        metric_name = radar_metrics[task_name]
        filtered_scores = {
            d: s for d, s in dir_scores.items()
            if (version_filter == "全部" or _parse_model_meta(d)[0] == version_filter)
            and (think_filter == "全部" or _parse_model_meta(d)[1] == think_filter)
        }
        if not filtered_scores:
            continue
        sorted_entries = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        ranked = []
        for i, (dir_name, score) in enumerate(sorted_entries):
            model = dir_name[len("results_"):] if dir_name.startswith("results_") else dir_name
            version, think = _parse_model_meta(dir_name)
            ranked.append({
                "rank": i + 1,
                "model": model,
                "metric": metric_name,
                "score": score,
                "version": version,
                "think": think,
            })
        leaderboard[task_name] = ranked

    return leaderboard


def get_available_versions() -> List[str]:
    """Return sorted list of unique version values found across all leaderboard entries.

    Excludes "?" unless it is the only value present.
    """
    leaderboard = build_leaderboard_data(top_n=10)
    versions = set()
    for entries in leaderboard.values():
        for entry in entries:
            versions.add(entry["version"])

    if len(versions) > 1:
        versions.discard("?")

    return sorted(versions)


def get_all_task_names(version_filter: str = "全部") -> List[str]:
    """Return sorted list of task names.

    If version_filter is "全部", returns all tasks found in leaderboard data.
    Otherwise returns tasks registered in that version's registry (with radar_metric).
    """
    if version_filter != "全部":
        try:
            meta = get_task_plot_metadata(version_filter)
            return sorted(meta.get("radar_task_metrics", {}).keys())
        except Exception:
            pass
    leaderboard = build_leaderboard_data(top_n=10)
    return sorted(leaderboard.keys())


# ---------------------------------------------------------------------------
# HTML generator
# ---------------------------------------------------------------------------

_RANK_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}
_RANK_BG = {
    1: "rgba(251,191,36,0.15)",
    2: "rgba(148,163,184,0.15)",
    3: "rgba(180,83,9,0.10)",
}

_BASE_STYLE = """
  .lb-root {
    font-family: var(--font), -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    background: #ffffff;
    color: #333;
    padding: 16px;
    border-radius: 12px;
  }
  .lb-radio { display: none; }
  .lb-tabs {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 20px;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 12px;
  }
  .lb-tab-btn {
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    color: #4b5563;
    padding: 6px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.15s;
  }
  .lb-tab-btn:hover {
    background: #e3f2fd;
    color: #1976d2;
    border-color: rgba(33,150,243,0.3);
  }
  .lb-panel { display: none; }
  .lb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
  }
  .lb-card {
    background: #f9f9f9;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px;
  }
  .lb-card-title {
    font-size: 13px;
    color: #555;
    margin: 0 0 10px 0;
    word-break: break-all;
  }
  .lb-card-title span { color: #1976d2; }
  .lb-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    table-layout: fixed;
  }
  .lb-table tr { min-height: 44px; height: auto; }
  .lb-table col.col-rank    { width: 32px; }
  .lb-table col.col-model   { width: auto; }
  .lb-table col.col-score   { width: 72px; }
  .lb-table col.col-version { width: 52px; }
  .lb-table col.col-think   { width: 64px; }
  .lb-table th {
    color: #666;
    text-align: left;
    padding: 4px 6px;
    border-bottom: 1px solid #e5e7eb;
    font-weight: 500;
  }
  .lb-table td {
    padding: 5px 6px;
    border-bottom: 1px solid #f3f4f6;
    color: #333;
    vertical-align: top;
  }
  .lb-table td.model {
    white-space: normal;
    line-height: 1.35;
    word-break: break-all;
  }
  .lb-table td.score {
    font-family: ui-monospace, monospace;
    color: #1976d2;
    white-space: nowrap;
  }
  .lb-badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
  }
  .lb-badge.think    { background: #e8f5e9; color: #2e7d32; }
  .lb-badge.nonthink { background: #f3f4f6; color: #6b7280; }
  .lb-empty {
    color: #999;
    font-size: 13px;
    text-align: center;
    padding: 20px 0;
  }
"""


def _build_style(active_cats: List[str]) -> str:
    """Generate <style> block with base styles + CSS-only tab switching rules."""
    rules = []
    for cat in active_cats:
        rules.append(
            f'#lbt-{cat}:checked ~ .lb-tabs label[for="lbt-{cat}"] {{'
            f'background:#1976d2;border-color:#1976d2;color:#fff;}}'
        )
        rules.append(
            f'#lbt-{cat}:checked ~ .lb-panels .lb-panel-{cat} {{display:block;}}'
        )
    return "<style>" + _BASE_STYLE + "\n  ".join(rules) + "</style>"


def _render_task_card(
    task_name: str,
    entries: List[Dict],
    model_tasks: Dict[str, List[Dict]] = None,
) -> str:
    if not entries:
        metric_label = ""
    else:
        metric_label = entries[0]["metric"]

    title_html = (
        f'<p class="lb-card-title">{task_name}'
        + (f' &middot; <span>{metric_label}</span>' if metric_label else "")
        + "</p>"
    )

    if not entries:
        return (
            f'<div class="lb-card">{title_html}'
            '<p class="lb-empty">暂无数据</p></div>'
        )

    rows_html = []
    for entry in entries:
        rank = entry["rank"]
        medal = _RANK_MEDALS.get(rank, str(rank))
        bg = _RANK_BG.get(rank, "transparent")
        score_str = f'{entry["score"]:.4f}'
        model_escaped = entry["model"].replace("<", "&lt;").replace(">", "&gt;")
        think_cls = "think" if entry["think"] == "think" else "nonthink"

        if model_tasks and entry["model"] in model_tasks:
            detail_rows = "".join(
                f'<tr>'
                f'<td style="padding:2px 4px;color:#555;font-size:11px;word-break:break-all">{r["task"]}</td>'
                f'<td style="padding:2px 4px;color:#333;font-size:11px">#{r["rank"]}</td>'
                f'<td style="padding:2px 4px;color:#1976d2;font-size:11px;font-family:monospace">{r["score"]:.4f}</td>'
                f'</tr>'
                for r in model_tasks[entry["model"]]
            )
            detail_table = (
                f'<table style="width:100%;border-collapse:collapse;margin-top:4px">'
                f'<thead><tr>'
                f'<th style="text-align:left;color:#999;font-size:11px;padding:2px 4px;border-bottom:1px solid #e5e7eb">任务</th>'
                f'<th style="color:#999;font-size:11px;padding:2px 4px;border-bottom:1px solid #e5e7eb">排名</th>'
                f'<th style="color:#999;font-size:11px;padding:2px 4px;border-bottom:1px solid #e5e7eb">分数</th>'
                f'</tr></thead><tbody>{detail_rows}</tbody></table>'
            )
            model_cell = (
                f'<td class="model">'
                f'<details style="cursor:pointer">'
                f'<summary style="list-style:none;cursor:pointer;outline:none" title="{model_escaped}">{model_escaped}</summary>'
                f'{detail_table}'
                f'</details>'
                f'</td>'
            )
        else:
            model_cell = f'<td class="model" title="{model_escaped}">{model_escaped}</td>'

        rows_html.append(
            f'<tr style="background:{bg}">'
            f'<td>{medal}</td>'
            + model_cell
            + f'<td class="score">{score_str}</td>'
            f'<td>{entry["version"]}</td>'
            f'<td><span class="lb-badge {think_cls}">{entry["think"]}</span></td>'
            "</tr>"
        )

    table_html = (
        '<table class="lb-table">'
        '<colgroup>'
        '<col class="col-rank">'
        '<col class="col-model">'
        '<col class="col-score">'
        '<col class="col-version">'
        '<col class="col-think">'
        '</colgroup>'
        "<thead><tr><th>#</th><th>模型</th><th>分数</th><th>版本</th><th>思考</th></tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody>"
        "</table>"
    )

    return f'<div class="lb-card">{title_html}{table_html}</div>'


def generate_leaderboard_html(
    version_filter: str = "全部",
    think_filter: str = "全部",
    top_n: int = 10,
    task_filter: List[str] = None,
) -> Tuple[str, str]:
    """
    Returns (html_string, status_string).
    Tab switching uses CSS-only checkbox inputs (no JS required).
    All category tabs are shown by default (all checked).
    task_filter: if non-empty list, only show tasks in that list.
    """
    leaderboard = build_leaderboard_data(top_n=top_n, version_filter=version_filter, think_filter=think_filter)

    # Apply task filter first (before building model_tasks)
    if task_filter:
        leaderboard = {k: v for k, v in leaderboard.items() if k in task_filter}

    # Build model->tasks map from full (unfiltered by version/think) leaderboard
    model_tasks: Dict[str, List[Dict]] = {}
    for task_name, entries in leaderboard.items():
        for e in entries:
            model_tasks.setdefault(e["model"], []).append({
                "task": task_name,
                "rank": e["rank"],
                "score": e["score"],
            })
    for tasks in model_tasks.values():
        tasks.sort(key=lambda x: x["rank"])

    dir_count = len(get_result_directories())
    task_count = len(leaderboard)
    status = f"已加载 {dir_count} 个目录，覆盖 {task_count} 个任务"

    def _empty(style):
        return (
            style + '<div class="lb-root"><p class="lb-empty">暂无数据</p></div>',
            status,
        )

    if not leaderboard:
        return _empty(_build_style([]))

    by_cat: Dict[str, Dict[str, List[Dict]]] = {cat: {} for cat in _TAB_ORDER}
    by_cat["other"] = {}
    for task_name, entries in leaderboard.items():
        cat = _categorize(task_name)
        by_cat.setdefault(cat, {})[task_name] = entries

    active_cats = [c for c in _TAB_ORDER if by_cat.get(c)]
    if by_cat.get("other"):
        active_cats.append("other")

    if not active_cats:
        return _empty(_build_style([]))

    style_block = _build_style(active_cats)

    # Checkboxes, all checked by default → multi-select categories
    checkbox_inputs = []
    for cat in active_cats:
        checkbox_inputs.append(
            f'<input type="checkbox" id="lbt-{cat}" class="lb-radio" checked>'
        )

    tab_labels = []
    for cat in active_cats:
        label = _CATEGORY_LABELS.get(cat, cat)
        tab_labels.append(f'<label for="lbt-{cat}" class="lb-tab-btn">{label}</label>')

    panels = []
    for cat in active_cats:
        cards_html = "".join(
            _render_task_card(task_name, entries, model_tasks)
            for task_name, entries in sorted(by_cat[cat].items())
        )
        panels.append(
            f'<div class="lb-panel lb-panel-{cat}">'
            f'<div class="lb-grid">{cards_html}</div>'
            f'</div>'
        )

    html = (
        style_block
        + '<div class="lb-root">'
        + "".join(checkbox_inputs)
        + '<div class="lb-tabs">' + "".join(tab_labels) + "</div>"
        + '<div class="lb-panels">' + "".join(panels) + "</div>"
        + "</div>"
    )
    return html, status
