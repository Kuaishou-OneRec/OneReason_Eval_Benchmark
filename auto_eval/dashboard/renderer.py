"""Rendering functions for the dashboard HTML, extracted from analyze.py."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from html import escape

from .data import (
    ModelSpec,
    LeafColumn,
    RowData,
    HighlightItem,
    SectionData,
    GroupSpec,
    build_group_specs,
    build_row_label,
    build_task_label,
    default_pair_group_key,
    format_number,
    format_display_number,
    format_signed,
)
from .sections import (
    EPSILON,
    SectionSpec,
    metric_filter_key,
    task_filter_key,
)
from .template import HTML_TEMPLATE


def render_table_header(spec: SectionSpec, leaf_columns: list[LeafColumn]) -> str:
    header_levels = 3 if spec.scope_headers else 2
    lines = ["<thead>"]

    index_row = ["<tr>"]
    for index, _header in enumerate(spec.left_headers, start=1):
        classes = f"row-head head-sticky sticky-{index} col-index-spacer"
        label = "列号" if index == 1 else ""
        index_row.append(f'<th class="{classes}">{escape(label)}</th>')
    for index, leaf in enumerate(leaf_columns, start=1):
        index_row.append(
            f'<th class="col-index-head" data-group-key="{escape(leaf.group_key)}" data-kind="{escape(leaf.kind)}" data-scope="{escape(leaf.scope or "")}" data-col-index-head="{index}">第{index}列</th>'
        )
    index_row.append("</tr>")
    lines.append("".join(index_row))

    first_row = ["<tr>"]
    for index, header in enumerate(spec.left_headers, start=1):
        classes = f"row-head head-sticky sticky-{index}"
        first_row.append(f'<th class="{classes}" rowspan="{header_levels}">{escape(header)}</th>')
    current_index = 0
    while current_index < len(leaf_columns):
        current_leaf = leaf_columns[current_index]
        span = 1
        while (
            current_index + span < len(leaf_columns)
            and leaf_columns[current_index + span].group_key == current_leaf.group_key
        ):
            span += 1
        first_row.append(
            '<th class="group-head" data-group-key="{group_key}" colspan="{span}"><div class="group-title" title="{title}">{title}</div>'
            '<div class="group-meta">{meta}</div></th>'.format(
                group_key=escape(current_leaf.group_key),
                span=span,
                title=escape(current_leaf.group_title),
                meta=escape(current_leaf.group_meta),
            )
        )
        current_index += span
    first_row.append("</tr>")
    lines.append("".join(first_row))

    second_row = ["<tr>"]
    current_index = 0
    while current_index < len(leaf_columns):
        current_leaf = leaf_columns[current_index]
        span = 1
        while (
            current_index + span < len(leaf_columns)
            and leaf_columns[current_index + span].group_key == current_leaf.group_key
            and leaf_columns[current_index + span].kind == current_leaf.kind
        ):
            span += 1
        second_row.append(
            f'<th class="kind-head {escape(current_leaf.kind)}" data-group-key="{escape(current_leaf.group_key)}" data-kind="{escape(current_leaf.kind)}" colspan="{span}">{escape(current_leaf.kind)}</th>'
        )
        current_index += span
    second_row.append("</tr>")
    lines.append("".join(second_row))

    if spec.scope_headers:
        third_row = ["<tr>"]
        for leaf in leaf_columns:
            third_row.append(
                f'<th class="scope-head" data-group-key="{escape(leaf.group_key)}" data-kind="{escape(leaf.kind)}" data-scope="{escape(leaf.scope or "")}">{escape(leaf.scope or "")}</th>'
            )
        third_row.append("</tr>")
        lines.append("".join(third_row))

    lines.append("</thead>")
    return "".join(lines)


def render_table_colgroup(spec: SectionSpec, leaf_columns: list[LeafColumn]) -> str:
    cols = ["<colgroup>"]
    for index, _ in enumerate(spec.left_headers, start=1):
        cols.append(f'<col class="sticky-col-{index}" />')
    for leaf in leaf_columns:
        cols.append(
            f'<col class="leaf-col" data-group-key="{escape(leaf.group_key)}" data-kind="{escape(leaf.kind)}" data-scope="{escape(leaf.scope or "")}" />'
        )
    cols.append("</colgroup>")
    return "".join(cols)


def value_background(value: float, min_value: float, max_value: float) -> str:
    if abs(max_value - min_value) <= EPSILON:
        alpha = 0.16
    else:
        ratio = (value - min_value) / (max_value - min_value)
        alpha = 0.06 + ratio * 0.30
    return f"rgba(59, 130, 246, {alpha:.4f})"


def render_value_cell(row: RowData, leaf: LeafColumn, threshold: float) -> str:
    value = row.values.get(leaf.key)
    if value is None:
        return (
            f'<td class="value-cell na" data-group-key="{escape(leaf.group_key)}" '
            f'data-kind="{escape(leaf.kind)}" data-scope="{escape(leaf.scope or "")}">-</td>'
        )

    classes = ["value-cell"]
    style = ""
    has_value_range = (
        row.max_value is not None
        and row.min_value is not None
        and abs(row.max_value - row.min_value) > EPSILON
    )

    if row.max_value is not None and row.min_value is not None:
        style = f' style="background:{value_background(value, row.min_value, row.max_value)}"'
        if has_value_range and row.numeric_count > 1 and abs(value - row.max_value) <= EPSILON:
            classes.append("value-best")
        if has_value_range and row.numeric_count > 1 and abs(value - row.min_value) <= EPSILON:
            classes.append("value-worst")

    return (
        f'<td class="{" ".join(classes)}" data-group-key="{escape(leaf.group_key)}" '
        f'data-kind="{escape(leaf.kind)}" data-scope="{escape(leaf.scope or "")}" '
        f'data-value="{escape(format_number(value))}" data-group-title="{escape(leaf.group_title)}" '
        f'data-group-meta="{escape(leaf.group_meta)}"{style}>'
        f'<div class="value-main" title="{escape(format_number(value))}">{escape(format_display_number(value))}</div>'
        '<div class="value-chip-row"></div></td>'
    )


def render_section_table(section: SectionData) -> str:
    body_lines = ["<tbody>"]
    for row_index, row in enumerate(section.rows):
        metric_key = metric_filter_key(section.spec.key, row.spec.labels[-1])
        task_key = task_filter_key(section.spec.key, build_task_label(row.spec))
        row_label = build_row_label(row.spec)
        line = [
            f'<tr data-metric-key="{escape(metric_key)}" data-section-key="{escape(section.spec.key)}" '
            f'data-task-key="{escape(task_key)}" data-section-title="{escape(section.spec.title)}" data-row-label="{escape(row_label)}">'
        ]
        for label_index, label in enumerate(row.spec.labels, start=1):
            classes = f"row-head sticky-{label_index}"
            line.append(f'<td class="{classes}">{escape(label)}</td>')
        for leaf in section.leaf_columns:
            line.append(render_value_cell(row, leaf, section.gap_threshold))
        line.append("</tr>")
        body_lines.append("".join(line))
    body_lines.append("</tbody>")

    return (
        f'<div class="table-shell" data-table-container id="table-{escape(section.spec.key)}">'
        '<div class="note filtered-empty is-hidden">当前没有可见内容：可能是版本列 / think-nothink 列都隐藏了，或者任务 / Metric 都被取消勾选了。</div><table data-filterable-table>'
        + render_table_colgroup(section.spec, section.leaf_columns)
        + render_table_header(section.spec, section.leaf_columns)
        + "".join(body_lines)
        + "</table></div>"
    )


def render_highlight_table(items: list[HighlightItem], positive: bool) -> str:
    if not items:
        return '<div class="note">当前没有可展示的 think/nothink 配对差异。</div>'

    body_key = "positive" if positive else "negative"
    rows = []
    for item in items:
        delta_class = "delta-pos" if item.delta >= 0 else "delta-neg"
        scope_text = item.scope or "-"
        rows.append(
            f'<tr class="highlight-row" data-group-key="{escape(item.group_key)}" data-metric-key="{escape(item.metric_key)}">'
            f"<td>{escape(item.section_key)}</td>"
            f"<td>{escape(item.row_label)}</td>"
            f"<td>{escape(item.group_title)}<br><span class=\"subtle\">{escape(item.group_meta)}</span></td>"
            f"<td>{escape(scope_text)}</td>"
            f"<td class=\"{delta_class}\">{escape(format_signed(item.delta))}</td>"
            "</tr>"
        )

    title = "提升" if positive else "下降"
    return (
        '<div class="mini-table-shell" data-table-container>'
        '<div class="note filtered-empty is-hidden">当前没有可见内容：可能是版本列 / think 列被隐藏了，或者任务 / Metric 都被取消勾选了。</div>'
        '<table class="mini-table">'
        "<thead><tr><th>分层</th><th>任务 / 指标</th><th>模型组</th><th>范围</th><th>"
        f"{title} Δ"
        f'</th></tr></thead><tbody data-highlight-body="{body_key}">'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_metric_controls(section: SectionData) -> str:
    metric_names = list(OrderedDict((row.spec.labels[-1], None) for row in section.rows))
    toggles = []
    for metric_name in metric_names:
        metric_key = metric_filter_key(section.spec.key, metric_name)
        toggles.append(
            '<label class="metric-toggle">'
            f'<input type="checkbox" checked data-section-key="{escape(section.spec.key)}" data-filter-metric-key="{escape(metric_key)}" />'
            f'<span>{escape(metric_name)}</span>'
            '</label>'
        )

    return (
        '<div class="metric-panel">'
        '<div class="metric-panel-head">'
        '<div class="subtle">Metric 显隐主作用于当前分层表格；显著 gap 速览、复制和 PNG 导出也会跟随当前筛选结果。</div>'
        '<div class="button-row">'
        f'<button type="button" data-action="select-all-metrics" data-section-key="{escape(section.spec.key)}">全选 Metric</button>'
        f'<button type="button" data-action="clear-metrics" data-section-key="{escape(section.spec.key)}">清空 Metric</button>'
        '</div></div>'
        f'<div class="metric-grid">{"".join(toggles)}</div>'
        '</div>'
    )


def render_task_controls(section: SectionData) -> str:
    task_counts: OrderedDict[str, int] = OrderedDict()
    for row in section.rows:
        task_label = build_task_label(row.spec)
        task_counts[task_label] = task_counts.get(task_label, 0) + 1

    toggles = []
    restore_items = []
    for task_label, metric_count in task_counts.items():
        task_key = task_filter_key(section.spec.key, task_label)
        meta = f"{metric_count} 个 Metric"
        toggles.append(
            '<label class="task-toggle">'
            f'<input type="checkbox" checked data-section-key="{escape(section.spec.key)}" data-filter-task-key="{escape(task_key)}" />'
            f'<span title="{escape(task_label)}">{escape(task_label)}</span>'
            '</label>'
        )
        restore_items.append(
            f'<label class="version-restore-toggle is-hidden" data-restore-task-item data-restore-task-key="{escape(task_key)}">'
            f'<input type="checkbox" data-action="restore-task" data-section-key="{escape(section.spec.key)}" data-restore-task-key="{escape(task_key)}" />'
            '<div class="version-restore-body">'
            f'<div class="version-restore-main" title="{escape(task_label)}">{escape(task_label)}</div>'
            f'<div class="version-restore-sub">{escape(meta)}</div>'
            '</div></label>'
        )

    return (
        '<div class="metric-panel task-panel">'
        '<div class="metric-panel-head">'
        '<div class="subtle">任务显隐主作用于当前分层表格；可以先全部隐藏，再从下面的隐藏任务池里反选需要展示的任务。</div>'
        '<div class="button-row">'
        f'<button type="button" data-action="select-all-tasks" data-section-key="{escape(section.spec.key)}">全选任务</button>'
        f'<button type="button" data-action="clear-tasks" data-section-key="{escape(section.spec.key)}">全部隐藏</button>'
        '</div></div>'
        f'<div class="task-grid">{"".join(toggles)}</div>'
        f'<div class="version-hidden-panel is-hidden" data-hidden-task-panel data-section-key="{escape(section.spec.key)}">'
        '<div class="version-hidden-head">'
        '<div class="subtle">已隐藏任务：勾选后会立即恢复到当前表格里。</div>'
        '<div class="button-row">'
        f'<button type="button" data-action="restore-hidden-tasks" data-section-key="{escape(section.spec.key)}">恢复全部隐藏任务</button>'
        '</div></div>'
        f'<div class="version-hidden-grid">{"".join(restore_items)}</div>'
        '</div>'
        '</div>'
    )


def render_scope_controls(section: SectionData) -> str:
    section_key = section.spec.key
    scope_toggles = []
    for scope in section.spec.scope_headers:
        scope_toggles.append(
            '<label class="scope-filter-toggle">'
            f'<input type="checkbox" checked data-section-key="{escape(section_key)}" data-filter-scope="{escape(scope)}" />'
            f'<span>{escape(scope)}</span>'
            '</label>'
        )
    return (
        f'<div class="metric-panel scope-filter-panel" data-scope-panel data-section-key="{escape(section_key)}">'
        '<div class="metric-panel-head">'
        '<div class="subtle">显隐单域 / 全域列；复制、导出也会跟随当前可见列。</div>'
        '<div class="button-row">'
        f'<button type="button" data-action="select-all-scopes" data-section-key="{escape(section_key)}">全选</button>'
        f'<button type="button" data-action="clear-scopes" data-section-key="{escape(section_key)}">清空</button>'
        '</div></div>'
        f'<div class="scope-filter-grid">{"".join(scope_toggles)}</div>'
        '</div>'
    )


def render_kind_controls(section_key: str) -> str:
    return (
        f'<div class="metric-panel kind-panel" data-kind-panel data-section-key="{escape(section_key)}">'
        '<div class="metric-panel-head">'
        '<div class="subtle">当前表格可单独隐藏 think / nothink 列；复制、导出和 gap 速览都会跟随这里的可见列。</div>'
        '<div class="button-row">'
        f'<button type="button" data-action="select-all-kinds" data-section-key="{escape(section_key)}">全选列类型</button>'
        f'<button type="button" data-action="show-only-nothink" data-section-key="{escape(section_key)}">仅 nothink</button>'
        f'<button type="button" data-action="show-only-think" data-section-key="{escape(section_key)}">仅 think</button>'
        f'<button type="button" data-action="clear-kinds" data-section-key="{escape(section_key)}">全部隐藏</button>'
        '</div></div>'
        '<div class="kind-grid">'
        '<label class="kind-toggle">'
        f'<input type="checkbox" checked data-section-key="{escape(section_key)}" data-filter-kind="nothink" />'
        '<span class="kind-badge nothink">N</span><span>nothink</span>'
        '</label>'
        '<label class="kind-toggle">'
        f'<input type="checkbox" checked data-section-key="{escape(section_key)}" data-filter-kind="think" />'
        '<span class="kind-badge think">T</span><span>think</span>'
        '</label>'
        '</div></div>'
    )


def render_width_controls(section_key: str) -> str:
    return (
        '<div class="width-panel">'
        '<div class="width-panel-head">'
        '<div class="subtle">这里是绝对像素宽度；只影响当前表格。超出可视区会出现横向滚动，收窄后右侧保留留白。</div>'
        '<div class="button-row">'
        f'<button type="button" data-width-preset="104" data-section-key="{escape(section_key)}">紧凑</button>'
        f'<button type="button" data-width-preset="132" data-section-key="{escape(section_key)}">标准</button>'
        f'<button type="button" data-width-preset="168" data-section-key="{escape(section_key)}">宽屏</button>'
        f'<button type="button" data-width-preset="208" data-section-key="{escape(section_key)}">超宽</button>'
        '</div></div>'
        '<div class="width-control-row">'
        f'<div class="width-readout">数据列宽：<span data-table-width-value data-section-key="{escape(section_key)}">122px</span></div>'
        '<div class="width-input-group">'
        f'<input type="range" min="96" max="240" step="2" value="122" data-table-width-range data-section-key="{escape(section_key)}" />'
        f'<input type="number" min="96" max="240" step="2" value="122" data-table-width-number data-section-key="{escape(section_key)}" />'
        '</div>'
        '</div>'
        '<div class="width-control-row">'
        f'<div class="width-readout">标签列宽：<span data-label-width-value data-section-key="{escape(section_key)}">168px</span></div>'
        '<div class="width-input-group">'
        f'<input type="range" min="60" max="420" step="4" value="168" data-label-width-range data-section-key="{escape(section_key)}" />'
        f'<input type="number" min="60" max="420" step="4" value="168" data-label-width-number data-section-key="{escape(section_key)}" />'
        '</div>'
        '</div></div>'
    )


def render_section(section: SectionData, models: list[ModelSpec]) -> str:
    chips = [
        f"表格行数：{len(section.rows)}",
        f"有效值：{section.present_cells}/{section.total_cells}",
        f"P80 gap 阈值：{format_number(section.gap_threshold)}",
        f"显著提升：{section.significant_up}",
        f"显著下降：{section.significant_down}",
    ]
    chips_html = "".join(f'<span class="chip">{escape(item)}</span>' for item in chips)

    return (
        f'<section class="card table-card" id="section-{escape(section.spec.key)}" data-width-scope="{escape(section.spec.key)}" data-dashboard-section-key="{escape(section.spec.key)}">'
        f'<h2>{escape(section.spec.title)}</h2>'
        f'<p class="subtle">{escape(section.spec.description)}</p>'
        f'<div class="section-meta">{chips_html}</div>'
        + render_version_controls(models, section_key=section.spec.key, title="当前表格版本显隐")
        + render_pairing_controls(models, section.spec.key)
        + render_kind_controls(section.spec.key)
        + (render_scope_controls(section) if section.spec.scope_headers else "")
        + render_task_controls(section)
        + render_width_controls(section.spec.key)
        + render_metric_controls(section)
        +
        '<div class="table-actions">'
        '<div class="subtle">下面的复制和导出只包含当前可见版本列、当前可见 think/nothink 列，以及当前可见任务 / Metric 行。</div>'
        '<div class="button-row">'
        f'<button type="button" data-action="copy-table" data-section-key="{escape(section.spec.key)}">复制当前表格</button>'
        f'<button type="button" data-action="export-png" data-section-key="{escape(section.spec.key)}">导出 PNG</button>'
        '</div></div>'
        f'<div id="export-{escape(section.spec.key)}">'
        + render_section_table(section)
        + '</div>'
        + "</section>"
    )


def render_version_controls(models: list[ModelSpec], section_key: str, title: str = "版本显隐") -> str:
    groups = build_group_specs(models)

    items = []
    restore_items = []
    for group in groups:
        kind_badges = []
        if group.has_nothink:
            kind_badges.append('<span class="kind-badge nothink" title="包含 nothink 结果">N</span>')
        if group.has_think:
            kind_badges.append('<span class="kind-badge think" title="包含 think 结果">T</span>')

        attrs = [
            'class="version-toggle"',
            f'data-group-key="{escape(group.group_key)}"',
            f'data-section-key="{escape(section_key)}"',
            f'data-group-title="{escape(group.title)}"',
            f'data-group-meta="{escape(group.meta)}"',
            f'data-has-think="{1 if group.has_think else 0}"',
            f'data-has-nothink="{1 if group.has_nothink else 0}"',
            'draggable="true"',
            'title="拖动可调整全局顺序"',
        ]

        items.append(
            f'<div {" ".join(attrs)}>'
            f'<input type="checkbox" checked data-filter-group-key="{escape(group.group_key)}" data-section-key="{escape(section_key)}" />'
            '<div class="version-card-body">'
            f'<div class="version-toggle-main" title="{escape(group.title)}">{escape(group.title)}</div>'
            f'<div class="version-toggle-sub">{escape(group.meta)}</div>'
            f'<div class="version-kind-row">{"".join(kind_badges)}</div>'
            '</div></div>'
        )

        restore_items.append(
            f'<label class="version-restore-toggle is-hidden" data-restore-item data-restore-group-key="{escape(group.group_key)}">'
            f'<input type="checkbox" data-action="restore-group" data-section-key="{escape(section_key)}" data-restore-group-key="{escape(group.group_key)}" />'
            '<div class="version-restore-body">'
            f'<div class="version-restore-main" title="{escape(group.title)}">{escape(group.title)}</div>'
            f'<div class="version-restore-sub">{escape(group.meta)}</div>'
            f'<div class="version-kind-row">{"".join(kind_badges)}</div>'
            '</div></label>'
        )

    hint_text = f"{title}：可以先隐藏不想展示的版本，也可以直接拖动卡片调整前后顺序；这些设置只作用于当前表格。"

    return (
        '<div class="version-panel">'
        '<div class="version-panel-head">'
        f'<div class="subtle">{escape(hint_text)}</div>'
        '<div class="button-row">'
        f'<button type="button" data-action="select-all" data-section-key="{escape(section_key)}">全选</button>'
        f'<button type="button" data-action="clear-all" data-section-key="{escape(section_key)}">清空</button>'
        '</div></div>'
        f'<div class="version-grid">{"".join(items)}</div>'
        '<div class="version-hidden-panel is-hidden" data-hidden-version-panel>'
        '<div class="version-hidden-head">'
        '<div class="subtle">已隐藏版本：下面勾选后会立即恢复到当前表格里。</div>'
        '<div class="button-row">'
        f'<button type="button" data-action="restore-hidden" data-section-key="{escape(section_key)}">恢复全部隐藏</button>'
        '</div></div>'
        f'<div class="version-hidden-grid">{"".join(restore_items)}</div>'
        '</div></div>'
    )


def render_pairing_controls(models: list[ModelSpec], section_key: str) -> str:
    groups = build_group_specs(models)
    think_groups = [group for group in groups if group.has_think]
    nothink_groups = [group for group in groups if group.has_nothink]
    if not think_groups or not nothink_groups:
        return ""

    items = []
    for think_group in think_groups:
        default_group_key = default_pair_group_key(think_group, nothink_groups) or ""
        options = []
        for candidate in nothink_groups:
            selected = ' selected' if candidate.group_key == default_group_key else ""
            options.append(
                f'<option value="{escape(candidate.group_key)}"{selected}>'
                f'{escape(candidate.title)} · {escape(candidate.meta)}'
                '</option>'
            )
        items.append(
            f'<label class="pairing-item" data-group-key="{escape(think_group.group_key)}" data-kind="think">'
            f'<div class="pairing-item-main">{escape(think_group.title)}</div>'
            f'<div class="pairing-item-sub">{escape(think_group.meta)}</div>'
            f'<select data-pair-select data-section-key="{escape(section_key)}" '
            f'data-think-group-key="{escape(think_group.group_key)}" '
            f'data-default-value="{escape(default_group_key)}">'
            + "".join(options)
            + '</select></label>'
        )

    return (
        '<div class="pairing-panel">'
        '<div class="pairing-panel-head">'
        '<div class="subtle">Think/Nothink 对照只作用于当前分表的 `TN` badge；`GS` badge 固定表示"全域相对单域"。</div>'
        f'<button type="button" data-action="reset-section-pairs" data-section-key="{escape(section_key)}">恢复默认对照</button>'
        '</div>'
        f'<div class="pairing-grid">{"".join(items)}</div>'
        '</div>'
    )


def build_summary_cards(
    models: list[ModelSpec],
    sections: list[SectionData],
    all_highlights: list[HighlightItem],
) -> list[dict[str, str]]:
    total_rows = sum(len(section.rows) for section in sections)
    present_cells = sum(section.present_cells for section in sections)
    total_cells = sum(section.total_cells for section in sections)
    group_count = len(OrderedDict((model.group_key, None) for model in models))

    best_up = max(all_highlights, key=lambda item: item.delta, default=None)
    worst_down = min(all_highlights, key=lambda item: item.delta, default=None)

    return [
        {
            "title": "模型目录",
            "main": str(len(models)),
            "sub": "选中目录数量",
        },
        {
            "title": "版本分组",
            "main": str(group_count),
            "sub": "同组内按 nothink / think 排布",
        },
        {
            "title": "表格总行数",
            "main": str(total_rows),
            "sub": "L0 / L1_i2i / L2_multi-hop / reasoning_bench / rec",
        },
        {
            "title": "有效值覆盖",
            "main": f"{present_cells}/{total_cells}",
            "sub": "缺失值显示为 -",
        },
        {
            "title": "最大 think 提升",
            "main": format_signed(best_up.delta) if best_up else "-",
            "sub": f"{best_up.section_key} · {best_up.row_label}" if best_up else "没有可用配对",
        },
        {
            "title": "最大 think 下降",
            "main": format_signed(worst_down.delta) if worst_down else "-",
            "sub": f"{worst_down.section_key} · {worst_down.row_label}" if worst_down else "没有可用配对",
        },
    ]


def render_summary_cards(cards: list[dict[str, str]]) -> str:
    html_parts = ['<div class="summary-grid">']
    for card in cards:
        html_parts.append(
            '<section class="summary-item">'
            f'<div class="tag">{escape(card["title"])}</div>'
            f'<div class="value">{escape(card["main"])}</div>'
            f'<div class="desc">{escape(card["sub"])}</div>'
            "</section>"
        )
    html_parts.append("</div>")
    return "".join(html_parts)


def render_race_score_cards(cards: list[dict]) -> str:
    if not cards:
        return ""

    html_parts = [
        '<section class="card race-card-section" id="section-race-mode">',
        '<h2>Race Mode Scores</h2>',
        '<p class="subtle">按固定任务-指标-权重计算 `R0 / R1 / R2 / R3`。若某个目录缺少依赖任务或指标，对应分数显示为 `-`。</p>',
        '<div class="race-score-grid">',
    ]

    score_order = ("R0", "R1", "R2", "R3")
    for card in cards:
        html_parts.append(
            '<div class="race-model-pair">'
            '<section class="race-score-card">'
            '<div class="race-score-head"><div>'
            f'<div class="race-score-title">{escape(card["group_title"])}</div>'
            f'<div class="race-score-meta">{escape(card["group_meta"])} · {escape(card["kind"])}</div>'
            '</div></div>'
            '<div class="race-score-values">'
        )

        for score_name in score_order:
            score_value = card["scores"].get(score_name)
            html_parts.append(
                '<div class="race-score-item">'
                f'<div class="race-score-label">{escape(score_name)}</div>'
                f'<div class="race-score-value">{escape(format_display_number(score_value))}</div>'
                '</div>'
            )

        html_parts.append('</div>')

        missing = card.get("missing", {})
        if missing:
            html_parts.append('<div class="race-score-missing">')
            for score_name in score_order:
                dependencies = missing.get(score_name)
                if not dependencies:
                    continue
                html_parts.append(
                    '<div class="race-score-missing-item">'
                    f'<span class="race-score-missing-label">{escape(score_name)} 缺失：</span>'
                    f'<span>{escape(", ".join(dependencies))}</span>'
                    '</div>'
                )
            html_parts.append('</div>')

        html_parts.append(
            '</section>'
            '<section class="race-contribution-card">'
            '<div class="race-score-head"><div>'
            '<div class="race-score-title">Subtask Contributions</div>'
            f'<div class="race-score-meta">{escape(card["group_title"])} · {escape(card["kind"])}</div>'
            '</div></div>'
            '<div class="race-contribution-groups">'
        )

        contributions = card.get("contributions", {})
        for score_name in score_order:
            html_parts.append(
                '<div class="race-contribution-group">'
                f'<div class="race-contribution-group-title">{escape(score_name)}</div>'
                '<div class="race-contribution-rows">'
            )
            for contribution in contributions.get(score_name, []):
                html_parts.append(
                    '<div class="race-contribution-row">'
                    f'<div class="race-contribution-task">{escape(str(contribution["task"]))}</div>'
                    f'<div class="race-contribution-value">{escape(format_display_number(contribution["value"]))}</div>'
                    '</div>'
                )
            html_parts.append('</div></div>')

        html_parts.append('</div></section></div>')

    html_parts.append('</div></section>')
    return "".join(html_parts)


def render_dashboard(
    title: str,
    source_description: str,
    models: list[ModelSpec],
    sections: list[SectionData],
    race_score_cards: list[dict],
) -> str:
    all_highlights = [item for section in sections for item in section.highlights]
    top_up = sorted(
        (item for item in all_highlights if item.delta > 0),
        key=lambda item: abs(item.delta),
        reverse=True,
    )[:8]
    top_down = sorted(
        (item for item in all_highlights if item.delta < 0),
        key=lambda item: abs(item.delta),
        reverse=True,
    )[:8]

    summary_cards = build_summary_cards(models, sections, all_highlights)
    section_html = "".join(render_section(section, models) for section in sections)

    # Build section nav bar
    nav_items = [
        '<button type="button" class="nav-link" data-nav-target="section-race-mode">Race</button>',
        '<button type="button" class="nav-link" data-nav-target="section-summary-gap">Gap 速览</button>',
    ]
    for s in sections:
        nav_items.append(f'<button type="button" class="nav-link" data-nav-target="section-{escape(s.spec.key)}">{escape(s.spec.title)}</button>')
    nav_html = '<nav class="section-nav">' + "".join(nav_items) + '</nav>'

    body_html = (
        '<section class="card hero">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">'
        f'<h1 style="margin:0;">{escape(title)}</h1>'
        '<button type="button" data-action="download-html" style="flex-shrink:0;padding:6px 16px;border-radius:10px;border:1px solid #cbd5e1;background:#fff;color:#0f172a;font-size:13px;font-weight:600;cursor:pointer;">下载 HTML</button>'
        '</div>'
        '<p class="subtle">底色深浅表示同一行里的绝对值高低；绿色/红色描边表示 think 相对 nothink 的显著提升或下降。带 scope 的表会同时展示 `TN` 与 `GS` 两种差值 badge。</p>'
        '<div class="chips">'
        f'<span class="chip">生成时间：{escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</span>'
        f'<span class="chip">{escape(source_description)}</span>'
        '</div>'
        '<div class="legend">'
        '<span class="legend-item"><i class="swatch" style="background: var(--blue-soft);"></i>nothink 列头</span>'
        '<span class="legend-item"><i class="swatch" style="background: var(--orange-soft);"></i>think 列头</span>'
        '<span class="legend-item">`TN`：think 相对所选 nothink 的提升/衰减</span>'
        '<span class="legend-item">`GS`：全域相对单域的提升/衰减</span>'
        '<span class="legend-item"><i class="swatch" style="background: var(--green);"></i>显著提升</span>'
        '<span class="legend-item"><i class="swatch" style="background: var(--red);"></i>显著下降</span>'
        '<span class="legend-item"><i class="swatch" style="background: rgba(59,130,246,0.24);"></i>绝对值更优</span>'
        "</div>"
        + render_summary_cards(summary_cards)
        + "</section>"
        + render_race_score_cards(race_score_cards)
        + nav_html
        +
        '<section class="card" data-dashboard-section-key="summary-gap">'
        '<h2>显著 gap 速览（TN）</h2>'
        '<p class="subtle">这里先按各分层内 think - nothink 的绝对差值排序；`GS`（全域相对单域）会直接展示在表格单元格里。这里的"显著"仅用于看板高亮，不代表统计显著性检验。版本显隐和各分层任务 / Metric 勾选都会联动到这里。</p>'
        + render_version_controls(models, section_key="summary-gap", title="gap 速览版本显隐")
        +
        '<div class="two-col">'
        f'<div><h3>提升 Top 8</h3>{render_highlight_table(top_up, positive=True)}</div>'
        f'<div><h3>下降 Top 8</h3>{render_highlight_table(top_down, positive=False)}</div>'
        "</div>"
        '<div class="note">如果某个版本没有对应任务，表格会保留列但显示为 - ，这样可以直接看出"没评"而不是误以为 0。</div>'
        "</section>"
        + section_html
    )

    html = HTML_TEMPLATE.replace("__TITLE__", escape(title))
    html = html.replace("__BODY__", body_html)
    return html
