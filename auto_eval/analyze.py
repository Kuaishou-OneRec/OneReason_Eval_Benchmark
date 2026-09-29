#!/usr/bin/env python3
"""从 eval_results 原始评估目录直接生成静态 HTML 对比看板。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path


from auto_eval.dashboard.sections import (
    DEFAULT_BINDING,
    GLOBAL_SCOPE,
    SINGLE_SCOPE,
    EPSILON,
    MetricBinding,
    RowSpec,
    SectionSpec,
    SECTION_SPECS,
    metric_filter_key,
    task_filter_key,
)


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




HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__TITLE__</title>
  <style id="dashboard-style">
    :root {
      --bg: #f5f7fb;
      --card: #ffffff;
      --line: #e5e7eb;
      --line-strong: #cbd5e1;
      --text: #111827;
      --subtle: #64748b;
      --blue: #3b82f6;
      --orange: #f59e0b;
      --green: #10b981;
      --red: #ef4444;
      --blue-soft: rgba(59, 130, 246, 0.16);
      --orange-soft: rgba(245, 158, 11, 0.18);
      --green-soft: rgba(16, 185, 129, 0.12);
      --red-soft: rgba(239, 68, 68, 0.12);
      --shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
      --radius: 18px;
      --page-max-width: 2600px;
      --leaf-col-width: 122px;
      --sticky-col-1-width: 170px;
      --label-col-width: 168px;
      --version-toggle-width: 154px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }

    .page {
      width: min(100%, var(--page-max-width));
      margin: 0 auto;
      padding: 24px;
    }

    .card {
      background: var(--card);
      border: 1px solid rgba(229, 231, 235, 0.92);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 22px 24px;
      margin-bottom: 18px;
    }

    h1, h2, h3, h4, p { margin: 0; }

    h1 {
      font-size: 30px;
      margin-bottom: 10px;
    }

    h2 {
      font-size: 21px;
      margin-bottom: 8px;
    }

    h3 {
      font-size: 16px;
      margin-bottom: 10px;
    }

    .subtle {
      color: var(--subtle);
      font-size: 14px;
    }

    .hero p + p {
      margin-top: 6px;
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 14px 0 0;
      font-size: 13px;
      color: var(--subtle);
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      background: #f8fafc;
      border: 1px solid var(--line);
    }

    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 4px;
      display: inline-block;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    .chip {
      padding: 6px 10px;
      border-radius: 999px;
      background: #f8fafc;
      border: 1px solid var(--line);
      font-size: 13px;
      color: var(--subtle);
    }

    .summary-grid,
    .two-col {
      display: grid;
      gap: 14px;
    }

    .summary-grid {
      grid-template-columns: repeat(6, minmax(0, 1fr));
      margin-top: 16px;
    }

    .two-col {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 16px;
    }

    .summary-item {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
      background: linear-gradient(180deg, #fbfdff 0%, #f8fafc 100%);
    }

    .summary-item .tag {
      color: var(--subtle);
      font-size: 13px;
      margin-bottom: 6px;
    }

    .summary-item .value {
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 4px;
    }

    .summary-item .desc {
      color: var(--subtle);
      font-size: 13px;
    }

    .mini-table-shell,
    .table-shell {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fff;
    }

    table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 14px;
    }

    .table-shell table {
      width: max-content;
      min-width: 0;
      table-layout: fixed;
    }

    .mini-table-shell table {
      width: 100%;
      min-width: 0;
      table-layout: auto;
    }

    th,
    td {
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 6px 8px;
      vertical-align: middle;
      height: 44px;
    }

    button {
      border: 1px solid var(--line);
      background: #fff;
      color: #0f172a;
      border-radius: 10px;
      padding: 7px 12px;
      font: inherit;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    button:hover {
      border-color: #93c5fd;
      background: #eff6ff;
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.45;
      background: #f8fafc;
    }

    th:last-child,
    td:last-child {
      border-right: 0;
    }

    thead th {
      background: #f8fafc;
      color: #334155;
      font-weight: 700;
      text-align: center;
    }

    tbody td {
      background: #fff;
    }

    tbody tr:nth-child(odd) td:not(.row-head) {
      background: rgba(248, 250, 252, 0.55);
    }

    .row-head {
      position: sticky;
      z-index: 2;
      background: #ffffff !important;
      color: #0f172a;
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .head-sticky {
      background: #eef2ff !important;
      z-index: 3;
    }

    .sticky-1 {
      left: 0;
      width: var(--sticky-col-1-width);
      min-width: var(--sticky-col-1-width);
      max-width: var(--sticky-col-1-width);
    }
    .sticky-2 {
      left: var(--sticky-col-1-width);
      width: var(--label-col-width);
      min-width: var(--label-col-width);
      max-width: var(--label-col-width);
    }
    .sticky-3 {
      left: calc(var(--sticky-col-1-width) + var(--label-col-width));
      width: var(--label-col-width);
      min-width: var(--label-col-width);
      max-width: var(--label-col-width);
    }

    .sticky-col-1 { width: var(--sticky-col-1-width); }
    .sticky-col-2 { width: var(--label-col-width); }
    .sticky-col-3 { width: var(--label-col-width); }
    .leaf-col { width: var(--leaf-col-width); }

    .group-head {
      min-width: 110px;
      padding: 8px 6px;
    }

    .col-index-spacer,
    .col-index-head {
      background: #eef2f7 !important;
      color: #475569;
      font-size: 12px;
      font-weight: 700;
      line-height: 1;
      white-space: nowrap;
      height: 32px;
    }

    .col-index-spacer {
      text-align: center;
    }

    .col-index-head {
      width: var(--leaf-col-width);
      min-width: var(--leaf-col-width);
      max-width: var(--leaf-col-width);
      text-align: center;
    }

    .group-title {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 20px;
      font-size: 14px;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .group-meta {
      display: flex;
      align-items: center;
      justify-content: center;
      margin-top: 4px;
      min-height: 16px;
      font-size: 12px;
      color: var(--subtle);
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .kind-head.nothink {
      background: var(--blue-soft);
      color: #1d4ed8;
    }

    .kind-head.think {
      background: var(--orange-soft);
      color: #b45309;
    }

    .scope-head {
      font-size: 13px;
      color: var(--subtle);
      width: var(--leaf-col-width);
      min-width: var(--leaf-col-width);
      max-width: var(--leaf-col-width);
    }

    .value-cell {
      width: var(--leaf-col-width);
      min-width: var(--leaf-col-width);
      max-width: var(--leaf-col-width);
      position: relative;
      text-align: center;
      font-variant-numeric: tabular-nums;
      overflow: hidden;
      padding: 6px 4px;
    }

    .value-cell.has-delta {
      background-clip: padding-box;
    }

    .value-main {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 16px;
      padding: 0 4px;
      position: relative;
      z-index: 1;
      font-weight: 600;
      color: #0f172a;
      line-height: 1.1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .value-chip-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
      flex-wrap: wrap;
      min-height: 14px;
      margin-top: 4px;
      position: relative;
      z-index: 1;
    }

    .value-best {
      z-index: 0;
    }

    .value-best::before {
      content: "最佳";
      position: absolute;
      top: 3px;
      right: 4px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 1px 5px;
      border-radius: 999px;
      background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
      color: #ffffff;
      font-size: 9px;
      font-weight: 900;
      line-height: 1;
      letter-spacing: 0.02em;
      box-shadow: 0 2px 6px rgba(185, 28, 28, 0.22);
      pointer-events: none;
      z-index: 3;
    }

    .value-best::after {
      content: "";
      position: absolute;
      inset: 1px;
      border-radius: 6px;
      border: 3px solid rgba(220, 38, 38, 0.92);
      box-shadow:
        inset 0 0 0 1px rgba(255, 255, 255, 0.72),
        0 0 0 1px rgba(220, 38, 38, 0.08);
      pointer-events: none;
      z-index: 2;
    }

    .value-best .value-main {
      font-weight: 900;
      color: #b91c1c;
      text-shadow: 0 0 0.01px rgba(185, 28, 28, 0.45);
    }

    .export-root .value-best::before {
      background: #dc2626 !important;
      box-shadow: none !important;
      color: #ffffff !important;
    }

    .export-root .value-best::after {
      border: 3px solid #dc2626 !important;
      box-shadow: none !important;
    }

    .export-root .value-best .value-main {
      color: #991b1b !important;
      text-shadow: none !important;
    }

    .value-worst .value-main {
      color: #475569;
    }

    .delta-chip,
    .scope-chip {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      max-width: 52px;
      padding: 2px 6px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.16);
      font-size: 9px;
      font-weight: 700;
      line-height: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .delta-up .delta-chip,
    .scope-up .scope-chip {
      color: #047857;
      background: rgba(16, 185, 129, 0.16);
    }
    .delta-down .delta-chip,
    .scope-down .scope-chip {
      color: #c2410c;
      background: rgba(249, 115, 22, 0.18);
    }
    .delta-flat .delta-chip,
    .scope-flat .scope-chip {
      color: var(--subtle);
      background: rgba(148, 163, 184, 0.18);
    }

    .scope-chip {
      border: 1px solid rgba(59, 130, 246, 0.18);
    }

    .sig-up {
      box-shadow: inset 0 0 0 2px rgba(16, 185, 129, 0.55);
    }

    .sig-down {
      box-shadow: inset 0 0 0 2px rgba(239, 68, 68, 0.55);
    }

    .export-root .sig-up,
    .export-root .sig-down {
      box-shadow: none !important;
      background-clip: padding-box !important;
    }

    .export-root .sig-up {
      outline: 2px solid #10b981 !important;
      outline-offset: -2px;
    }

    .export-root .sig-down {
      outline: 2px solid #ef4444 !important;
      outline-offset: -2px;
    }

    .na {
      background: #f8fafc !important;
      color: #94a3b8;
      text-align: center;
      font-style: italic;
    }

    .section-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0 14px;
    }

    .section-meta .chip {
      background: #fff;
    }

    .mini-table th,
    .mini-table td {
      min-width: auto;
      text-align: left;
      font-size: 12px;
    }

    .mini-table td:last-child,
    .mini-table th:last-child {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .mini-table .delta-pos {
      color: #047857;
      font-weight: 700;
    }

    .mini-table .delta-neg {
      color: #c2410c;
      font-weight: 700;
    }

    .note {
      margin-top: 10px;
      padding: 10px 12px;
      border-left: 4px solid #c7d2fe;
      background: #eef2ff;
      color: #4338ca;
      border-radius: 10px;
      font-size: 13px;
    }

    .version-panel {
      margin-top: 16px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, #fbfdff 0%, #f8fafc 100%);
      padding: 16px;
    }

    .metric-panel {
      margin-bottom: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #f8fafc;
      padding: 12px;
    }

    .version-panel-head,
    .metric-panel-head,
    .table-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }

    .version-panel-head {
      margin-bottom: 12px;
    }

    .metric-panel-head {
      margin-bottom: 10px;
    }

    .version-hidden-panel {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px dashed var(--line-strong);
    }

    .version-hidden-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }

    .version-hidden-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .version-restore-toggle {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      min-width: min(240px, 100%);
      max-width: 100%;
      padding: 10px 12px;
      border: 1px dashed #cbd5e1;
      border-radius: 12px;
      background: #ffffff;
      color: var(--text);
      cursor: pointer;
    }

    .version-restore-toggle:hover {
      border-color: #94a3b8;
      background: #f8fafc;
    }

    .version-restore-toggle input {
      margin-top: 2px;
      flex: 0 0 auto;
    }

    .version-restore-body {
      min-width: 0;
    }

    .version-restore-main {
      font-size: 13px;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .version-restore-sub {
      margin-top: 2px;
      font-size: 12px;
      color: var(--subtle);
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .button-row,
    .version-grid,
    .metric-grid,
    .task-grid,
    .kind-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .version-toggle {
      display: inline-flex;
      align-items: flex-start;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      cursor: grab;
      min-width: var(--version-toggle-width);
    }

    .version-toggle.off {
      opacity: 0.48;
      background: #f8fafc;
    }

    .version-toggle.dragging {
      opacity: 0.45;
      cursor: grabbing;
    }

    .version-toggle.drag-over {
      border-color: #60a5fa;
      box-shadow: inset 0 0 0 2px rgba(96, 165, 250, 0.24);
      background: #eff6ff;
    }

    .version-card-body {
      display: flex;
      flex-direction: column;
      gap: 5px;
      min-width: 0;
    }

    .version-kind-row {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }

    .kind-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 20px;
      height: 18px;
      border-radius: 999px;
      font-size: 10px;
      font-weight: 700;
      padding: 0 6px;
    }

    .kind-badge.think {
      color: #b45309;
      background: var(--orange-soft);
    }

    .kind-badge.nothink {
      color: #1d4ed8;
      background: var(--blue-soft);
    }

    .pairing-panel {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      padding: 12px;
    }

    .pairing-panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }

    .pairing-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
    }

    .pairing-item {
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f8fafc;
    }

    .pairing-item-main {
      font-size: 12px;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .pairing-item-sub {
      font-size: 11px;
      color: var(--subtle);
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .pairing-item select {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 6px 8px;
      font: inherit;
      background: #fff;
      color: #0f172a;
    }

    .metric-toggle,
    .task-toggle,
    .kind-toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      cursor: pointer;
      white-space: nowrap;
      font-size: 12px;
    }

    .metric-toggle.off,
    .task-toggle.off,
    .kind-toggle.off {
      opacity: 0.45;
      background: #f1f5f9;
    }

    .version-toggle input {
      margin: 0;
      accent-color: #2563eb;
    }

    .metric-toggle input,
    .task-toggle input,
    .kind-toggle input {
      margin: 0;
      accent-color: #2563eb;
    }

    .version-toggle-main {
      font-size: 13px;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.2;
      white-space: nowrap;
    }

    .version-toggle-sub {
      font-size: 11px;
      color: var(--subtle);
      line-height: 1.2;
      white-space: nowrap;
    }

    .table-actions {
      margin-bottom: 12px;
    }

    .table-actions .subtle {
      font-size: 12px;
    }

    .width-panel {
      margin-top: 16px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, #fbfdff 0%, #f8fafc 100%);
      padding: 16px;
    }

    .width-panel-head,
    .width-control-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }

    .width-panel-head {
      margin-bottom: 12px;
    }

    .width-control-row input[type="range"] {
      flex: 1 1 320px;
      min-width: 220px;
      accent-color: #2563eb;
    }

    .width-input-group {
      display: flex;
      align-items: center;
      gap: 10px;
      flex: 1 1 420px;
      min-width: 280px;
    }

    .width-control-row input[type="number"] {
      width: 92px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 7px 10px;
      font: inherit;
      color: #0f172a;
      background: #fff;
    }

    .width-readout {
      font-size: 13px;
      color: var(--subtle);
      white-space: nowrap;
    }

    .filtered-empty {
      margin-top: 0;
    }

    .is-hidden {
      display: none !important;
    }

    @media (max-width: 1440px) {
      .summary-grid,
      .two-col {
        grid-template-columns: 1fr;
      }

      .page {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    __BODY__
  </div>
  <script>
    function qs(selector, root = document) {
      return root.querySelector(selector);
    }

    function qsa(selector, root = document) {
      return Array.from(root.querySelectorAll(selector));
    }

    function widthStorageKey(sectionKey) {
      return `rec_dashboard_leaf_width::${sectionKey}`;
    }

    function kindStorageKey(sectionKey) {
      return `rec_dashboard_visible_kinds::${sectionKey}`;
    }

    function labelWidthStorageKey(sectionKey) {
      return `rec_dashboard_label_width::${sectionKey}`;
    }

    function pairStorageKey(sectionKey, groupKey) {
      return `rec_dashboard_pair_group::${sectionKey}::${groupKey}`;
    }

    function groupOrderStorageKey(sectionKey) {
      return `rec_dashboard_group_order::${sectionKey}`;
    }

    function getSectionRoot(node) {
      return node?.closest('[data-dashboard-section-key]') || null;
    }

    function getSectionRootByKey(sectionKey) {
      if (!sectionKey) return null;
      return qs(`[data-dashboard-section-key="${sectionKey}"]`);
    }

    function getVisibleGroupKeys(root = document) {
      return new Set(
        qsa('.version-toggle input:checked', root).map((input) => input.dataset.filterGroupKey)
      );
    }

    function getVisibleKinds(root = document) {
      const inputs = qsa('.kind-toggle input', root);
      if (!inputs.length) {
        return new Set(['nothink', 'think']);
      }
      return new Set(inputs.filter((input) => input.checked).map((input) => input.dataset.filterKind));
    }

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function formatDisplayNumber(value) {
      if (!Number.isFinite(value)) return '-';
      const text = value.toFixed(4).replace(/0+$/, '').replace(/\\.$/, '');
      return text || '0';
    }

    function formatSigned(value) {
      if (!Number.isFinite(value)) return '-';
      const sign = value >= 0 ? '+' : '-';
      const text = Math.abs(value).toFixed(4).replace(/0+$/, '').replace(/\\.$/, '');
      return `${sign}${text || '0'}`;
    }

    function formatSignedPercent(value) {
      if (!Number.isFinite(value)) return 'baseline≈0';
      const sign = value >= 0 ? '+' : '-';
      const text = (Math.abs(value) * 100).toFixed(1).replace(/0+$/, '').replace(/\\.$/, '');
      return `${sign}${text || '0'}%`;
    }

    function formatDeltaBadge(delta, relativeDelta) {
      const arrow = delta > 1e-12 ? '↑' : delta < -1e-12 ? '↓' : '≈';
      if (relativeDelta === null || relativeDelta === undefined) {
        return Math.abs(delta) <= 1e-12 ? '≈0%' : `${arrow}new`;
      }
      const text = (Math.abs(relativeDelta) * 100).toFixed(1).replace(/0+$/, '').replace(/\\.$/, '');
      return `${arrow}${text || '0'}%`;
    }

    function getCurrentGroupOrder(root = document) {
      const order = [];
      qsa('.version-toggle', root).forEach((node) => {
        const groupKey = node.dataset.groupKey || qs('input', node)?.dataset.filterGroupKey;
        if (groupKey && !order.includes(groupKey)) {
          order.push(groupKey);
        }
      });
      return order;
    }

    function normalizeGroupOrder(root, order) {
      const merged = [];
      const seen = new Set();
      [...order, ...getCurrentGroupOrder(root)].forEach((groupKey) => {
        if (groupKey && !seen.has(groupKey)) {
          seen.add(groupKey);
          merged.push(groupKey);
        }
      });
      return merged;
    }

    function reorderGroupedChildren(parent, order) {
      const children = Array.from(parent.children);
      const fixedNodes = children.filter((node) => !node.dataset.groupKey);
      const groupedNodes = new Map();
      children
        .filter((node) => node.dataset.groupKey)
        .forEach((node) => {
          const groupKey = node.dataset.groupKey;
          if (!groupedNodes.has(groupKey)) groupedNodes.set(groupKey, []);
          groupedNodes.get(groupKey).push(node);
        });

      const fragment = document.createDocumentFragment();
      fixedNodes.forEach((node) => fragment.appendChild(node));
      order.forEach((groupKey) => {
        (groupedNodes.get(groupKey) || []).forEach((node) => fragment.appendChild(node));
      });
      parent.appendChild(fragment);
    }

    function reorderPairSelectOptions(root, order) {
      qsa('[data-pair-select]', root).forEach((select) => {
        const selected = select.value;
        const optionsByValue = new Map(Array.from(select.options).map((option) => [option.value, option]));
        order.forEach((groupKey) => {
          const option = optionsByValue.get(groupKey);
          if (option) select.appendChild(option);
        });
        select.value = selected;
      });
    }

    function applyGroupOrder(sectionKey, order) {
      const root = getSectionRootByKey(sectionKey);
      if (!root) return;
      const normalizedOrder = normalizeGroupOrder(root, order);

      qsa('.version-grid, .pairing-grid', root).forEach((container) => reorderGroupedChildren(container, normalizedOrder));
      qsa('table[data-filterable-table]', root).forEach((table) => {
        const colgroup = qs('colgroup', table);
        if (colgroup) reorderGroupedChildren(colgroup, normalizedOrder);
        qsa('thead tr, tbody tr', table).forEach((row) => reorderGroupedChildren(row, normalizedOrder));
      });
      reorderPairSelectOptions(root, normalizedOrder);

      try {
        localStorage.setItem(groupOrderStorageKey(sectionKey), JSON.stringify(normalizedOrder));
      } catch (error) {
        console.warn('save group order failed', error);
      }
    }

    function restoreGroupOrder(sectionKey) {
      let savedOrder = null;
      try {
        savedOrder = JSON.parse(localStorage.getItem(groupOrderStorageKey(sectionKey)) || 'null');
      } catch (error) {
        console.warn('load group order failed', error);
      }
      if (Array.isArray(savedOrder) && savedOrder.length) {
        applyGroupOrder(sectionKey, savedOrder);
      }
    }

    function restoreAllGroupOrders() {
      qsa('[data-dashboard-section-key]').forEach((root) => {
        const sectionKey = root.dataset.dashboardSectionKey || '';
        if (sectionKey) restoreGroupOrder(sectionKey);
      });
    }

    function bindVersionDragAndDrop() {
      let dragState = null;

      qsa('.version-toggle').forEach((node) => {
        const groupKey = node.dataset.groupKey || qs('input', node)?.dataset.filterGroupKey;
        const root = getSectionRoot(node);
        const sectionKey = root?.dataset.dashboardSectionKey || '';
        if (!groupKey) return;

        node.addEventListener('dragstart', (event) => {
          dragState = { groupKey, sectionKey };
          node.classList.add('dragging');
          if (event.dataTransfer) {
            event.dataTransfer.effectAllowed = 'move';
            event.dataTransfer.setData('text/plain', groupKey);
          }
        });

        node.addEventListener('dragend', () => {
          dragState = null;
          qsa('.version-toggle', root || document).forEach((item) => item.classList.remove('dragging', 'drag-over'));
        });

        node.addEventListener('dragover', (event) => {
          if (!dragState || dragState.sectionKey !== sectionKey || dragState.groupKey === groupKey) return;
          event.preventDefault();
          node.classList.add('drag-over');
        });

        node.addEventListener('dragleave', () => {
          node.classList.remove('drag-over');
        });

        node.addEventListener('drop', (event) => {
          if (!dragState || dragState.sectionKey !== sectionKey || dragState.groupKey === groupKey) return;
          event.preventDefault();
          node.classList.remove('drag-over');

          const order = getCurrentGroupOrder(root || document);
          const sourceIndex = order.indexOf(dragState.groupKey);
          const targetIndex = order.indexOf(groupKey);
          if (sourceIndex === -1 || targetIndex === -1) return;

          order.splice(sourceIndex, 1);
          const nextTargetIndex = order.indexOf(groupKey);
          order.splice(nextTargetIndex, 0, dragState.groupKey);
          applyGroupOrder(sectionKey, order);
          applyAllFilters();
        });
      });
    }

    function setVersionGroupChecked(sectionKey, groupKey, checked) {
      const root = getSectionRootByKey(sectionKey);
      if (!root) return;
      qsa('.version-toggle input', root).forEach((input) => {
        if (input.dataset.filterGroupKey === groupKey) {
          input.checked = checked;
        }
      });
    }

    function updateVersionVisibility(root) {
      if (!root) return;
      const checked = getVisibleGroupKeys(root);

      qsa('[data-group-key]', root).forEach((node) => {
        const shouldShow = checked.has(node.dataset.groupKey);
        node.classList.toggle('is-hidden', !shouldShow);
      });

      qsa('.version-toggle', root).forEach((label) => {
        const input = qs('input', label);
        label.classList.toggle('off', !input.checked);
      });

      updateHiddenVersionPanels(root, checked);
    }

    function updateKindVisibility(root) {
      if (!root) return;
      const visibleKinds = getVisibleKinds(root);
      const visibleGroupKeys = getVisibleGroupKeys(root);

      qsa('.kind-toggle', root).forEach((label) => {
        const input = qs('input', label);
        label.classList.toggle('off', !input.checked);
      });

      qsa('.pairing-item[data-kind]', root).forEach((node) => {
        const groupKey = node.dataset.groupKey || '';
        const kind = node.dataset.kind || '';
        const shouldShow = visibleGroupKeys.has(groupKey) && visibleKinds.has(kind);
        node.classList.toggle('is-hidden', !shouldShow);
      });

      qsa('table[data-filterable-table]', root).forEach((table) => {
        const visibleLeafCountByGroup = new Map();
        const visibleLeafCountByGroupKind = new Map();

        qsa('col.leaf-col, th.col-index-head, th.scope-head, td.value-cell', table).forEach((node) => {
          const groupKey = node.dataset.groupKey || '';
          const kind = node.dataset.kind || '';
          const shouldShow = visibleGroupKeys.has(groupKey) && visibleKinds.has(kind);
          node.classList.toggle('is-hidden', !shouldShow);

          if (!shouldShow || !node.matches('col.leaf-col')) return;
          visibleLeafCountByGroup.set(groupKey, (visibleLeafCountByGroup.get(groupKey) || 0) + 1);
          const groupKindKey = `${groupKey}::${kind}`;
          visibleLeafCountByGroupKind.set(groupKindKey, (visibleLeafCountByGroupKind.get(groupKindKey) || 0) + 1);
        });

        qsa('th.kind-head[data-group-key][data-kind]', table).forEach((node) => {
          const groupKey = node.dataset.groupKey || '';
          const kind = node.dataset.kind || '';
          const visibleLeafCount = visibleLeafCountByGroupKind.get(`${groupKey}::${kind}`) || 0;
          node.colSpan = Math.max(1, visibleLeafCount);
          node.classList.toggle('is-hidden', visibleLeafCount === 0);
        });

        qsa('th.group-head[data-group-key]', table).forEach((node) => {
          const groupKey = node.dataset.groupKey || '';
          const visibleLeafCount = visibleLeafCountByGroup.get(groupKey) || 0;
          node.colSpan = Math.max(1, visibleLeafCount);
          node.classList.toggle('is-hidden', visibleLeafCount === 0);
        });
      });
    }

    function updateColumnIndexHeads(root = document) {
      qsa('table[data-filterable-table]', root).forEach((table) => {
        const visibleHeads = qsa('th.col-index-head', table).filter((cell) => !cell.classList.contains('is-hidden'));
        visibleHeads.forEach((cell, index) => {
          cell.textContent = `第${index + 1}列`;
        });
      });
    }

    function updateHiddenVersionPanels(root, visibleGroupKeys) {
      if (!root) return;
      const checked = visibleGroupKeys || getVisibleGroupKeys(root);

      qsa('[data-hidden-version-panel]', root).forEach((panel) => {
        let hiddenCount = 0;
        qsa('[data-restore-item]', panel).forEach((item) => {
          const groupKey = item.dataset.restoreGroupKey || '';
          const shouldShow = !!groupKey && !checked.has(groupKey);
          item.classList.toggle('is-hidden', !shouldShow);
          const input = qs('input', item);
          if (input) input.checked = false;
          if (shouldShow) hiddenCount += 1;
        });
        panel.classList.toggle('is-hidden', hiddenCount === 0);
      });
    }

    function updateMetricVisibility() {
      const checked = new Set(
        qsa('.metric-toggle input:checked').map((input) => input.dataset.filterMetricKey)
      );

      qsa('[data-metric-key]').forEach((node) => {
        const shouldShow = checked.has(node.dataset.metricKey);
        if (node.matches('tr[data-task-key]')) {
          node.classList.toggle('metric-filter-hidden', !shouldShow);
          syncRowFilterVisibility(node);
        } else {
          node.classList.toggle('is-hidden', !shouldShow);
        }
      });

      qsa('.metric-toggle').forEach((label) => {
        const input = qs('input', label);
        label.classList.toggle('off', !input.checked);
      });
    }

    function updateHiddenTaskPanels(root, visibleTaskKeys) {
      if (!root) return;
      const checked = visibleTaskKeys || new Set(
        qsa('.task-toggle input:checked', root).map((input) => input.dataset.filterTaskKey)
      );

      qsa('[data-hidden-task-panel]', root).forEach((panel) => {
        let hiddenCount = 0;
        qsa('[data-restore-task-item]', panel).forEach((item) => {
          const taskKey = item.dataset.restoreTaskKey || '';
          const shouldShow = !!taskKey && !checked.has(taskKey);
          item.classList.toggle('is-hidden', !shouldShow);
          const input = qs('input', item);
          if (input) input.checked = false;
          if (shouldShow) hiddenCount += 1;
        });
        panel.classList.toggle('is-hidden', hiddenCount === 0);
      });
    }

    function syncRowFilterVisibility(row) {
      const hidden = row.classList.contains('task-filter-hidden') || row.classList.contains('metric-filter-hidden');
      row.classList.toggle('is-hidden', hidden);
    }

    function updateTaskVisibility() {
      qsa('[data-dashboard-section-key]').forEach((root) => {
        const sectionKey = root.dataset.dashboardSectionKey || '';
        if (!sectionKey) return;

        const checked = new Set(
          qsa('.task-toggle input:checked', root).map((input) => input.dataset.filterTaskKey)
        );

        qsa(`tr[data-task-key][data-section-key="${sectionKey}"]`, root).forEach((row) => {
          const shouldShow = checked.has(row.dataset.taskKey);
          row.classList.toggle('task-filter-hidden', !shouldShow);
          syncRowFilterVisibility(row);
        });

        qsa('.task-toggle', root).forEach((label) => {
          const input = qs('input', label);
          label.classList.toggle('off', !input.checked);
        });

        updateHiddenTaskPanels(root, checked);
      });
    }

    function updateSectionStates() {
      qsa('[data-table-container]').forEach((container) => {
        const hasVisibleGroup = !!qs('[data-group-key]:not(.is-hidden)', container);
        const hasVisibleRows = !!qs('tbody tr:not(.is-hidden)', container);
        const empty = qs('.filtered-empty', container);
        const table = qs('table', container);
        const shouldShowTable = hasVisibleGroup && hasVisibleRows;
        if (empty) empty.classList.toggle('is-hidden', shouldShowTable);
        if (table) table.classList.toggle('is-hidden', !shouldShowTable);
      });
    }

    function restorePairSelections() {
      qsa('[data-pair-select]').forEach((select) => {
        const sectionKey = select.dataset.sectionKey || '';
        const thinkGroupKey = select.dataset.thinkGroupKey || '';
        const defaultValue = select.dataset.defaultValue || select.value;
        select.value = defaultValue;
        try {
          const saved = localStorage.getItem(pairStorageKey(sectionKey, thinkGroupKey));
          if (saved && qsa('option', select).some((option) => option.value === saved)) {
            select.value = saved;
          }
        } catch (error) {
          console.warn('load pair selection failed', error);
        }
      });
    }

    function getPairMaps() {
      const pairMaps = {};
      qsa('[data-pair-select]').forEach((select) => {
        const sectionKey = select.dataset.sectionKey || '';
        const thinkGroupKey = select.dataset.thinkGroupKey || '';
        if (!sectionKey || !thinkGroupKey || !select.value) return;
        if (!pairMaps[sectionKey]) pairMaps[sectionKey] = {};
        pairMaps[sectionKey][thinkGroupKey] = select.value;
      });
      return pairMaps;
    }

    function resetSectionPairs(sectionKey) {
      qsa(`[data-pair-select][data-section-key="${sectionKey}"]`).forEach((select) => {
        const next = select.dataset.defaultValue || select.value;
        select.value = next;
        try {
          localStorage.removeItem(pairStorageKey(sectionKey, select.dataset.thinkGroupKey || ''));
        } catch (error) {
          console.warn('reset pair selection failed', error);
        }
      });
      applyAllFilters();
    }

    function percentile(values, ratio) {
      if (!values.length) return 0;
      const sorted = [...values].sort((left, right) => left - right);
      const index = Math.max(0, Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1));
      return sorted[index];
    }

    function parseCellValue(cell) {
      const raw = cell?.dataset.value;
      if (raw === undefined || raw === null || raw === '') return null;
      const value = Number(raw);
      return Number.isFinite(value) ? value : null;
    }

    function updateVisibleExtrema() {
      qsa('tr[data-section-key]').forEach((row) => {
        const cells = qsa('td.value-cell', row);
        cells.forEach((cell) => {
          cell.classList.remove('value-best', 'value-worst');
        });

        const visibleCells = cells.filter((cell) =>
          !cell.classList.contains('na') && !cell.classList.contains('is-hidden')
        );
        const numericCells = visibleCells
          .map((cell) => ({ cell, value: parseCellValue(cell) }))
          .filter((item) => item.value !== null);

        if (numericCells.length === 1) {
          numericCells[0].cell.classList.add('value-best');
          return;
        }
        if (!numericCells.length) return;

        const values = numericCells.map((item) => item.value);
        const maxValue = Math.max(...values);
        const minValue = Math.min(...values);
        if (Math.abs(maxValue - minValue) <= 1e-12) return;

        numericCells.forEach((item) => {
          if (Math.abs(item.value - maxValue) <= 1e-12) {
            item.cell.classList.add('value-best');
          }
          if (Math.abs(item.value - minValue) <= 1e-12) {
            item.cell.classList.add('value-worst');
          }
        });
      });
    }

    function ensureValueChipRow(cell) {
      let chipRow = qs('.value-chip-row', cell);
      if (!chipRow) {
        chipRow = document.createElement('div');
        chipRow.className = 'value-chip-row';
        cell.appendChild(chipRow);
      }
      return chipRow;
    }

    function formatMetricBadge(prefix, delta, relativeDelta) {
      const arrow = delta > 1e-12 ? '↑' : delta < -1e-12 ? '↓' : '≈';
      if (relativeDelta === null || relativeDelta === undefined) {
        return Math.abs(delta) <= 1e-12 ? `${prefix}≈0` : `${prefix}${arrow}new`;
      }
      const text = (Math.abs(relativeDelta) * 100).toFixed(1).replace(/0+$/, '').replace(/\\.$/, '');
      return `${prefix}${arrow}${text || '0'}%`;
    }

    function clearValueDiffState(cell) {
      cell.classList.remove(
        'has-delta',
        'delta-up',
        'delta-down',
        'delta-flat',
        'sig-up',
        'sig-down',
        'scope-up',
        'scope-down',
        'scope-flat'
      );
      const chipRow = ensureValueChipRow(cell);
      chipRow.innerHTML = '';
    }

    function appendMetricChip(cell, type, delta, relativeDelta, title) {
      const chipRow = ensureValueChipRow(cell);
      const chip = document.createElement('span');
      chip.className = type === 'scope' ? 'scope-chip' : 'delta-chip';
      chip.textContent = formatMetricBadge(type === 'scope' ? 'GS' : 'TN', delta, relativeDelta);
      chip.title = title;
      chipRow.appendChild(chip);
    }

    function findThinkBaselineCell(row, thinkCell, pairMaps) {
      const sectionKey = row.dataset.sectionKey || '';
      const sectionPairMap = pairMaps[sectionKey] || {};
      const baselineGroupKey = sectionPairMap[thinkCell.dataset.groupKey] || thinkCell.dataset.groupKey;
      const thinkScope = thinkCell.dataset.scope || '';
      return qsa('td.value-cell', row).find((cell) =>
        cell.dataset.groupKey === baselineGroupKey &&
        cell.dataset.kind === 'nothink' &&
        (cell.dataset.scope || '') === thinkScope &&
        !cell.classList.contains('na')
      ) || null;
    }

    function collectThinkDeltaInfo(thinkCell, pairMaps) {
      const row = thinkCell.closest('tr');
      if (!row) return null;
      const thinkValue = parseCellValue(thinkCell);
      if (thinkValue === null) return null;
      const baselineCell = findThinkBaselineCell(row, thinkCell, pairMaps);
      if (!baselineCell) return null;
      const baselineValue = parseCellValue(baselineCell);
      if (baselineValue === null) return null;

      const delta = thinkValue - baselineValue;
      const relativeDelta = Math.abs(baselineValue) <= 1e-12
        ? (Math.abs(delta) <= 1e-12 ? 0 : null)
        : delta / baselineValue;

      return {
        sectionKey: row.dataset.sectionKey || '',
        sectionTitle: row.dataset.sectionTitle || '',
        rowLabel: row.dataset.rowLabel || '',
        taskKey: row.dataset.taskKey || '',
        metricKey: row.dataset.metricKey || '',
        groupKey: thinkCell.dataset.groupKey || '',
        groupTitle: thinkCell.dataset.groupTitle || '',
        groupMeta: thinkCell.dataset.groupMeta || '',
        scope: thinkCell.dataset.scope || '',
        delta,
        relativeDelta,
        thinkValue,
        baselineValue,
        baselineGroupTitle: baselineCell.dataset.groupTitle || '',
        baselineGroupMeta: baselineCell.dataset.groupMeta || '',
      };
    }

    function collectScopeDeltaInfo(cell) {
      if ((cell.dataset.scope || '') !== '全域') return null;
      const row = cell.closest('tr');
      if (!row) return null;
      const globalValue = parseCellValue(cell);
      if (globalValue === null) return null;

      const singleCell = qsa('td.value-cell', row).find((candidate) =>
        candidate.dataset.groupKey === cell.dataset.groupKey &&
        candidate.dataset.kind === cell.dataset.kind &&
        (candidate.dataset.scope || '') === '单域' &&
        !candidate.classList.contains('na')
      ) || null;
      if (!singleCell) return null;

      const singleValue = parseCellValue(singleCell);
      if (singleValue === null) return null;

      const delta = globalValue - singleValue;
      const relativeDelta = Math.abs(singleValue) <= 1e-12
        ? (Math.abs(delta) <= 1e-12 ? 0 : null)
        : delta / singleValue;

      return {
        delta,
        relativeDelta,
        globalValue,
        singleValue,
      };
    }

    function updateValueCell(cell, thinkInfo, thinkThreshold, scopeInfo) {
      clearValueDiffState(cell);

      if (thinkInfo) {
        cell.classList.add('has-delta');
        if (thinkInfo.delta > 1e-12) {
          cell.classList.add('delta-up');
        } else if (thinkInfo.delta < -1e-12) {
          cell.classList.add('delta-down');
        } else {
          cell.classList.add('delta-flat');
        }

        if (thinkThreshold > 0 && Math.abs(thinkInfo.delta) >= thinkThreshold) {
          if (thinkInfo.delta > 1e-12) cell.classList.add('sig-up');
          if (thinkInfo.delta < -1e-12) cell.classList.add('sig-down');
        }

        appendMetricChip(
          cell,
          'delta',
          thinkInfo.delta,
          thinkInfo.relativeDelta,
          `TN：think - nothink = ${formatSigned(thinkInfo.delta)}；相对当前基线 = ${formatSignedPercent(thinkInfo.relativeDelta)}；基线 = ${thinkInfo.baselineGroupTitle || '-'}${thinkInfo.baselineGroupMeta ? ` · ${thinkInfo.baselineGroupMeta}` : ''}`
        );
      }

      if (!scopeInfo) {
        return;
      }

      if (scopeInfo.delta > 1e-12) {
        cell.classList.add('scope-up');
      } else if (scopeInfo.delta < -1e-12) {
        cell.classList.add('scope-down');
      } else {
        cell.classList.add('scope-flat');
      }

      appendMetricChip(
        cell,
        'scope',
        scopeInfo.delta,
        scopeInfo.relativeDelta,
        `GS：全域 - 单域 = ${formatSigned(scopeInfo.delta)}；相对单域 = ${formatSignedPercent(scopeInfo.relativeDelta)}`
      );
    }

    function renderHighlightRows(bodyKey, items) {
      const body = qs(`[data-highlight-body="${bodyKey}"]`);
      if (!body) return;
      body.innerHTML = items.map((item) => `
        <tr class="highlight-row" data-group-key="${escapeHtml(item.groupKey)}" data-metric-key="${escapeHtml(item.metricKey)}">
          <td>${escapeHtml(item.sectionKey)}</td>
          <td>${escapeHtml(item.rowLabel)}</td>
          <td>${escapeHtml(item.groupTitle)}<br><span class="subtle">${escapeHtml(item.groupMeta)}</span></td>
          <td>${escapeHtml(item.scope || '-')}</td>
          <td class="${item.delta >= 0 ? 'delta-pos' : 'delta-neg'}">${escapeHtml(formatSigned(item.delta))}</td>
        </tr>
      `).join('');
    }

    function refreshComputedDeltas() {
      const pairMaps = getPairMaps();
      const highlightRoot = getSectionRootByKey('summary-gap');
      const visibleGroupKeys = getVisibleGroupKeys(highlightRoot || document);
      const visibleMetricKeys = new Set(
        qsa('.metric-toggle input:checked').map((input) => input.dataset.filterMetricKey)
      );
      const visibleKindsBySection = {};
      qsa('[data-kind-panel]').forEach((panel) => {
        const sectionKey = panel.dataset.sectionKey || '';
        if (sectionKey) visibleKindsBySection[sectionKey] = new Set();
      });
      qsa('.kind-toggle input:checked').forEach((input) => {
        const sectionKey = input.dataset.sectionKey || '';
        const kind = input.dataset.filterKind || '';
        if (!sectionKey || !kind) return;
        if (!visibleKindsBySection[sectionKey]) visibleKindsBySection[sectionKey] = new Set();
        visibleKindsBySection[sectionKey].add(kind);
      });
      const visibleTaskKeysBySection = {};
      qsa('.task-toggle input:checked').forEach((input) => {
        const sectionKey = input.dataset.sectionKey || '';
        const taskKey = input.dataset.filterTaskKey || '';
        if (!sectionKey || !taskKey) return;
        if (!visibleTaskKeysBySection[sectionKey]) visibleTaskKeysBySection[sectionKey] = new Set();
        visibleTaskKeysBySection[sectionKey].add(taskKey);
      });
      const sectionAbsDeltas = {};
      const allInfos = [];

      qsa('td.value-cell[data-kind="think"]').forEach((cell) => {
        const info = collectThinkDeltaInfo(cell, pairMaps);
        cell.__thinkDeltaInfo = info;
        if (info) {
          if (!sectionAbsDeltas[info.sectionKey]) sectionAbsDeltas[info.sectionKey] = [];
          sectionAbsDeltas[info.sectionKey].push(Math.abs(info.delta));
          allInfos.push(info);
        }
      });

      const thresholdBySection = {};
      Object.entries(sectionAbsDeltas).forEach(([sectionKey, values]) => {
        thresholdBySection[sectionKey] = percentile(values, 0.8);
      });

      qsa('td.value-cell:not(.na)').forEach((cell) => {
        const thinkInfo = cell.dataset.kind === 'think' ? (cell.__thinkDeltaInfo || null) : null;
        const thinkThreshold = thinkInfo ? (thresholdBySection[thinkInfo.sectionKey] || 0) : 0;
        const scopeInfo = collectScopeDeltaInfo(cell);
        updateValueCell(cell, thinkInfo, thinkThreshold, scopeInfo);
      });

      const highlightCandidates = allInfos.filter((item) =>
        visibleGroupKeys.has(item.groupKey)
        && !!visibleKindsBySection[item.sectionKey]
        && visibleKindsBySection[item.sectionKey].has('think')
        && visibleMetricKeys.has(item.metricKey)
        && !!visibleTaskKeysBySection[item.sectionKey]
        && visibleTaskKeysBySection[item.sectionKey].has(item.taskKey)
      );

      const positives = highlightCandidates
        .filter((item) => item.delta > 0)
        .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta))
        .slice(0, 8);
      const negatives = highlightCandidates
        .filter((item) => item.delta < 0)
        .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta))
        .slice(0, 8);

      renderHighlightRows('positive', positives);
      renderHighlightRows('negative', negatives);
    }

    function applyAllFilters() {
      qsa('[data-dashboard-section-key]').forEach((root) => updateVersionVisibility(root));
      qsa('[data-dashboard-section-key]').forEach((root) => updateKindVisibility(root));
      updateColumnIndexHeads();
      updateTaskVisibility();
      updateMetricVisibility();
      refreshComputedDeltas();
      updateVisibleExtrema();
      updateSectionStates();
    }

    function setAllVersions(sectionKey, checked) {
      const root = getSectionRootByKey(sectionKey);
      if (!root) return;
      qsa('.version-toggle input', root).forEach((input) => {
        input.checked = checked;
      });
      applyAllFilters();
    }

    function setSectionMetrics(sectionKey, checked) {
      qsa(`.metric-toggle input[data-section-key="${sectionKey}"]`).forEach((input) => {
        input.checked = checked;
      });
      applyAllFilters();
    }

    function saveSectionKinds(sectionKey) {
      const root = getSectionRootByKey(sectionKey);
      if (!root) return;
      const visibleKinds = qsa('.kind-toggle input:checked', root).map((input) => input.dataset.filterKind);
      try {
        localStorage.setItem(kindStorageKey(sectionKey), JSON.stringify(visibleKinds));
      } catch (error) {
        console.warn('save kind selection failed', error);
      }
    }

    function setSectionKinds(sectionKey, visibleKinds) {
      const allowedKinds = new Set(Array.isArray(visibleKinds) ? visibleKinds : []);
      qsa(`.kind-toggle input[data-section-key="${sectionKey}"]`).forEach((input) => {
        input.checked = allowedKinds.has(input.dataset.filterKind);
      });
      saveSectionKinds(sectionKey);
      applyAllFilters();
    }

    function setTaskChecked(sectionKey, taskKey, checked) {
      qsa(`.task-toggle input[data-section-key="${sectionKey}"]`).forEach((input) => {
        if (input.dataset.filterTaskKey === taskKey) {
          input.checked = checked;
        }
      });
    }

    function setSectionTasks(sectionKey, checked) {
      qsa(`.task-toggle input[data-section-key="${sectionKey}"]`).forEach((input) => {
        input.checked = checked;
      });
      applyAllFilters();
    }

    function restoreAllHiddenTasks(sectionKey) {
      setSectionTasks(sectionKey, true);
    }

    function restoreAllHiddenVersions(sectionKey) {
      setAllVersions(sectionKey, true);
    }

    function restoreKindSelections() {
      qsa('[data-kind-panel]').forEach((panel) => {
        const sectionKey = panel.dataset.sectionKey || '';
        if (!sectionKey) return;
        let savedKinds = null;
        try {
          savedKinds = JSON.parse(localStorage.getItem(kindStorageKey(sectionKey)) || 'null');
        } catch (error) {
          console.warn('load kind selection failed', error);
        }
        if (!Array.isArray(savedKinds)) return;
        const allowedKinds = new Set(savedKinds);
        qsa(`.kind-toggle input[data-section-key="${sectionKey}"]`).forEach((input) => {
          input.checked = allowedKinds.has(input.dataset.filterKind);
        });
      });
    }

    function normalizeCellText(cell) {
      const valueMain = qs('.value-main', cell);
      if (valueMain) {
        return valueMain.textContent.trim();
      }
      return cell.textContent.replace(/\\s+/g, ' ').trim();
    }

    function cloneVisibleTable(table) {
      const clone = table.cloneNode(true);
      qsa('.is-hidden', clone).forEach((node) => node.remove());
      qsa('.value-chip-row', clone).forEach((node) => node.remove());
      qsa('td.value-cell', clone).forEach((cell) => {
        const main = qs('.value-main', cell);
        if (main) {
          cell.textContent = main.textContent.trim();
        }
      });
      return clone;
    }

    function collectTableMatrix(table) {
      const rows = qsa('tr', table);
      const grid = [];

      rows.forEach((row, rowIndex) => {
        if (row.classList.contains('is-hidden')) return;
        if (!grid[rowIndex]) grid[rowIndex] = [];
        let colIndex = 0;

        Array.from(row.children).forEach((cell) => {
          if (cell.classList.contains('is-hidden')) return;
          while (grid[rowIndex][colIndex] !== undefined) {
            colIndex += 1;
          }
          const rowspan = parseInt(cell.getAttribute('rowspan') || '1', 10);
          const colspan = parseInt(cell.getAttribute('colspan') || '1', 10);
          const text = normalizeCellText(cell);

          for (let rowOffset = 0; rowOffset < rowspan; rowOffset += 1) {
            if (!grid[rowIndex + rowOffset]) grid[rowIndex + rowOffset] = [];
            for (let colOffset = 0; colOffset < colspan; colOffset += 1) {
              grid[rowIndex + rowOffset][colIndex + colOffset] =
                rowOffset === 0 && colOffset === 0 ? text : '';
            }
          }

          colIndex += colspan;
        });
      });

      return grid
        .filter((row) => row && row.some((cell) => cell !== undefined))
        .map((row) => row.map((cell) => (cell === undefined ? '' : cell)));
    }

    function matrixToTsv(matrix) {
      return matrix
        .map((row) => row.map((cell) => String(cell).replace(/\\t/g, ' ')).join('\\t'))
        .join('\\n');
    }

    const HTML2CANVAS_CDN_URLS = [
      'https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js',
      'https://unpkg.com/html2canvas@1.4.1/dist/html2canvas.min.js',
    ];
    let html2canvasLoader = null;

    function downloadBlob(blob, filename) {
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.download = filename;
      link.href = url;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function loadExternalScript(url) {
      return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = url;
        script.async = true;
        script.onload = resolve;
        script.onerror = () => reject(new Error(`script_load_failed:${url}`));
        document.head.appendChild(script);
      });
    }

    async function ensureHtml2Canvas() {
      if (window.html2canvas) return window.html2canvas;
      if (!html2canvasLoader) {
        html2canvasLoader = (async () => {
          let lastError = null;
          for (const url of HTML2CANVAS_CDN_URLS) {
            try {
              await loadExternalScript(url);
              if (window.html2canvas) return window.html2canvas;
            } catch (error) {
              lastError = error;
              console.warn('load html2canvas failed', error);
            }
          }
          throw lastError || new Error('html2canvas_unavailable');
        })().catch((error) => {
          html2canvasLoader = null;
          throw error;
        });
      }
      return html2canvasLoader;
    }

    function computeExportScale(width, height, preferredScale = 2) {
      const safeWidth = Math.max(1, Math.ceil(width || 0));
      const safeHeight = Math.max(1, Math.ceil(height || 0));
      const maxPixels = 42_000_000;
      const currentPixels = safeWidth * safeHeight * preferredScale * preferredScale;
      if (currentPixels <= maxPixels) return preferredScale;
      const ratio = Math.sqrt(maxPixels / (safeWidth * safeHeight));
      if (!Number.isFinite(ratio)) return 1;
      return Math.max(1, Math.min(preferredScale, Math.floor(ratio * 100) / 100));
    }

    function prepareExportClone(root) {
      qsa('.is-hidden', root).forEach((node) => node.remove());
      qsa('.table-actions', root).forEach((node) => node.remove());
      qsa('button', root).forEach((node) => node.remove());
      qsa('.table-shell, .mini-table-shell', root).forEach((node) => {
        node.style.overflow = 'visible';
        node.style.width = 'max-content';
        node.style.maxWidth = 'none';
      });
      qsa('table', root).forEach((node) => {
        node.style.width = 'auto';
        node.style.minWidth = '0';
      });
      qsa('.row-head, thead th', root).forEach((node) => {
        node.style.position = 'static';
        node.style.left = 'auto';
        node.style.top = 'auto';
      });
    }

    function buildExportSvgText(sectionKey, clone, width, height) {
      const styleText = qs('#dashboard-style')?.textContent || '';
      const scopeNode = qs(`[data-width-scope="${sectionKey}"]`) || document.documentElement;
      const scopeStyle = getComputedStyle(scopeNode);
      const exportLeafWidth = scopeStyle.getPropertyValue('--leaf-col-width').trim() || '122px';
      const exportLabelWidth = scopeStyle.getPropertyValue('--label-col-width').trim() || '150px';
      const exportStickyCol1Width = scopeStyle.getPropertyValue('--sticky-col-1-width').trim() || '170px';
      const exportVersionWidth = scopeStyle.getPropertyValue('--version-toggle-width').trim() || '154px';
      const exportPageWidth = scopeStyle.getPropertyValue('--page-max-width').trim() || '2600px';
      const exportExtraStyle = `
        :root {
          --leaf-col-width: ${exportLeafWidth};
          --label-col-width: ${exportLabelWidth};
          --sticky-col-1-width: ${exportStickyCol1Width};
          --version-toggle-width: ${exportVersionWidth};
          --page-max-width: ${exportPageWidth};
        }
        .export-root { width: max-content; background: #ffffff; }
        .export-root .table-shell, .export-root .mini-table-shell { overflow: visible !important; width: max-content !important; }
        .export-root table { width: auto !important; min-width: 0 !important; }
        .export-root .row-head, .export-root thead th { position: static !important; left: auto !important; top: auto !important; }
      `;

      return `
        <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
          <foreignObject width="100%" height="100%">
            <div xmlns="http://www.w3.org/1999/xhtml">
              <style>${styleText}${exportExtraStyle}</style>
              ${clone.outerHTML}
            </div>
          </foreignObject>
        </svg>
      `;
    }

    function downloadCanvas(canvas, filename) {
      const link = document.createElement('a');
      link.download = filename;
      link.href = canvas.toDataURL('image/png');
      link.click();
    }

    async function exportSectionPngWithHtml2Canvas(sectionKey, clone, width, height) {
      const renderer = await ensureHtml2Canvas();
      const scale = computeExportScale(width, height, 2);
      const canvas = await renderer(clone, {
        backgroundColor: '#ffffff',
        scale,
        useCORS: true,
        logging: false,
        width,
        height,
        windowWidth: width,
        windowHeight: height,
        scrollX: 0,
        scrollY: 0,
      });
      downloadCanvas(canvas, `${sectionKey}.png`);
      return scale;
    }

    async function exportSectionPngWithSvg(sectionKey, clone, width, height) {
      const svgText = buildExportSvgText(sectionKey, clone, width, height);
      const blob = new Blob([svgText], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const image = new Image();
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
        image.src = url;
      });

      const scale = computeExportScale(width, height, 2);
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.ceil(width * scale));
      canvas.height = Math.max(1, Math.ceil(height * scale));
      const context = canvas.getContext('2d');
      if (!context) {
        URL.revokeObjectURL(url);
        throw new Error('canvas_context_unavailable');
      }
      context.scale(scale, scale);
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, width, height);
      context.drawImage(image, 0, 0);
      downloadCanvas(canvas, `${sectionKey}.png`);
      URL.revokeObjectURL(url);
      return scale;
    }

    async function copySectionTable(sectionKey) {
      const table = qs(`#table-${sectionKey} table`);
      if (!table) return;

      const visibleClone = cloneVisibleTable(table);
      const matrix = collectTableMatrix(visibleClone);
      const plain = matrixToTsv(matrix);
      const html = visibleClone.outerHTML;

      try {
        if (navigator.clipboard && window.ClipboardItem) {
          await navigator.clipboard.write([
            new ClipboardItem({
              'text/plain': new Blob([plain], { type: 'text/plain' }),
              'text/html': new Blob([html], { type: 'text/html' }),
            }),
          ]);
        } else if (navigator.clipboard) {
          await navigator.clipboard.writeText(plain);
        } else {
          throw new Error('clipboard_unavailable');
        }
        alert(`已复制 ${sectionKey} 当前可见表格，可直接粘贴到 Excel。`);
      } catch (error) {
        console.error(error);
        alert('复制失败：当前浏览器不支持直接写入剪贴板。');
      }
    }

    async function exportSectionPng(sectionKey) {
      const source = qs(`#export-${sectionKey}`);
      if (!source) return;

      const sandbox = document.createElement('div');
      sandbox.style.position = 'fixed';
      sandbox.style.left = '-100000px';
      sandbox.style.top = '0';
      sandbox.style.background = '#ffffff';
      sandbox.style.padding = '0';
      sandbox.style.zIndex = '-1';

      const clone = source.cloneNode(true);
      clone.classList.add('export-root');
      prepareExportClone(clone);
      sandbox.appendChild(clone);
      document.body.appendChild(sandbox);

      const width = Math.ceil(clone.scrollWidth);
      const height = Math.ceil(clone.scrollHeight);

      try {
        try {
          await exportSectionPngWithHtml2Canvas(sectionKey, clone, width, height);
        } catch (primaryError) {
          console.warn('html2canvas export failed, fallback to svg renderer', primaryError);
          await exportSectionPngWithSvg(sectionKey, clone, width, height);
        }
      } catch (error) {
        console.error(error);
        try {
          const svgText = buildExportSvgText(sectionKey, clone, width, height);
          downloadBlob(new Blob([svgText], { type: 'image/svg+xml;charset=utf-8' }), `${sectionKey}.svg`);
          alert(`导出 PNG 失败，已自动下载 SVG 兜底文件：${sectionKey}.svg；错误：${error?.message || error}`);
        } catch (svgError) {
          console.error(svgError);
          alert(`导出 PNG 失败：${error?.message || error}。如果你愿意，我可以继续把导出改成纯本地内嵌依赖方案。`);
        }
      } finally {
        sandbox.remove();
      }
    }

    function clampTableWidth(value) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return 122;
      return Math.max(96, Math.min(240, Math.round(numeric)));
    }

    function clampLabelWidth(value) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return 168;
      return Math.max(120, Math.min(420, Math.round(numeric)));
    }

    function applyTableWidth(sectionKey, value) {
      const next = clampTableWidth(value);
      qsa(`[data-width-scope="${sectionKey}"]`).forEach((node) => {
        node.style.setProperty('--leaf-col-width', `${next}px`);
        node.style.setProperty('--version-toggle-width', `${Math.max(154, next + 32)}px`);
      });
      qsa(`[data-table-width-range][data-section-key="${sectionKey}"]`).forEach((input) => {
        input.value = String(next);
      });
      qsa(`[data-table-width-number][data-section-key="${sectionKey}"]`).forEach((input) => {
        input.value = String(next);
      });
      qsa(`[data-table-width-value][data-section-key="${sectionKey}"]`).forEach((node) => {
        node.textContent = `${next}px`;
      });
      try {
        localStorage.setItem(widthStorageKey(sectionKey), String(next));
      } catch (error) {
        console.warn('save table width failed', error);
      }
    }

    function applyLabelWidth(sectionKey, value) {
      const next = clampLabelWidth(value);
      qsa(`[data-width-scope="${sectionKey}"]`).forEach((node) => {
        node.style.setProperty('--label-col-width', `${next}px`);
      });
      qsa(`[data-label-width-range][data-section-key="${sectionKey}"]`).forEach((input) => {
        input.value = String(next);
      });
      qsa(`[data-label-width-number][data-section-key="${sectionKey}"]`).forEach((input) => {
        input.value = String(next);
      });
      qsa(`[data-label-width-value][data-section-key="${sectionKey}"]`).forEach((node) => {
        node.textContent = `${next}px`;
      });
      try {
        localStorage.setItem(labelWidthStorageKey(sectionKey), String(next));
      } catch (error) {
        console.warn('save label width failed', error);
      }
    }

    function restoreTableWidths() {
      qsa('[data-width-scope]').forEach((node) => {
        const sectionKey = node.dataset.widthScope;
        if (!sectionKey) return;
        let savedTableWidth = null;
        let savedLabelWidth = null;
        try {
          savedTableWidth = localStorage.getItem(widthStorageKey(sectionKey));
          savedLabelWidth = localStorage.getItem(labelWidthStorageKey(sectionKey));
        } catch (error) {
          console.warn('load table width failed', error);
        }
        applyTableWidth(sectionKey, savedTableWidth || 122);
        applyLabelWidth(sectionKey, savedLabelWidth || 168);
      });
    }

    function bindDashboardActions() {
      bindVersionDragAndDrop();

      qsa('.version-toggle input').forEach((input) => {
        input.addEventListener('change', () => {
          const sectionKey = getSectionRoot(input)?.dataset.dashboardSectionKey || '';
          setVersionGroupChecked(sectionKey, input.dataset.filterGroupKey, input.checked);
          applyAllFilters();
        });
      });

      qsa('.metric-toggle input').forEach((input) => {
        input.addEventListener('change', applyAllFilters);
      });

      qsa('.kind-toggle input').forEach((input) => {
        input.addEventListener('change', () => {
          saveSectionKinds(input.dataset.sectionKey || '');
          applyAllFilters();
        });
      });

      qsa('.task-toggle input').forEach((input) => {
        input.addEventListener('change', applyAllFilters);
      });

      qsa('[data-action="select-all"]').forEach((button) => {
        button.addEventListener('click', () => setAllVersions(button.dataset.sectionKey || '', true));
      });

      qsa('[data-action="clear-all"]').forEach((button) => {
        button.addEventListener('click', () => setAllVersions(button.dataset.sectionKey || '', false));
      });

      qsa('[data-action="restore-hidden"]').forEach((button) => {
        button.addEventListener('click', () => restoreAllHiddenVersions(button.dataset.sectionKey || ''));
      });

      qsa('[data-action="restore-group"]').forEach((input) => {
        input.addEventListener('change', () => {
          if (!input.checked) return;
          setVersionGroupChecked(input.dataset.sectionKey || '', input.dataset.restoreGroupKey, true);
          input.checked = false;
          applyAllFilters();
        });
      });

      qsa('[data-action="select-all-metrics"]').forEach((button) => {
        button.addEventListener('click', () => setSectionMetrics(button.dataset.sectionKey, true));
      });

      qsa('[data-action="clear-metrics"]').forEach((button) => {
        button.addEventListener('click', () => setSectionMetrics(button.dataset.sectionKey, false));
      });

      qsa('[data-action="select-all-kinds"]').forEach((button) => {
        button.addEventListener('click', () => setSectionKinds(button.dataset.sectionKey || '', ['nothink', 'think']));
      });

      qsa('[data-action="show-only-nothink"]').forEach((button) => {
        button.addEventListener('click', () => setSectionKinds(button.dataset.sectionKey || '', ['nothink']));
      });

      qsa('[data-action="show-only-think"]').forEach((button) => {
        button.addEventListener('click', () => setSectionKinds(button.dataset.sectionKey || '', ['think']));
      });

      qsa('[data-action="clear-kinds"]').forEach((button) => {
        button.addEventListener('click', () => setSectionKinds(button.dataset.sectionKey || '', []));
      });

      qsa('[data-action="select-all-tasks"]').forEach((button) => {
        button.addEventListener('click', () => setSectionTasks(button.dataset.sectionKey || '', true));
      });

      qsa('[data-action="clear-tasks"]').forEach((button) => {
        button.addEventListener('click', () => setSectionTasks(button.dataset.sectionKey || '', false));
      });

      qsa('[data-action="restore-hidden-tasks"]').forEach((button) => {
        button.addEventListener('click', () => restoreAllHiddenTasks(button.dataset.sectionKey || ''));
      });

      qsa('[data-action="restore-task"]').forEach((input) => {
        input.addEventListener('change', () => {
          if (!input.checked) return;
          setTaskChecked(input.dataset.sectionKey || '', input.dataset.restoreTaskKey || '', true);
          input.checked = false;
          applyAllFilters();
        });
      });

      qsa('[data-action="copy-table"]').forEach((button) => {
        button.addEventListener('click', () => copySectionTable(button.dataset.sectionKey));
      });

      qsa('[data-action="export-png"]').forEach((button) => {
        button.addEventListener('click', () => exportSectionPng(button.dataset.sectionKey));
      });

      qsa('[data-width-preset]').forEach((button) => {
        button.addEventListener('click', () => applyTableWidth(button.dataset.sectionKey, button.dataset.widthPreset));
      });

      qsa('[data-table-width-range]').forEach((input) => {
        input.addEventListener('input', () => applyTableWidth(input.dataset.sectionKey, input.value));
      });

      qsa('[data-table-width-number]').forEach((input) => {
        input.addEventListener('input', () => applyTableWidth(input.dataset.sectionKey, input.value));
      });

      qsa('[data-label-width-range]').forEach((input) => {
        input.addEventListener('input', () => applyLabelWidth(input.dataset.sectionKey, input.value));
      });

      qsa('[data-label-width-number]').forEach((input) => {
        input.addEventListener('input', () => applyLabelWidth(input.dataset.sectionKey, input.value));
      });

      qsa('[data-pair-select]').forEach((select) => {
        select.addEventListener('change', () => {
          const sectionKey = select.dataset.sectionKey || '';
          const thinkGroupKey = select.dataset.thinkGroupKey || '';
          try {
            localStorage.setItem(pairStorageKey(sectionKey, thinkGroupKey), select.value);
          } catch (error) {
            console.warn('save pair selection failed', error);
          }
          applyAllFilters();
        });
      });

      qsa('[data-action="reset-section-pairs"]').forEach((button) => {
        button.addEventListener('click', () => {
          resetSectionPairs(button.dataset.sectionKey || '');
        });
      });

      restoreAllGroupOrders();
      restorePairSelections();
      restoreKindSelections();
      restoreTableWidths();
      applyAllFilters();
    }

    bindDashboardActions();
  </script>
</body>
</html>
"""


def clean_text(value: str) -> str:
    return " ".join((value or "").replace("\ufeff", "").strip().split())


def parse_bool(text: str) -> bool:
    return clean_text(text).lower() in {"1", "true", "yes", "y"}


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


def find_result_dir(eval_root: Path, row: dict[str, str]) -> Path:
    candidates = [
        eval_root / clean_text(row["archive_dir"]),
        eval_root / clean_text(row["actual_source_dir"]),
        eval_root / clean_text(row["source_dir"]),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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


def parse_manifest(eval_root: Path, manifest_path: Path) -> list[ModelSpec]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在：{manifest_path}")

    models: list[ModelSpec] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            version = clean_text(row["version"])
            step = clean_text(row["step"])
            epoch = clean_text(row["epoch"])
            kind = "think" if parse_bool(row["think_enabled"]) else "nothink"
            archive_dir = clean_text(row["archive_dir"])
            result_dir = find_result_dir(eval_root, row)
            group_key = f"{version}|{step}|{epoch}"
            models.append(
                ModelSpec(
                    key=build_model_key(archive_dir, kind),
                    source_dir=clean_text(row["source_dir"]),
                    actual_source_dir=clean_text(row["actual_source_dir"]),
                    archive_dir=archive_dir,
                    version=version,
                    step=step,
                    epoch=epoch,
                    think_enabled=parse_bool(row["think_enabled"]),
                    kind=kind,
                    group_key=group_key,
                    group_title=build_group_title(version),
                    group_meta=build_group_meta(epoch, step),
                    result_dir=result_dir,
                    converted=read_converted(result_dir),
                )
            )

    if not models:
        raise ValueError(f"manifest 里没有可用模型：{manifest_path}")
    return models


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


def build_leaf_columns(models: list[ModelSpec], scope_headers: tuple[str, ...]) -> list[LeafColumn]:
    leaves: list[LeafColumn] = []
    if scope_headers:
        scope_order = {SINGLE_SCOPE: 0, GLOBAL_SCOPE: 1}
        scopes = tuple(sorted(scope_headers, key=lambda scope: scope_order.get(scope, 99)))
    else:
        scopes = (None,)
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
            f'<th class="col-index-head" data-group-key="{escape(leaf.group_key)}" data-kind="{escape(leaf.kind)}" data-col-index-head="{index}">第{index}列</th>'
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
                f'<th class="scope-head" data-group-key="{escape(leaf.group_key)}" data-kind="{escape(leaf.kind)}">{escape(leaf.scope or "")}</th>'
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
            f'<col class="leaf-col" data-group-key="{escape(leaf.group_key)}" data-kind="{escape(leaf.kind)}" />'
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
        f'<input type="range" min="120" max="420" step="4" value="168" data-label-width-range data-section-key="{escape(section_key)}" />'
        f'<input type="number" min="120" max="420" step="4" value="168" data-label-width-number data-section-key="{escape(section_key)}" />'
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
        '<div class="subtle">Think/Nothink 对照只作用于当前分表的 `TN` badge；`GS` badge 固定表示“全域相对单域”。</div>'
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
    missing_dirs = sum(1 for model in models if not model.result_dir.exists())

    best_up = max(all_highlights, key=lambda item: item.delta, default=None)
    worst_down = min(all_highlights, key=lambda item: item.delta, default=None)

    return [
        {
            "title": "模型目录",
            "main": str(len(models)),
            "sub": "manifest 顺序保留原始列顺序",
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
        {
            "title": "缺失目录",
            "main": str(missing_dirs),
            "sub": "manifest 中但本地目录不存在的数量",
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


def render_dashboard(
    title: str,
    eval_root: Path,
    manifest_path: Path,
    models: list[ModelSpec],
    sections: list[SectionData],
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

    body_html = (
        '<section class="card hero">'
        f'<h1>📊 {escape(title)}</h1>'
        '<p class="subtle">底色深浅表示同一行里的绝对值高低；绿色/红色描边表示 think 相对 nothink 的显著提升或下降。带 scope 的表会同时展示 `TN` 与 `GS` 两种差值 badge。</p>'
        '<div class="chips">'
        f'<span class="chip">生成时间：{escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</span>'
        f'<span class="chip">原始目录：{escape(str(eval_root))}</span>'
        f'<span class="chip">manifest：{escape(str(manifest_path))}</span>'
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
        +
        '<section class="card" data-dashboard-section-key="summary-gap">'
        '<h2>显著 gap 速览（TN）</h2>'
        '<p class="subtle">这里先按各分层内 think - nothink 的绝对差值排序；`GS`（全域相对单域）会直接展示在表格单元格里。这里的“显著”仅用于看板高亮，不代表统计显著性检验。版本显隐和各分层任务 / Metric 勾选都会联动到这里。</p>'
        + render_version_controls(models, section_key="summary-gap", title="gap 速览版本显隐")
        +
        '<div class="two-col">'
        f'<div><h3>提升 Top 8</h3>{render_highlight_table(top_up, positive=True)}</div>'
        f'<div><h3>下降 Top 8</h3>{render_highlight_table(top_down, positive=False)}</div>'
        "</div>"
        '<div class="note">如果某个版本没有对应任务，表格会保留列但显示为 - ，这样可以直接看出“没评”而不是误以为 0。</div>'
        "</section>"
        + section_html
    )

    html = HTML_TEMPLATE.replace("__TITLE__", escape(title))
    html = html.replace("__BODY__", body_html)
    return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 eval_results 原始目录生成静态 HTML 推荐评测看板。")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("../eval_results"),
        help="原始评测目录，默认 ../eval_results",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="结果目录 manifest，默认 <eval-root>/result_dir_manifest.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rec_dashboard.html"),
        help="输出 HTML 路径，默认当前目录 rec_dashboard.html",
    )
    parser.add_argument(
        "--title",
        default="推荐评测对比看板",
        help="输出 HTML 标题",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else (eval_root / "result_dir_manifest.tsv").resolve()

    models = parse_manifest(eval_root, manifest_path)
    sections = [build_section_data(spec, models) for spec in SECTION_SPECS]
    html = render_dashboard(args.title, eval_root, manifest_path, models, sections)

    args.output.write_text(html, encoding="utf-8")

    print(f"[OK] eval_root={eval_root}")
    print(f"[OK] manifest={manifest_path}")
    print(f"[OK] output={args.output.resolve()}")
    print(f"[OK] models={len(models)} sections={len(sections)}")
    print(
        "[OK] cells="
        f"{sum(section.present_cells for section in sections)}/{sum(section.total_cells for section in sections)}"
    )


if __name__ == "__main__":
    main()
