"""HTML template with inline CSS and JS for the dashboard, extracted from analyze.py."""

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
      padding: 52px 24px 24px;
    }

    .section-nav {
      position: fixed;
      top: 8px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 100;
      display: flex;
      gap: 4px;
      padding: 8px 16px;
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(8px);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.10);
      overflow-x: auto;
      white-space: nowrap;
      max-width: calc(100% - 32px);
    }
    .section-nav .nav-link {
      display: inline-block;
      padding: 5px 14px;
      border-radius: 8px;
      border: none;
      background: transparent;
      font-size: 13px;
      font-weight: 600;
      color: #475569;
      cursor: pointer;
      transition: background 0.15s, color 0.15s;
      flex-shrink: 0;
    }
    .section-nav .nav-link:hover {
      background: var(--blue-soft);
      color: #1d4ed8;
    }
    .section-nav .nav-link.active {
      background: #3b82f6;
      color: #fff;
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

    .race-score-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(620px, 1fr));
      gap: 14px;
      margin-top: 16px;
    }

    .race-model-pair {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }

    .race-score-card,
    .race-contribution-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      min-width: 0;
    }

    .race-score-head {
      margin-bottom: 12px;
    }

    .race-score-title {
      font-size: 16px;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.2;
    }

    .race-score-meta {
      margin-top: 4px;
      font-size: 12px;
      color: var(--subtle);
    }

    .race-score-values {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }

    .race-score-item {
      border: 1px solid rgba(203, 213, 225, 0.9);
      border-radius: 12px;
      padding: 10px 8px;
      background: #ffffff;
      text-align: center;
    }

    .race-score-label {
      font-size: 12px;
      font-weight: 700;
      color: var(--subtle);
    }

    .race-score-value {
      margin-top: 6px;
      font-size: 22px;
      font-weight: 800;
      color: #0f172a;
      line-height: 1;
      font-variant-numeric: tabular-nums;
    }

    .race-score-missing {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px dashed var(--line-strong);
      font-size: 12px;
      color: #475569;
    }

    .race-contribution-groups {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .race-contribution-group {
      min-width: 0;
      border: 1px solid rgba(203, 213, 225, 0.9);
      border-radius: 12px;
      padding: 10px;
      background: #ffffff;
    }

    .race-contribution-group-title {
      margin-bottom: 6px;
      color: var(--subtle);
      font-size: 12px;
      font-weight: 700;
    }

    .race-contribution-rows {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .race-contribution-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: start;
      gap: 10px;
      font-size: 12px;
    }

    .race-contribution-task {
      min-width: 0;
      color: #475569;
      overflow-wrap: anywhere;
    }

    .race-contribution-value {
      color: #0f172a;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }

    .race-score-missing-item {
      line-height: 1.4;
    }

    .race-score-missing-label {
      font-weight: 700;
      color: #b45309;
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
      white-space: normal;
      word-break: break-all;
      text-align: center;
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
    .kind-toggle,
    .scope-filter-toggle {
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
    .kind-toggle.off,
    .scope-filter-toggle.off {
      opacity: 0.45;
      background: #f1f5f9;
    }

    .version-toggle input {
      margin: 0;
      accent-color: #2563eb;
    }

    .metric-toggle input,
    .task-toggle input,
    .kind-toggle input,
    .scope-filter-toggle input {
      margin: 0;
      accent-color: #2563eb;
    }

    .scope-filter-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
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

    .scope-hidden {
      display: none !important;
    }

    @media (max-width: 1440px) {
      .summary-grid,
      .two-col {
        grid-template-columns: 1fr;
      }

      .race-score-values {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .race-model-pair {
        grid-template-columns: 1fr;
      }

      .page {
        width: 100%;
      }
    }

    @media (max-width: 720px) {
      .race-score-grid {
        grid-template-columns: 1fr;
      }

      .race-contribution-groups {
        grid-template-columns: 1fr;
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

    function updateScopeVisibility() {
      qsa('[data-scope-panel]').forEach((panel) => {
        const sectionKey = panel.dataset.sectionKey || '';
        const visibleScopes = new Set(
          qsa('.scope-filter-toggle input:checked', panel).map((input) => input.dataset.filterScope)
        );

        qsa('.scope-filter-toggle', panel).forEach((label) => {
          const input = label.querySelector('input');
          label.classList.toggle('off', !input.checked);
        });

        const root = getSectionRootByKey(sectionKey);
        if (!root) return;

        qsa('table[data-filterable-table]', root).forEach((table) => {
          const visibleLeafCountByGroup = new Map();
          const visibleLeafCountByGroupKind = new Map();

          qsa('col.leaf-col, th.col-index-head, th.scope-head, td.value-cell', table).forEach((node) => {
            const scope = node.dataset.scope || '';
            const isScoped = node.hasAttribute('data-scope') && scope !== '';
            if (!isScoped) return;
            const shouldShow = visibleScopes.has(scope);
            if (shouldShow) {
              node.classList.remove('scope-hidden');
            } else {
              node.classList.add('scope-hidden');
            }
          });

          qsa('col.leaf-col', table).forEach((col) => {
            const hidden = col.classList.contains('is-hidden') || col.classList.contains('scope-hidden');
            if (hidden) return;
            const groupKey = col.dataset.groupKey || '';
            const kind = col.dataset.kind || '';
            visibleLeafCountByGroup.set(groupKey, (visibleLeafCountByGroup.get(groupKey) || 0) + 1);
            const groupKindKey = `${groupKey}::${kind}`;
            visibleLeafCountByGroupKind.set(groupKindKey, (visibleLeafCountByGroupKind.get(groupKindKey) || 0) + 1);
          });

          qsa('th.kind-head[data-group-key][data-kind]', table).forEach((node) => {
            const groupKey = node.dataset.groupKey || '';
            const kind = node.dataset.kind || '';
            const visibleLeafCount = visibleLeafCountByGroupKind.get(`${groupKey}::${kind}`) || 0;
            node.colSpan = Math.max(1, visibleLeafCount);
          });

          qsa('th.group-head[data-group-key]', table).forEach((node) => {
            const groupKey = node.dataset.groupKey || '';
            const visibleLeafCount = visibleLeafCountByGroup.get(groupKey) || 0;
            node.colSpan = Math.max(1, visibleLeafCount);
          });
        });
      });
    }

    function updateColumnIndexHeads(root = document) {
      qsa('table[data-filterable-table]', root).forEach((table) => {
        const visibleHeads = qsa('th.col-index-head', table).filter(
          (cell) => !cell.classList.contains('is-hidden') && !cell.classList.contains('scope-hidden')
        );
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
      updateScopeVisibility();
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
      return Math.max(60, Math.min(420, Math.round(numeric)));
    }

    function applyTableWidth(sectionKey, value) {
      const next = clampTableWidth(value);
      const px = `${next}px`;
      qsa(`[data-width-scope="${sectionKey}"]`).forEach((node) => {
        node.style.setProperty('--leaf-col-width', px);
        node.style.setProperty('--version-toggle-width', `${Math.max(154, next + 32)}px`);
      });
      qsa(`[data-width-scope="${sectionKey}"] col.leaf-col`).forEach((col) => {
        col.style.width = px;
      });
      qsa(`[data-width-scope="${sectionKey}"] .value-cell, [data-width-scope="${sectionKey}"] .col-index-head, [data-width-scope="${sectionKey}"] .scope-head, [data-width-scope="${sectionKey}"] .kind-head`).forEach((cell) => {
        cell.style.width = px;
        cell.style.minWidth = px;
        cell.style.maxWidth = px;
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
      const px = `${next}px`;
      qsa(`[data-width-scope="${sectionKey}"]`).forEach((node) => {
        node.style.setProperty('--label-col-width', px);
      });
      qsa(`[data-width-scope="${sectionKey}"] col.sticky-col-2, [data-width-scope="${sectionKey}"] col.sticky-col-3`).forEach((col) => {
        col.style.width = px;
      });
      const stickyCol1Width = parseFloat(getComputedStyle(qs(`[data-width-scope="${sectionKey}"]`) || document.documentElement).getPropertyValue('--sticky-col-1-width')) || 170;
      qsa(`[data-width-scope="${sectionKey}"] .sticky-2`).forEach((cell) => {
        cell.style.width = px;
        cell.style.minWidth = px;
        cell.style.maxWidth = px;
        cell.style.left = `${stickyCol1Width}px`;
      });
      qsa(`[data-width-scope="${sectionKey}"] .sticky-3`).forEach((cell) => {
        cell.style.width = px;
        cell.style.minWidth = px;
        cell.style.maxWidth = px;
        cell.style.left = `${stickyCol1Width + next}px`;
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
        if (savedTableWidth) {
          manualWidthSections.add(sectionKey);
          applyTableWidth(sectionKey, savedTableWidth);
        }
        applyLabelWidth(sectionKey, savedLabelWidth || 168);
      });
    }

    const manualWidthSections = new Set();

    const DEFAULT_LEAF_WIDTH = 122;

    function autoFitTableWidths() {
      qsa('[data-width-scope]').forEach((node) => {
        const sectionKey = node.dataset.widthScope;
        if (!sectionKey) return;
        if (manualWidthSections.has(sectionKey)) return;
        const tableShell = node.querySelector('.table-shell');
        if (!tableShell) return;
        const availableWidth = tableShell.clientWidth;
        if (availableWidth <= 0) return;
        const stickyStyle = getComputedStyle(node);
        const stickyCol1 = parseFloat(stickyStyle.getPropertyValue('--sticky-col-1-width')) || 170;
        const labelColW = parseFloat(stickyStyle.getPropertyValue('--label-col-width')) || 168;
        const stickyCount = node.querySelectorAll('col[class^="sticky-col-"]').length;
        const labelCount = Math.max(0, stickyCount - 1);
        const fixedWidth = stickyCol1 + labelColW * labelCount;
        const visibleLeafCols = Array.from(node.querySelectorAll('col.leaf-col')).filter((col) => {
          const groupKey = col.dataset.groupKey || '';
          const kind = col.dataset.kind || '';
          const scope = col.dataset.scope || '';
          const groupRow = node.querySelector(`.version-toggle input[data-filter-group-key="${groupKey}"]`);
          if (groupRow && !groupRow.checked) return false;
          const kindInput = node.querySelector(`.kind-toggle input[data-filter-kind="${kind}"]`);
          if (kindInput && !kindInput.checked) return false;
          if (scope) {
            const scopeInput = node.querySelector(`.scope-filter-toggle input[data-filter-scope="${scope}"]`);
            if (scopeInput && !scopeInput.checked) return false;
          }
          return true;
        });
        const leafCount = visibleLeafCols.length;
        if (leafCount <= 0) return;
        const totalNeeded = fixedWidth + leafCount * DEFAULT_LEAF_WIDTH + 2;
        if (totalNeeded <= availableWidth) {
          // Fits at default width — use default, don't stretch
          applyTableWidth(sectionKey, DEFAULT_LEAF_WIDTH);
        } else {
          // Would overflow — shrink to fit
          const shrunkWidth = Math.floor((availableWidth - fixedWidth - 2) / leafCount);
          applyTableWidth(sectionKey, clampTableWidth(shrunkWidth));
        }
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

      qsa('.scope-filter-toggle input').forEach((input) => {
        input.addEventListener('change', () => {
          applyAllFilters();
        });
      });

      qsa('[data-action="select-all-scopes"]').forEach((button) => {
        button.addEventListener('click', () => {
          const sectionKey = button.dataset.sectionKey || '';
          const panel = document.querySelector(`[data-scope-panel][data-section-key="${sectionKey}"]`);
          if (!panel) return;
          qsa('.scope-filter-toggle input', panel).forEach((input) => { input.checked = true; });
          applyAllFilters();
        });
      });

      qsa('[data-action="clear-scopes"]').forEach((button) => {
        button.addEventListener('click', () => {
          const sectionKey = button.dataset.sectionKey || '';
          const panel = document.querySelector(`[data-scope-panel][data-section-key="${sectionKey}"]`);
          if (!panel) return;
          qsa('.scope-filter-toggle input', panel).forEach((input) => { input.checked = false; });
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

      qsa('[data-action="download-html"]').forEach((button) => {
        button.addEventListener('click', () => {
          const html = '<!DOCTYPE html>\\n' + document.documentElement.outerHTML;
          const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
          const filename = `dashboard_${ts}.html`;
          const a = document.createElement('a');
          a.href = 'data:text/html;charset=utf-8,' + encodeURIComponent(html);
          a.download = filename;
          a.target = '_blank';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        });
      });

      qsa('[data-width-preset]').forEach((button) => {
        button.addEventListener('click', () => {
          manualWidthSections.add(button.dataset.sectionKey);
          applyTableWidth(button.dataset.sectionKey, button.dataset.widthPreset);
        });
      });

      qsa('[data-table-width-range]').forEach((input) => {
        input.addEventListener('input', () => {
          manualWidthSections.add(input.dataset.sectionKey);
          applyTableWidth(input.dataset.sectionKey, input.value);
        });
      });

      qsa('[data-table-width-number]').forEach((input) => {
        input.addEventListener('input', () => {
          manualWidthSections.add(input.dataset.sectionKey);
          applyTableWidth(input.dataset.sectionKey, input.value);
        });
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
      autoFitTableWidths();

      const resizeObserver = new ResizeObserver(() => autoFitTableWidths());
      const pageEl = document.querySelector('.page');
      if (pageEl) resizeObserver.observe(pageEl);

      // Section nav click + scroll spy
      const navLinks = qsa('.section-nav .nav-link');
      if (navLinks.length) {
        navLinks.forEach((btn) => {
          btn.addEventListener('click', () => {
            const target = document.getElementById(btn.dataset.navTarget || '');
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          });
        });
        const updateActiveNav = () => {
          let current = '';
          for (const btn of navLinks) {
            const id = btn.dataset.navTarget || '';
            const el = document.getElementById(id);
            if (el && el.getBoundingClientRect().top <= 80) current = id;
          }
          navLinks.forEach((b) => {
            b.classList.toggle('active', b.dataset.navTarget === current);
          });
        };
        window.addEventListener('scroll', updateActiveNav, { passive: true });
        updateActiveNav();
      }
    }

    try {
      bindDashboardActions();
      console.log('[dashboard] bindDashboardActions OK');
      console.log('[dashboard] buttons found:', qsa('button[data-action]').length);
      console.log('[dashboard] inputs found:', qsa('input[type="range"]').length);
    } catch (e) {
      console.error('[dashboard] bindDashboardActions FAILED:', e);
    }
  </script>
</body>
</html>
"""
