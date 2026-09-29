"""
CSS and JavaScript styles for task submission UI
"""

# Custom CSS for submit page
SUBMIT_CSS = """
/* --- Global: Border radius --- */
button, .gr-button { border-radius: 12px !important; }
input[type='text'], textarea, .gr-textbox, .gr-input { border-radius: 12px !important; }
.gr-tabs { border-radius: 12px !important; }

/* Username and model path input boxes rounded corners */
/* Make all parent containers white with rounded corners */
.gap:has(.rounded-textbox),
.form:has(.rounded-textbox) {
    background: white !important;
    border-radius: 12px !important;
    border-color: #e5e5e5 !important;
}

.username-row .gap,
.username-row .form,
.username-row .rounded-textbox,
.username-row .block {
    background: white !important;
    border-radius: 12px !important;
    border-color: #e5e5e5 !important;
}

.rounded-textbox {
    background: white !important;
    border-radius: 12px !important;
    border: none !important;
}

.rounded-textbox .block,
.rounded-textbox .form {
    background: white !important;
    border-radius: 12px !important;
}

.rounded-textbox textarea,
.rounded-textbox input {
    border-radius: 12px !important;
    background-color: white !important; /* Ensure input itself is white */
    border: 1px solid #ddd !important; /* Light gray border */
}

/* Force remove all wrapper styling */
.think-checkbox,
.think-checkbox .block,
.think-checkbox .wrap,
.think-checkbox .form,
.think-checkbox .gap,
.think-checkbox > div {
    border: none !important;
    border-width: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Icon buttons (Refresh, Delete, Copy) */
.icon-button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.icon-button:hover {
    background: rgba(0,0,0,0.05) !important;
}

/* Add scrollbar to the tab content area at 70vh */
.queue-area .tabitem,
.queue-area [role="tabpanel"],
.queue-area .svelte-19hvt5v {
    max-height: 70vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

/* Allow all other elements to expand naturally */
.queue-area,
.queue-area > .block,
.queue-area > .block > .wrap,
.queue-area .tabs,
.queue-df,
.queue-df .wrap,
.queue-df .dataframe,
.queue-df .table-wrap,
.queue-df .tbody-wrap,
.queue-df table {
    max-height: none !important;
    overflow: visible !important;
}

/* Tab buttons: rounded top, straight bottom */
.gr-tab-button,
button[role="tab"],
.tabs button,
div[role="tablist"] button,
.svelte-1uw5tnk button[role="tab"],
.queue-area button[role="tab"],
.queue-area .tabs button {
    border-radius: 12px 12px 0 0 !important;
}

/* All DataFrames have rounded corners by default */
.gr-dataframe { border-radius: 12px !important; overflow: visible; }

/* Preview table: all four corners rounded */
.preview-df,
.preview-df .dataframe,
.preview-df table,
.preview-df .table-wrap,
.preview-df .wrap {
    border-radius: 12px !important;
    overflow: visible !important;
    --font: 'ui-sans-serif', 'system-ui', sans-serif;
    --font-mono: 'ui-monospace', 'Consolas', monospace;
    font-family: var(--font) !important;
    font-size: 12px !important;
}

/* Preview DataFrame block container - prevent bottom spacing */
.preview-df.block,
.preview-df > .wrap,
.preview-df .svelte-1oa6fve {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* Ensure preview table shows all content without height restrictions */
.preview-df,
.preview-df .wrap,
.preview-df .table-wrap,
.preview-df .tbody-wrap,
.preview-df .svelte-1oa6fve {
    max-height: none !important;
    height: auto !important;
    overflow: visible !important;
}

/* Remove dynamic bottom padding from tbody that causes green space */
.preview-df tbody,
.preview-df tbody.tbody,
.preview-df .svelte-82jkx {
    padding-bottom: 0 !important;
    padding-top: 0 !important;
}

/* Disable Gradio's virtual scrolling for preview table */
.preview-df .table-wrap,
.preview-df .tbody-wrap {
    contain: none !important;
}

/* Force all rows to render (disable virtual scrolling) */
.preview-df tbody tr {
    display: table-row !important;
    visibility: visible !important;
}

/* Widen Think column in preview table */
.preview-table td:nth-child(2), .preview-table th:nth-child(2) {{
    min-width: 120px !important;
}}

/* Preview table: individual cell corner radius */
.preview-table thead tr:first-child th:first-child {
    border-top-left-radius: 11px !important;
}

.preview-table thead tr:first-child th:last-child {
    border-top-right-radius: 11px !important;
}

.preview-table tbody tr:last-child td:first-child {
    border-bottom-left-radius: 11px !important;
}

.preview-table tbody tr:last-child td:last-child {
    border-bottom-right-radius: 11px !important;
}

/* Queue table inner layer: all four corners rounded */
.queue-df,
.queue-df .dataframe,
.queue-df table,
.queue-df .wrap,
.queue-df.block {
    border-radius: 12px !important;
}

    /* Queue Title Row */
    .queue-title-row {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        margin-bottom: 10px !important;
    }
    
    .queue-title-row h2 {
        margin: 0 !important;
    }

    /* Custom HTML Table Styling */
    .custom-queue-table {
        width: 100%;
        border-collapse: separate !important; /* Changed from collapse to separate for border-radius */
        border-spacing: 0;
        font-size: 12px;
        border: 1px solid #e5e7eb !important;
        border-top-color: #e5e7eb !important;
        border-left-color: #e5e7eb !important;
        border-right-color: #e5e7eb !important;
        border-bottom-color: #e5e7eb !important;
        border-radius: 8px !important; /* Rounded corners */
        overflow: visible !important; /* Clip content for rounded corners */
    }

    .custom-queue-table th, .custom-queue-table td {
        padding: 8px 12px;
        text-align: left;
        border: 1px solid #e5e7eb !important;
    }

    .custom-queue-table th:last-child, .custom-queue-table td:last-child {
        border-right: none;
    }

    .custom-queue-table tr:last-child td {
        border-bottom: none;
    }

    /* 在tab-nav下方添加遮罩，遮挡滚动上去的表格内容 */
    .queue-area .tab-nav::after {
        content: '';
        position: absolute;
        left: 50%;                     /* 从中心开始 */
        transform: translateX(-50%);   /* 居中对齐 */
        width: 98%;                    /* 占总宽度的95% */
        bottom: -14px;  /* 在tab-nav下方 */
        height: 12.5px;
        background-color: white;
        z-index: 50;
        pointer-events: none;
    }

    .custom-queue-table th {
        background-color: #f9f9f9;
        font-weight: 600;
        color: #333;
        position: sticky;
        top: 0;
        z-index: 10;
    }

    .custom-queue-table th:first-child {
        border-top-left-radius: 7px; /* slightly less than container to fit */
    }

    .custom-queue-table th:last-child {
        border-top-right-radius: 7px;
    }

    .custom-queue-table tbody tr:last-child td:first-child {
        border-bottom-left-radius: 7px;
    }

    .custom-queue-table tbody tr:last-child td:last-child {
        border-bottom-right-radius: 7px;
    }

    .custom-queue-table tr:hover {
        background-color: #f5f5f5;
    }
    
    /* Icon Buttons */
    .queue-actions {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    
    .icon-button {
        background: none;
        border: none;
        cursor: pointer;
        padding: 4px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        transition: background-color 0.2s;
        margin: 0 !important; /* Fix alignment issue */
    }
    
    .icon-button:hover {
        background-color: #f0f0f0;
    }
    
    .icon-button svg {
        width: 20px;
        height: 20px;
    }
    
.custom-queue-table td {
    color: #4b5563;
    vertical-align: middle;
    word-break: break-word;
}

    border-bottom: none;
}

.custom-queue-table tr:hover td {
    background-color: #f3f4f6;
}

/* Checkbox styles */
.del-checkbox {
    width: 16px;
    height: 16px;
    cursor: pointer;
    accent-color: #1976d2;
}

.gr-box { border-radius: 12px !important; }

/* Border for submit and queue areas */
.submit-area, .queue-area {
    border: 2px solid #888888;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
}

/* Reduce bottom spacing */
.queue-area {
    margin-bottom: 10px !important;
    padding-bottom: 5px !important;
}

/* Remove excessive bottom padding from gradio container */
.gradio-container {
    padding-bottom: 10px !important;
}

/* Remove excessive spacing after tables */
.queue-df,
.deletable-df,
.preview-df {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* Reduce spacing in tab content */
.queue-area .tabitem {
    padding-bottom: 5px !important;
}

/* Username row alignment (bottom align) */
.username-row { align-items: end !important; }
.username-row .gr-column:nth-child(2) div {
    margin-bottom: 10px;
    font-weight: bold;
    font-size: 1.1em;
}

/* Queue status title row */
.queue-title-row { display: flex; align-items: center; }
.queue-title-row > h2 { flex-grow: 1; margin: 0; }
.queue-title-row > button { flex-grow: 0; }

/* Hide preview DataFrame add row/column buttons */
.preview-df .add-row,
.preview-df .add-col,
.preview-df [data-testid="add-row"],
.preview-df [data-testid="add-column"] {
    display: none !important;
}

/* Hide controls-wrap area (contains "Add Row" button) - highest priority */
#component-13 > div > div.controls-wrap,
.preview-df > div > div.controls-wrap,
.block.preview-df .controls-wrap {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Universal selector - for all possible structures */
.preview-df .svelte-1oa6fve .controls-wrap {
    display: none !important;
}

/* Hide deletable queue DataFrame add row and column buttons */
.deletable-df .controls-wrap .button-wrap {
    display: none !important;
}

/* Modify deletable queue DataFrame controls-wrap area */
.deletable-df .controls-wrap {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    padding: 8px !important;
    gap: 8px !important;
    min-height: 40px !important;
}

/* Beautify "Select" column checkbox */
.deletable-df th:first-child,
.deletable-df td:first-child {
    text-align: center !important;
}

/* Force first column width to fixed 40px */
.deletable-df .table-wrap {
    --cell-width-0: 40px !important;
}

/* Ensure first column width is 40px fixed value - apply to all possible selectors */
.deletable-df th:first-child,
.deletable-df td:first-child,
.deletable-df .svelte-1oa6fve th:first-child,
.deletable-df .svelte-1oa6fve td:first-child,
.deletable-df table th:first-child,
.deletable-df table td:first-child {
    width: 40px !important;
    min-width: 40px !important;
    max-width: 40px !important;
}

/* Force override inline styles */
.deletable-df th:first-child[style],
.deletable-df td:first-child[style] {
    width: 40px !important;
}

/* Make all tabitem containers have rounded bottom corners */
.svelte-19hvt5v {
    border-radius: 0 0 12px 12px !important;
}

/* Ensure table outer container has rounded corners */
.queue-df.block,
.deletable-df.block,
#pending-df,
#done-df,
#failed-df {
    border-radius: 12px !important;
}

/* Set border color for all queue dataframes */
#pending-df,
#processing-df,
#done-df,
#failed-df {
    border-color: #e5e7eb !important;
}

/* Adjust checkbox cell inner elements */
.deletable-df th:first-child .cell-wrap,
.deletable-df td:first-child .cell-wrap {
    justify-content: center !important;
    padding: 0 !important;
}

/* Delete button style (inline in table) */
.delete-btn-inline {
    background-color: #e0e0e0 !important;
    color: #555 !important;
    border: 1px solid #ccc !important;
    border-radius: 12px !important;
    padding: 6px 12px !important;
    font-size: 13px !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    min-width: 80px !important;
    overflow: visible !important;
}
.delete-btn-inline:hover {
    background-color: #d0d0d0 !important;
}

/* Ensure buttons in container also keep rounded corners */
.deletable-df .controls-wrap button {
    border-radius: 12px !important;
}

/* Refresh button (light gray) */
.refresh-button {
    max-width: 120px !important;
    background-color: #f0f0f0 !important;
    color: #555 !important;
    border: 1px solid #ccc !important;
}
.refresh-button:hover { background-color: #e0e0e0 !important; }

/* Unified delete button (red) */
.delete-btn-unified {
    max-width: 150px !important;
    background-color: #DC143C !important;
    color: white !important;
    border: 1px solid #DC143C !important;
}
.delete-btn-unified:hover { background-color: #B22222 !important; }

/* Submit button (dark green) */
.submit-button {
    background-color: #006400 !important;
    color: white !important;
    border: 1px solid #006400 !important;
}
.submit-button:hover { background-color: #007500 !important; }

/* Cancel button (red) */
.cancel-button {
    background-color: #DC143C !important;
    color: white !important;
    border: 1px solid #DC143C !important;
}
.cancel-button:hover { background-color: #B22222 !important; }

/* Hide footer */
footer { display: none !important; }


/* ===== Radio Button Styles (Segmented Control) ===== */

/* Clean up Radio wrapper styling */
.form:has(#think-mode-radio),
.form:has(#rollout-mode-radio),
.form:has(#version-radio),
.form:has(#seed-radio),
.form:has(#eval-mode-radio),
.form:has(#sample-mode-radio),
.form:has(#model-type-radio) {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    flex-grow: 1 !important;
    min-width: auto !important;
    display: flex !important;
    align-items: center !important;
}

#think-mode-radio,
#rollout-mode-radio,
#version-radio,
#seed-radio,
#eval-mode-radio,
#sample-mode-radio,
#model-type-radio {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
}

#think-mode-radio .block,
#rollout-mode-radio .block,
#version-radio .block,
#seed-radio .block,
#eval-mode-radio .block,
#sample-mode-radio .block,
#model-type-radio .block {
    display: flex !important;
    align-items: center !important;
}

/* Horizontal layout with gap */
#think-mode-radio .wrap,
#rollout-mode-radio .wrap,
#version-radio .wrap,
#seed-radio .wrap,
#eval-mode-radio .wrap,
#sample-mode-radio .wrap,
#model-type-radio .wrap {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 10px !important;
}

/* Hide the default radio buttons */
#think-mode-radio input[type="radio"],
#rollout-mode-radio input[type="radio"],
#version-radio input[type="radio"],
#seed-radio input[type="radio"],
#eval-mode-radio input[type="radio"],
#sample-mode-radio input[type="radio"],
#model-type-radio input[type="radio"] {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Style the labels as rounded buttons */
#think-mode-radio label,
#rollout-mode-radio label,
#version-radio label,
#seed-radio label,
#eval-mode-radio label,
#sample-mode-radio label,
#model-type-radio label {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 8px 18px !important;
    margin: 0 !important;
    cursor: pointer !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #333 !important;
    background: white !important;
    border: 1px solid #ddd !important;
    border-radius: 16px !important;
    transition: all 0.2s ease !important;
    user-select: none !important;
    min-width: 110px !important;
    text-align: center !important;
}

#think-mode-radio label span,
#rollout-mode-radio label span,
#version-radio label span,
#seed-radio label span,
#eval-mode-radio label span,
#sample-mode-radio label span,
#model-type-radio label span {
    margin: 0 !important;
}

/* Hover effect */
#think-mode-radio label:hover,
#rollout-mode-radio label:hover,
#version-radio label:hover,
#seed-radio label:hover,
#eval-mode-radio label:hover,
#sample-mode-radio label:hover,
#model-type-radio label:hover {
    border-color: #bbb !important;
    background: #fafafa !important;
}

/* Selected state */
#think-mode-radio input[type="radio"]:checked + label,
#think-mode-radio label:has(input[type="radio"]:checked),
#rollout-mode-radio input[type="radio"]:checked + label,
#rollout-mode-radio label:has(input[type="radio"]:checked),
#version-radio input[type="radio"]:checked + label,
#version-radio label:has(input[type="radio"]:checked),
#seed-radio input[type="radio"]:checked + label,
#seed-radio label:has(input[type="radio"]:checked),
#eval-mode-radio input[type="radio"]:checked + label,
#eval-mode-radio label:has(input[type="radio"]:checked),
#sample-mode-radio input[type="radio"]:checked + label,
#sample-mode-radio label:has(input[type="radio"]:checked),
#model-type-radio input[type="radio"]:checked + label,
#model-type-radio label:has(input[type="radio"]:checked) {
    background: white !important;
    color: #1976d2 !important;
    border-color: #1976d2 !important;
    font-weight: 600 !important;
    box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.1) !important;
}

/* Disabled state for sample-mode-radio (when rollout_mode = Final) */
#sample-mode-radio:not(.enabledRadio) label {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    pointer-events: none !important;
    background: #f5f5f5 !important;
    color: #999 !important;
    border-color: #e0e0e0 !important;
}

#sample-mode-radio:not(.enabledRadio) input[type="radio"]:checked + label,
#sample-mode-radio:not(.enabledRadio) label:has(input[type="radio"]:checked) {
    background: #f5f5f5 !important;
    color: #999 !important;
    border-color: #e0e0e0 !important;
    box-shadow: none !important;
}

/* Tooltip container for radio options */
.think-mode-label-container {
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
}

.think-mode-label {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #333 !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Think switches row - white card container */
.think-switches-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 15px !important;
    padding: 12px 15px 12px 15px !important;
    background: white !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}

/* First child in think-switches-row has no left margin */
.think-switches-row > div:first-child,
.think-switches-row > .think-mode-label-container {
    margin-left: 0 !important;
    padding-left: 0 !important;
}

/* Make each switch item (HTML + Checkbox) compact */
.think-switches-row > div {
    flex: none !important;
    min-width: auto !important;
    max-width: none !important;
    width: auto !important;
    padding: 0 !important;
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: center !important;
}

/* Override Gradio's default column behavior for children */
.think-switches-row > .gr-column,
.think-switches-row > [class*="col"] {
    flex: none !important;
    width: auto !important;
    min-width: auto !important;
    align-items: center !important;
    display: flex !important;
}

/* Label container - keep close to switch */
.think-switches-row .think-switch-container {
    margin-right: 8px !important;
}

/* Checkbox wrapper - keep close to label, add spacing after for next group */
.think-switches-row .think-checkbox {
    margin-left: 0 !important;
    margin-right: 30px !important;
}

.think-switch-container {
    display: flex !important;
    align-items: center !important;
}

.think-switch-item {
    display: flex !important;
    align-items: center !important;
    gap: 5px !important;
}

.think-switch-label {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #333 !important;
}

/* Tooltip icon */
.tooltip-icon {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 16px !important;
    color: #999 !important;
    cursor: help !important;
    border: none !important;
    background: transparent !important;
    margin-left: 4px !important;
    position: relative !important;
}

.tooltip-icon:hover {
    color: #666 !important;
}

/* Custom tooltip on hover */
.tooltip-icon::after {
    content: attr(data-tooltip);
    position: absolute !important;
    bottom: 125% !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    background: rgba(0, 0, 0, 0.8) !important;
    color: white !important;
    padding: 6px 10px !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    white-space: nowrap !important;
    opacity: 0 !important;
    visibility: hidden !important;
    transition: opacity 0.2s, visibility 0.2s !important;
    z-index: 1000 !important;
    pointer-events: none !important;
}

.tooltip-icon:hover::after {
    opacity: 1 !important;
    visibility: visible !important;
}

/* Think checkbox container cleanup */
.think-checkbox {
    margin: 0 !important;
    padding: 0 !important;
}

.think-checkbox .block {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Task pills container */
.task-pills-container {
    background: white !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin-bottom: 15px !important;
}

.task-pills-label {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #333 !important;
    margin-bottom: 10px !important;
}

.task-pills-wrapper {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
}

/* Task pill styling (similar to model-pill in analyze) */
.task-pill {
    display: flex !important;
    align-items: center !important;
    padding: 6px 14px !important;
    border: 1px solid #2196f3 !important;
    border-radius: 16px !important;
    background-color: #e3f2fd !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #1976d2 !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
    user-select: none !important;
}

.task-pill:hover {
    background-color: #bbdefb !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.task-pill.hidden-task {
    opacity: 0.5 !important;
    background-color: #f5f5f5 !important;
    border-color: #ccc !important;
    color: #999 !important;
    text-decoration: line-through !important;
    box-shadow: none !important;
    transform: none !important;
}

.task-pill.hidden-task:hover {
    opacity: 0.7 !important;
    background-color: #eee !important;
}

.task-pill-text {
    margin: 0 !important;
}

/* Switch Container */
.switch-container {
    display: flex !important;
    align-items: center !important;
    gap: 15px !important;
    background-color: white !important;
    padding: 10px 15px !important;
    box-shadow: none !important;
    border-radius: 16px !important;
    margin-bottom: 15px !important;
}

/* iOS Switch Styling */
.ios-switch {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    background-color: white !important;
    background: white !important;
    margin-right: 20px !important;
}

/* Remove margin from last switch */
.ios-switch:last-child {
    margin-right: 0 !important;
}

/* Add border and border-radius to individual switches */
#overwrite-switch {
    border: 1px solid #e0e0e0 !important;
    border-radius: 16px !important;
    padding: 8px 12px !important;
}

/* Force white background for all ios-switch child elements */
.ios-switch *,
.ios-switch .block,
.ios-switch .form,
.ios-switch form {
    background-color: white !important;
    background: white !important;
    box-shadow: none !important;
}

.ios-switch label {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #333 !important;
    margin-right: 5px !important;
    user-select: none !important;
    background-color: transparent !important;
    background: transparent !important;
}

/* iOS-style switch for Overwrite */
#overwrite-switch input[type="checkbox"] {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 44px !important;
    height: 24px !important;
    background: #ccc !important;
    border-radius: 24px !important;
    position: relative !important;
    cursor: pointer !important;
    transition: background 0.3s !important;
    outline: none !important;
    border: none !important;
}

#overwrite-switch input[type="checkbox"]:before {
    content: '' !important;
    position: absolute !important;
    top: 2px !important;
    left: 2px !important;
    width: 20px !important;
    height: 20px !important;
    background: white !important;
    border-radius: 50% !important;
    transition: transform 0.3s !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
}

#overwrite-switch input[type="checkbox"]:checked {
    background: #4CD964 !important;
}

#overwrite-switch input[type="checkbox"]:checked:before {
    transform: translateX(20px) !important;
}

#overwrite-switch input[type="checkbox"]:focus {
    outline: none !important;
    border: none !important;
}
"""

# Custom JavaScript for submit page
SUBMIT_JS = """
function() {
    console.log('=== SUBMIT PAGE JS LOADED ===');

    // Toggle task pill selection
    window.toggleTask = function(taskName) {
        console.log('Toggling task:', taskName);

        // Toggle the pill style
        const pill = document.getElementById('pill_' + taskName);
        if (pill) {
            pill.classList.toggle('hidden-task');
        }

        // Trigger Gradio event by setting value in hidden textbox
        const taskClickInput = document.getElementById('task-click-input');
        if (taskClickInput) {
            const textarea = taskClickInput.querySelector('textarea') || taskClickInput.querySelector('input');
            if (textarea) {
                // Add timestamp to ensure change event fires even for same task
                textarea.value = taskName;
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
    };

    // Move getSelectedTaskIds logic to global function for Gradio to call
    window.getSelectedTaskIds = function() {
        const activeTab = document.querySelector('.tabs button[aria-selected="true"]');
        if (!activeTab) {
            console.log('[Delete] No active tab');
            return JSON.stringify({ taskJsons: [], queueType: '' });
        }

        const tabText = activeTab.textContent.trim();
        console.log('[Delete] Active tab:', tabText);

        let tableId, queueType;

        if (tabText.includes('等待中')) {
            tableId = 'pending-table';
            queueType = 'pending';
        } else if (tabText.includes('已完成')) {
            tableId = 'done-table';
            queueType = 'done';
        } else if (tabText.includes('已失败')) {
            tableId = 'failed-table';
            queueType = 'failed';
        } else {
            console.log('[Delete] Tab does not support deletion');
            return JSON.stringify({ taskJsons: [], queueType: '' });
        }

        const table = document.getElementById(tableId);
        if (!table) {
            console.log('[Delete] Table not found:', tableId);
            return JSON.stringify({ taskJsons: [], queueType: '' });
        }

        const selectedTaskJsons = [];
        const rows = table.querySelectorAll('tbody tr');

        console.log('[Delete] Total rows:', rows.length);

        rows.forEach(function(row, index) {
            const checkbox = row.querySelector('.del-checkbox');
            const cells = row.querySelectorAll('td');

            // Task JSON is at index 5 (6th column)
            // Columns: [Checkbox, Model Path, Think, Output Dir, Log Path, Task JSON, ...]
            const taskJsonCell = cells[5];

            if (checkbox && checkbox.checked && taskJsonCell) {
                let taskJson = taskJsonCell.textContent.trim();

                console.log('[Delete] Row', index, 'Task JSON:', taskJson);

                if (taskJson && taskJson !== 'N/A') {
                    selectedTaskJsons.push(taskJson);
                }
            }
        });

        console.log('[Delete] Selected task JSONs:', selectedTaskJsons);
        console.log('[Delete] Queue type:', queueType);

        const result = JSON.stringify({ taskJsons: selectedTaskJsons, queueType: queueType });
        console.log('[Delete] Returning:', result);
        return result;
    };

    // No dynamic updates needed for HTML tables
    console.log('=== SUBMIT PAGE JS SETUP COMPLETE ===');
}
"""
