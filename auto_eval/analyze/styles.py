"""
CSS and JavaScript styles for results analysis UI
"""
import base64
from config import QUERY_ICON, EXCLUDE_ICON

# Custom CSS for analyze page
ANALYZE_CSS = """
/* Analyze page root */
.analyze-root {
    height: 100% !important;
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Sidebar styling */
.analyze-sidebar {
    height: 100% !important;
    background-color: #ffffff !important;
    border-right: 1px solid #e0e0e0 !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    width: 350px !important;
    min-width: 250px !important;
    max-width: 600px !important;
    resize: horizontal !important;
    z-index: 10 !important;
}

/* Aggressively remove default padding/margin/GAP from ALL sidebar children */
.analyze-sidebar * {
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
    box-sizing: border-box !important;
}

/* Force all direct children of sidebar (and their wrappers) to stack vertically */
.analyze-sidebar > *, 
.analyze-sidebar > .gradio-container,
.analyze-sidebar .block, 
.analyze-sidebar .form {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    flex-shrink: 0 !important;
    flex-grow: 0 !important;
    float: none !important;
    clear: both !important;
    overflow-x: hidden !important;
}

/* Allow the directory list container to grow and scroll */
.analyze-sidebar .directory-list {
    flex-grow: 1 !important;
    /* height: 100% !important; REMOVED to prevent overflow issues */
    min-height: 0 !important; /* Critical for flex child scrolling */
    overflow-y: auto !important;
}

/* Search box styling */
.search-box {
    margin: 20px 10px 10px 15px !important;
    flex-grow: 0 !important;
    width: 300px !important;
    display: block !important;
    position: relative !important; /* Ensure it doesn't get overlapped */
    z-index: 2 !important;
}

.search-box textarea, .search-box input {
    padding: 8px 15px 8px 35px !important;
    height: 36px !important;
    font-size: 14px !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 18px !important;
    background-color: #f5f5f5 !important;
    background-image: url('__QUERY_ICON_PLACEHOLDER__') !important;
    background-repeat: no-repeat !important;
    background-position: 10px center !important;
    background-size: 16px 16px !important;
    transition: all 0.3s ease !important;
    width: 90% !important;
}

.search-box textarea:focus, .search-box input:focus {
    background-color: #ffffff !important;
    border-color: #2196f3 !important;
    box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.1) !important;
}

/* Exclude box styling (same as search box but with different icon) */
.exclude-box {
    margin: 5px 10px 10px 15px !important;
    flex-grow: 0 !important;
    width: 300px !important;
    display: block !important;
    position: relative !important;
    z-index: 2 !important;
}

.exclude-box textarea, .exclude-box input {
    padding: 8px 15px 8px 35px !important;
    height: 36px !important;
    font-size: 14px !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 18px !important;
    background-color: #f5f5f5 !important;
    background-image: url('__EXCLUDE_ICON_PLACEHOLDER__') !important;
    background-repeat: no-repeat !important;
    background-position: 10px center !important;
    background-size: 16px 16px !important;
    transition: all 0.3s ease !important;
    width: 90% !important;
}

.exclude-box textarea:focus, .exclude-box input:focus {
    background-color: #ffffff !important;
    border-color: #2196f3 !important;
    box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.1) !important;
}

/* Directory list container */
.analyze-sidebar .directory-list {
    display: flex !important;
    flex-direction: column !important;
    flex-grow: 1 !important;
    height: auto !important;
    overflow-y: auto !important;
    min-height: 0 !important;
    border: none !important;
    box-shadow: none !important;
    padding: 5px 10px !important;
    gap: 2px !important;
}

/* Checkbox items */
.directory-list label {
    display: flex !important;
    align-items: center !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    margin: 0 !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    min-height: auto !important;
}

.directory-list label:hover {
    background-color: #f5f5f5 !important;
}

.directory-list label.selected {
    background-color: #e3f2fd !important;
    border-color: rgba(33, 150, 243, 0.1) !important;
}

/* Checkbox input */
.directory-list input[type="checkbox"] {
    margin-right: 10px !important;
    width: 18px !important;
    height: 18px !important;
    cursor: pointer !important;
    appearance: none !important;
    -webkit-appearance: none !important;
    border: 1px solid #d9d9d9 !important;
    border-radius: 50% !important; /* Circular checkbox */
    background-color: white !important;
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s !important;
    flex-shrink: 0 !important;
}

.directory-list input[type="checkbox"]:checked {
    background-color: #2196f3 !important;
    border-color: #2196f3 !important;
}

.directory-list input[type="checkbox"]:checked::after {
    content: '' !important;
    width: 10px !important;
    height: 6px !important;
    border-left: 2px solid white !important;
    border-bottom: 2px solid white !important;
    transform: rotate(-45deg) translate(1px, -1px) !important;
    display: block !important;
}

.directory-list input[type="checkbox"]:hover {
    border-color: #2196f3 !important;
}

/* Text styling */
.directory-list span {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    font-size: 12px !important;
    color: #333 !important;
    line-height: 1.3 !important;
    white-space: normal !important;
    word-break: break-word !important;
}

/* Content styling */
.analyze-content {
    height: auto !important;
    min-height: 100% !important;
    padding: 30px 0px !important;
    overflow-y: visible !important;
    background-color: white !important;
}

/* Header row styling */
.analyze-content .header-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 0 30px 0 !important;
    gap: 20px !important;
    width: 100% !important;
    flex-grow: 0 !important;
    min-height: auto !important;
    position: relative !important;
}

/* Title styling */
#analyze-title h1 {
    font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif !important;
    font-size: 28px !important;
    color: #333 !important;
    margin: 0 !important;
    font-weight: bold !important;
}

/* Generate button */
.generate-btn {
    background-color: #28a745 !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 8px 24px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    box-shadow: 0 2px 4px rgba(40, 167, 69, 0.2) !important;
    transition: all 0.2s !important;
    position: absolute !important;
    right: 0 !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: auto !important;
}

.generate-btn:hover {
    background-color: #218838 !important;
    box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3) !important;
    transform: translateY(-50%) scale(1.02) !important;
}

/* Radar chart container */
.radar-chart-container {
    text-align: center;
    margin-bottom: 30px;
    display: flex;
    justify-content: center;
}

.radar-plot {
    width: 100% !important;
    margin: 0 auto !important;
    border-radius: 12px !important;
    overflow: visible !important;
}

.radar-plot img {
    border-radius: 12px !important;
}

/* Custom HTML Summary Table Container */
.summary-table-container {
    width: 100%;
    overflow-x: auto;
    border-radius: 8px;
    border: 1px solid #ccc;
}

/* Custom HTML Summary Table */
.custom-summary-table {
    width: 100%;
    min-width: 800px; /* Ensure table is wide enough to trigger scroll on small screens */
    border-collapse: collapse;
    font-family: var(--font);
    font-size: 14px;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: none; /* Shadow moved to container or removed to avoid clipping */
}

.custom-summary-table th {
    background-color: #f8f9fa;
    color: #444;
    font-weight: 600;
    text-align: left;
    padding: 12px 16px;
    border-bottom: 2px solid #ccc;
    white-space: normal; /* Allow text wrapping */
    word-wrap: break-word;
    word-break: break-word; /* Better breaking for long words */
    overflow-wrap: break-word;
    vertical-align: top;
    min-width: 180px; /* Default min-width for model columns to prevent excessive wrapping */
}

.custom-summary-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #ccc;
    color: #555;
    vertical-align: middle;
    min-width: 180px; /* Default min-width for model columns */
}

.custom-summary-table tr:last-child td {
    border-bottom: none;
}

.custom-summary-table tr:hover {
    background-color: #f5f9ff;
}

.view-btn {
    background-color: #e3f2fd;
    color: #1976d2;
    border: none;
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.2s;
}

.view-btn:hover {
    background-color: #bbdefb;
    color: #1565c0;
}

/* Column specific sizing */
.custom-summary-table th:nth-child(1),
.custom-summary-table td:nth-child(1) {
    width: 15%;
    min-width: 120px;
    max-width: 200px;
}

.custom-summary-table th:nth-child(2),
.custom-summary-table td:nth-child(2) {
    width: 15%;
    min-width: 120px;
    max-width: 200px;
}

.custom-summary-table th:last-child,
.custom-summary-table td:last-child {
    width: 100px;
    min-width: 100px; /* Override default min-width */
    text-align: center;
    white-space: nowrap;
}

/* Table header with line style indicator */
.custom-summary-table th .line-indicator {
    display: block;
    margin: 0 0 6px 0;
    text-align: left;
}

.custom-summary-table th .model-name {
    display: block;
    text-align: left;
    font-size: 13px;
}

/* Task grouping styles - for better visual separation */
.custom-summary-table tr.task-group-even {
    background-color: #f8f9fa !important;
}

.custom-summary-table tr.task-group-odd {
    background-color: #ffffff !important;
}

.custom-summary-table tr.task-group-first {
    border-top: 2px solid #dee2e6 !important;
}

.custom-summary-table tr.task-group-even:hover,
.custom-summary-table tr.task-group-odd:hover {
    background-color: #e3f2fd !important;
}

.custom-summary-table td.task-name-cell {
    font-weight: 600 !important;
    color: #333 !important;
    vertical-align: top !important;
}





/* Switch and Copy Container */
.switch-and-copy-container {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    gap: 20px !important;
    margin-top: 30px !important;
    margin-bottom: 0 !important;
    background-color: white !important;
    background: white !important;
}

/* Force white background for all switch-and-copy-container children */
.switch-and-copy-container > *,
.switch-and-copy-container .block,
.switch-and-copy-container form {
    background-color: white !important;
    background: white !important;
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
}

/* Force white background for switch container form elements */
.switch-container .form,
.switch-container form,
.switch-container > * {
    background-color: white !important;
    background: white !important;
    border: none !important;
    box-shadow: none !important;
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
#sid-switch,
#pid-switch,
#grounding-switch {
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

/* iOS-style switch for SID, PID, and Grounding */
#sid-switch input[type="checkbox"],
#pid-switch input[type="checkbox"],
#grounding-switch input[type="checkbox"] {
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

#sid-switch input[type="checkbox"]:before,
#pid-switch input[type="checkbox"]:before,
#grounding-switch input[type="checkbox"]:before {
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

#sid-switch input[type="checkbox"]:checked,
#pid-switch input[type="checkbox"]:checked,
#grounding-switch input[type="checkbox"]:checked {
    background: #4CD964 !important;
}

#sid-switch input[type="checkbox"]:checked:before,
#pid-switch input[type="checkbox"]:checked:before,
#grounding-switch input[type="checkbox"]:checked:before {
    transform: translateX(20px) !important;
}

#sid-switch input[type="checkbox"]:focus,
#pid-switch input[type="checkbox"]:focus,
#grounding-switch input[type="checkbox"]:focus {
    outline: none !important;
    border: none !important;
}

/* Copy button container */
.copy-btn-container {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    margin: 0 !important;
}

/* Copy button */
.copy-btn {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 8px !important;
    min-width: auto !important;
    width: auto !important;
    cursor: pointer !important;
    opacity: 0.6 !important;
    transition: all 0.2s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.copy-btn:hover {
    opacity: 1 !important;
    background: rgba(0,0,0,0.05) !important;
    border-radius: 8px !important;
}

.copy-btn svg {
    width: 20px !important;
    height: 20px !important;
    display: block !important;
}

/* Modal overlay */
.modal-overlay {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background-color: rgba(0,0,0,0.5) !important;
    z-index: 1000 !important;
    backdrop-filter: blur(2px) !important;
}

/* Modal content */
.modal-content {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    background: white !important;
    padding: 30px !important; /* Normal padding, no extra space needed */
    border-radius: 8px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
    z-index: 1001 !important;
    max-width: 90vw !important;
    max-height: 90vh !important;
    overflow: auto !important; /* Changed from overflow-y to overflow */
    word-wrap: break-word !important; /* Enable word wrapping */
    overflow-wrap: break-word !important; /* Modern alternative */
}

/* Clip content to rounded corners */
.modal-content > * {
    border-radius: inherit !important;
}

/* Style the scrollbar to match rounded corners */
.modal-content::-webkit-scrollbar {
    width: 10px !important;
}

.modal-content::-webkit-scrollbar-track {
    background: transparent !important;
    border-radius: 16px !important;
}

.modal-content::-webkit-scrollbar-thumb {
    background: #ccc !important;
    border-radius: 10px !important;
    border: 2px solid white !important; /* Creates space from edge */
}

.modal-content::-webkit-scrollbar-thumb:hover {
    background: #999 !important;
}

/* Force text wrapping for all content inside modal */
.modal-content * {
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
    max-width: 100% !important;
}

/* Specific styling for model name headings in modal */
.modal-content h4 {
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
    white-space: normal !important; /* Allow wrapping */
    max-width: 100% !important;
}

/* Ensure main gradio containers don't overflow the modal rounded corners */
.modal-content > .gradio-container,
.modal-content > .block,
.modal-content > [class*="svelte-"] {
    border-radius: 0 !important; /* No border radius on top-level containers */
    overflow: visible !important; /* Allow content to flow */
}

/* Keep rounded corners for inner content cards */
.modal-content div[style*="min-width: 300px"] {
    border-radius: 16px !important;
}

/* Target sample container with border */
.modal-content div[style*="border: 2px solid"] {
    border-radius: 20px !important;
}

/* Target model cards with 1px border */
.modal-content div[style*="border: 1px solid"] {
    border-radius: 16px !important;
}

/* Close modal button - fixed at viewport top right corner, on the overlay */
.close-modal-btn {
    position: fixed !important;
    top: 20px !important;
    right: 20px !important;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    min-height: 44px !important;
    padding: 0 !important;
    margin: 0 !important;
    background-color: rgba(50, 50, 50, 0.8) !important; /* Dark semi-transparent background */
    color: white !important;
    border: none !important;
    border-radius: 50% !important; /* Circular button */
    cursor: pointer !important;
    transition: all 0.2s !important;
    font-size: 28px !important;
    font-weight: 300 !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    z-index: 9999 !important; /* Very high z-index to ensure it's on top */
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
}

.close-modal-btn:hover {
    background-color: rgba(30, 30, 30, 0.95) !important;
    transform: rotate(90deg) scale(1.1) !important; /* Rotate and scale on hover */
    box-shadow: 0 4px 16px rgba(0,0,0,0.4) !important;
}

/* Refresh Samples Custom UI */
/* Refresh Samples Custom UI */
.refresh-container {
    /* Position at the end of content flow */
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background-color: white !important;
    padding: 8px 16px !important;
    border-radius: 24px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    border: 1px solid #eee !important;
    transition: all 0.2s !important;
    
    /* Float to right or align to right */
    margin-left: auto !important; 
    margin-top: 20px !important;
    margin-bottom: 10px !important;
    width: fit-content !important;
}

.refresh-container:hover {
    box-shadow: 0 6px 16px rgba(0,0,0,0.2) !important;
    transform: translateY(-2px) !important;
}

.refresh-icon {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-right: 8px !important;
    color: #2196f3 !important;
    cursor: default !important;
}

/* Fix SVG size */
.refresh-icon svg {
    width: 20px !important;
    height: 20px !important;
}

.refresh-text {
    color: #2196f3 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    cursor: pointer !important;
    user-select: none !important;
    transition: color 0.2s !important;
}

.refresh-text {
    color: #2196f3 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    cursor: pointer !important;
    user-select: none !important;
    transition: color 0.2s !important;
}

.refresh-text:hover {
    color: #1976d2 !important;
    text-decoration: underline !important;
}

/* Loading indicator */
.loading-container {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 60px 20px !important;
    min-height: 200px !important;
}

.loading-spinner {
    width: 50px !important;
    height: 50px !important;
    border: 4px solid #f3f3f3 !important;
    border-top: 4px solid #3498db !important;
    border-radius: 50% !important;
    animation: spin 1s linear infinite !important;
    margin-bottom: 20px !important;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-text {
    color: #666 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

/* Error/Success messages */
.error-message {
    background-color: #fff2f0;
    color: #ff4d4f;
    border: 1px solid #ffccc7;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 14px;
}

/* Accordion (样本对比) rounded corners */
.analyze-content .block.svelte-12cmxck {
    border-radius: 12px !important;
    overflow: visible !important;
}

.analyze-content .label-wrap {
    border-radius: 12px 12px 0 0 !important;
}

.analyze-content .block.svelte-12cmxck[open] .label-wrap {
    border-radius: 12px 12px 0 0 !important;
}

/* Only apply to accordion elements, not modal content */
.analyze-content .svelte-vt1mxs:not(.modal-content):not(.modal-content *) {
    border-radius: 12px 12px 12px 12px !important;
}

/* Toast Notification */
.toast-notification {
    position: fixed !important;
    top: 20px !important;
    right: 20px !important;
    background-color: #DEEBF7 !important; /* Green success color */
    color: grey !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    z-index: 9999 !important;
    font-size: 14px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    opacity: 0;
    transform: translateY(-20px);
    transition: all 0.3s ease !important;
    pointer-events: none !important;
}

.toast-notification.show {
    opacity: 1 !important;
    transform: translateY(0) !important;
}

/* Gradio Warning/Info boxes */
.toast-wrap {
    max-width: 800px !important;
    min-width: 500px !important;
    width: auto !important;
}

.toast-body {
    max-width: 800px !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
}

/* Text container with copy button and expand/collapse */
.text-container {
    position: relative !important;
}

.text-container pre {
    background-color: #fff !important;
    padding: 10px 35px 10px 10px !important;
    border-radius: 4px !important;
    overflow-x: auto !important;
    font-size: 12px !important;
    margin: 0 !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
}

/* Copy icon button */
.copy-icon-btn {
    position: absolute !important;
    top: 8px !important;
    right: 8px !important;
    background: transparent !important;
    border: none !important;
    cursor: pointer !important;
    padding: 4px !important;
    opacity: 0.5 !important;
    transition: opacity 0.2s ease !important;
    color: #666 !important;
    z-index: 10 !important;
}

.copy-icon-btn:hover {
    opacity: 1 !important;
    color: #1976d2 !important;
}

.copy-icon-btn svg {
    display: block !important;
}

/* Expand/collapse text links */
.expand-text-link, .collapse-text-link {
    color: #1976d2 !important;
    font-size: 13px !important;
    cursor: pointer !important;
    display: inline !important;
    user-select: none !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

.expand-text-link:hover, .collapse-text-link:hover {
    text-decoration: underline !important;
}

/* Model Filter Styles */
.comparison-header-container {
    display: flex !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 20px !important;
    margin-bottom: 20px !important;
    padding-bottom: 10px !important;
    border-bottom: 1px solid #eee !important;
}

.model-filter-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
}

.model-pill {
    display: flex !important;
    align-items: center !important;
    padding: 6px 12px !important;
    border: 1px solid transparent !important;
    border-radius: 16px !important;
    background-color: white !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
    user-select: none !important;
}

.model-pill:hover {
    opacity: 0.8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.model-pill.hidden-model {
    opacity: 0.5 !important;
    background-color: #eee !important;
    text-decoration: line-through !important;
    box-shadow: none !important;
    transform: none !important;
}

.model-pill-text {
    margin: 0 !important;
}

/* Collapsible Sample Styles */
.sample-details {
    margin-bottom: 30px !important;
    border-radius: 20px !important;
    border: 2px solid #ddd !important;
    overflow: hidden !important;
}

.sample-summary {
    padding: 15px 20px !important;
    background-color: #f0f0f0 !important;
    cursor: pointer !important;
    list-style: none !important; /* Hide default triangle in some browsers */
    display: flex !important;
    align-items: center !important;
    font-weight: bold !important;
}

/* Custom triangle indicator */
.sample-summary::before {
    content: '▶' !important;
    display: inline-block !important;
    margin-right: 10px !important;
    transition: transform 0.2s !important;
    font-size: 12px !important;
    color: #666 !important;
}

.sample-details[open] .sample-summary::before {
    transform: rotate(90deg) !important;
}

.sample-details[open] .sample-summary {
    border-bottom: 1px solid #ddd !important;
}

/* Hide default marker */
.sample-summary::-webkit-details-marker {
    display: none !important;
}
"""

# Custom JavaScript for analyze page (if needed)
ANALYZE_JS = r"""
function() {
    console.log('=== ANALYZE PAGE JS LOADED ===');

    // Toast notification function
    window.showToast = function(message, duration = 3000) {
        // Create toast element if it doesn't exist
        let toast = document.getElementById('custom-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'custom-toast';
            toast.className = 'toast-notification';
            document.body.appendChild(toast);
        }

        // Set content
        toast.innerHTML = `<span></span> <span>${message}</span>`;

        // Show toast
        requestAnimationFrame(() => {
            toast.classList.add('show');
        });

        // Hide after duration
        setTimeout(() => {
            toast.classList.remove('show');
        }, duration);
    };

    // Function to copy table to clipboard
    window.copyTableToClipboard = function() {
        // Target the custom HTML table directly
        const table = document.querySelector('.custom-summary-table');
        if (!table) {
            console.error('Table not found');
            window.showToast('未找到表格，请先生成分析报告');
            return;
        }

        // 收集表头
        let tsvContent = '';
        const headerRow = table.querySelector('thead tr');
        if (headerRow) {
            const headers = Array.from(headerRow.querySelectorAll('th')).map(th => {
                // Skip the "Action" column (last column) if needed, or keep it
                // For copy, we usually want the data, so we might skip the "View" button column
                if (th.textContent.trim() === '操作') return null;
                return th.textContent.trim();
            }).filter(h => h !== null);
            tsvContent = headers.join('\t') + '\n';
        }

        // 收集所有数据行
        const allRows = table.querySelectorAll('tbody tr');
        allRows.forEach(row => {
            const cells = Array.from(row.querySelectorAll('td')).map((td, index) => {
                // Skip the "Action" column (last column)
                // Check if this is the "Action" column based on header or position
                // In our case, it's the column with the button
                if (td.querySelector('button')) return null;
                
                return td.textContent.trim();
            }).filter(c => c !== null);
            
            if (cells.length > 0) {
                tsvContent += cells.join('\t') + '\n';
            }
        });

        // 使用现代 Clipboard API 复制
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(tsvContent).then(() => {
                window.showToast('复制成功！');
            }).catch(err => {
                console.error('Failed to copy: ', err);
                fallbackCopy(tsvContent);
            });
        } else {
            fallbackCopy(tsvContent);
        }
    };

    // 备用复制方法
    function fallbackCopy(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            document.execCommand('copy');
            window.showToast('复制成功！');
        } catch(err) {
            console.error('Failed to copy: ', err);
            window.showToast('复制失败，请手动选择复制');
        }
        document.body.removeChild(textarea);
    }

    // Function to add "View" buttons to table rows
    function addViewButtonsToTable() {
        // This would be implemented based on specific requirements
        console.log('[addViewButtonsToTable] Setting up view buttons...');
    }

    // Function to handle dataset selection from HTML table
    window.selectDataset = function(datasetName, metricName) {
        console.log('Selecting dataset:', datasetName, 'metric:', metricName);
        // Find the hidden textbox by its class
        const selectorContainer = document.querySelector('.dataset-selector');
        if (!selectorContainer) {
            console.error('Dataset selector container not found');
            return;
        }

        const textarea = selectorContainer.querySelector('textarea');
        if (textarea) {
            // Combine dataset and metric with a separator
            const combinedValue = datasetName + '|||' + metricName;
            // Set value and trigger input event for Gradio to pick it up
            textarea.value = combinedValue;
            textarea.dispatchEvent(new Event('input', { bubbles: true }));
        } else {
            console.error('Dataset selector textarea not found');
        }
    };

    // Function to toggle text display between preview and full
    window.toggleText = function(contentId) {
        const previewDiv = document.getElementById(contentId + '_preview');
        const fullDiv = document.getElementById(contentId + '_full');

        if (!previewDiv || !fullDiv) {
            console.error('Text elements not found');
            return;
        }

        // Toggle visibility
        if (previewDiv.style.display === 'none') {
            // Currently showing full, switch to preview
            previewDiv.style.display = 'block';
            fullDiv.style.display = 'none';
        } else {
            // Currently showing preview, switch to full
            previewDiv.style.display = 'none';
            fullDiv.style.display = 'block';
        }
    };

    // Function to copy text content
    window.copyText = function(contentId) {
        // Check which div is currently visible
        const previewDiv = document.getElementById(contentId + '_preview');
        const fullDiv = document.getElementById(contentId + '_full');

        let textElement;

        // If full div is visible, copy from there; otherwise copy from preview
        if (fullDiv && fullDiv.style.display !== 'none') {
            textElement = fullDiv.querySelector('pre');
        } else if (previewDiv) {
            textElement = previewDiv.querySelector('pre');
        }

        if (!textElement) {
            console.error('Text element not found for copying');
            window.showToast('未找到要复制的内容', 2000);
            return;
        }

        const textContent = textElement.textContent;

        // Use modern Clipboard API
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(textContent).then(() => {
                window.showToast('复制成功！', 2000);
            }).catch(err => {
                console.error('Failed to copy: ', err);
                fallbackCopyText(textContent);
            });
        } else {
            fallbackCopyText(textContent);
        }
    };

    // Fallback copy method for older browsers
    function fallbackCopyText(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
            document.execCommand('copy');
            window.showToast('复制成功！', 2000);
        } catch(err) {
            console.error('Failed to copy: ', err);
            window.showToast('复制失败，请手动选择复制', 2000);
        }
        document.body.removeChild(textarea);
    }


    // Function to toggle model visibility
    window.toggleModel = function(modelName) {
        console.log('Toggling model:', modelName);
        
        // 1. Toggle the pill style
        const pill = document.getElementById('pill_' + modelName);
        if (pill) {
            pill.classList.toggle('hidden-model');
        }
        
        // 2. Toggle all model cards
        // We use querySelectorAll with the data attribute
        const cards = document.querySelectorAll(`.model-card[data-model="${modelName}"]`);
        
        cards.forEach(card => {
            if (card.style.display === 'none') {
                card.style.display = ''; // Restore default (flex/block)
            } else {
                card.style.display = 'none';
            }
        });
    };

    // Initialize
    setTimeout(addViewButtonsToTable, 500);
}
"""

# Inject query icon
try:
    with open(QUERY_ICON, "rb") as f:
        _query_icon_b64 = base64.b64encode(f.read()).decode('utf-8')
        _query_icon_url = f"data:image/svg+xml;base64,{_query_icon_b64}"
        ANALYZE_CSS = ANALYZE_CSS.replace("__QUERY_ICON_PLACEHOLDER__", _query_icon_url)
except Exception as e:
    print(f"Failed to load query icon: {e}")

# Inject exclude icon
try:
    with open(EXCLUDE_ICON, "rb") as f:
        _exclude_icon_b64 = base64.b64encode(f.read()).decode('utf-8')
        _exclude_icon_url = f"data:image/svg+xml;base64,{_exclude_icon_b64}"
        ANALYZE_CSS = ANALYZE_CSS.replace("__EXCLUDE_ICON_PLACEHOLDER__", _exclude_icon_url)
except Exception as e:
    print(f"Failed to load exclude icon: {e}")
