"""CSS and JS for the dashboard module in the Gradio web app."""

DASHBOARD_CSS = """
/* Generate button at top of dashboard content */
.dashboard-generate-btn {
    margin: 10px 0 !important;
    flex-grow: 0 !important;
    flex-shrink: 0 !important;
    max-width: 200px !important;
}

.dashboard-generate-btn button {
    padding: 8px 16px !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* Dashboard status text */
.dashboard-status {
    margin-bottom: 8px !important;
}

/* Dashboard iframe container */
.dashboard-iframe-container iframe {
    width: 100%;
    height: calc(100vh - 120px);
    border: none;
}

.dashboard-iframe-placeholder {
    width: 100%;
    min-height: calc(100vh - 120px);
}
"""

DASHBOARD_JS = """
function() {
    // Dashboard JS - placeholder for future interactivity
}
"""
