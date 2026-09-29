"""
Main entry point for OneReason Eval Benchmark Auto Evaluation Web Interface
Integrates submission and analysis modules with navigation
"""
import os
import gradio as gr

from submit.ui import create_submit_ui
from submit.styles import SUBMIT_CSS, SUBMIT_JS
from analyze.ui import create_analyze_sidebar, create_analyze_content, refresh_directories
from analyze.styles import ANALYZE_CSS, ANALYZE_JS
from extract_pid.ui import create_extract_pid_ui, _get_loading_status_html, LOADING_STATUS_REFRESH_INTERVAL
from extract_pid.styles import EXTRACT_PID_CSS, EXTRACT_PID_JS
from extract_pid.handlers import is_data_ready
from extract_sid.ui import create_extract_sid_ui
from extract_sid.ui import _get_loading_status_html as _get_loading_status_html_sid
from extract_sid.styles import EXTRACT_SID_CSS, EXTRACT_SID_JS
from generation.ui import create_generation_ui
from generation.styles import GENERATION_CSS, GENERATION_JS
from dashboard.ui import create_dashboard_content
from dashboard.styles import DASHBOARD_CSS, DASHBOARD_JS
from leaderboard.ui import create_leaderboard_content
from speedtest.ui import create_speedtest_ui
from speedtest.styles import SPEEDTEST_CSS, SPEEDTEST_JS
from submit.handlers import get_queue_status
from analyze.handlers import get_result_directories
from config import AUTO_REFRESH_INTERVAL, SUBMIT_ICON, ANALYZE_ICON, TRANSFORM_ICON, GENERATION_ICON, DASHBOARD_ICON, LEADERBOARD_ICON, DASHBOARD_OUTPUT_DIR
from combined_scripts import COMBINED_JS
from utils import load_svg_icon


# Load icons
submit_icon_svg = load_svg_icon(SUBMIT_ICON)
analyze_icon_svg = load_svg_icon(ANALYZE_ICON)
transform_icon_svg = load_svg_icon(TRANSFORM_ICON)
generation_icon_svg = load_svg_icon(GENERATION_ICON)
dashboard_icon_svg = load_svg_icon(DASHBOARD_ICON)
leaderboard_icon_svg = load_svg_icon(LEADERBOARD_ICON)
speedtest_icon_svg = load_svg_icon(DASHBOARD_ICON)


# Combined CSS
COMBINED_CSS = f"""
/* Override dark mode CSS variables to force light mode appearance */
.dark {{
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f9f9f9 !important;
    --body-background-fill: #ffffff !important;
    --body-text-color: #333333 !important;
    --neutral-900: #ffffff !important;
    --neutral-800: #f9f9f9 !important;
    --neutral-700: #f0f0f0 !important;
    --neutral-600: #e0e0e0 !important;
    --neutral-500: #cccccc !important;
    --neutral-400: #999999 !important;
    --neutral-300: #666666 !important;
    --neutral-200: #333333 !important;
    --neutral-100: #333333 !important;
    --table-even-background-fill: #ffffff !important;
    --table-odd-background-fill: #f9f9f9 !important;
    --background-color: #ffffff;
    --button-primary-text-color: #333333;
}}

{SUBMIT_CSS}
{ANALYZE_CSS}
{EXTRACT_PID_CSS}
{EXTRACT_SID_CSS}
{GENERATION_CSS}
{DASHBOARD_CSS}
{SPEEDTEST_CSS}

/* Navigation sidebar */
.nav-sidebar {{
    background-color: #2c2c2c !important;
    padding: 0 !important;
    min-width: 60px !important;
    max-width: 60px !important;
    height: 100vh !important;
    position: fixed !important;
    left: 0 !important;
    top: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    z-index: 1000 !important;
}}

/* Analyze Sidebar (Sibling of Nav) */
.analyze-sidebar-wrapper {{
    position: fixed !important;
    left: 60px !important;
    top: 0 !important;
    height: 100vh !important;
    width: 250px !important;
    background-color: #f9f9f9 !important;
    border-right: 1px solid #e0e0e0 !important;
    z-index: 900 !important;
    padding: 0 !important;
}}

/* Remove gap between nav button columns */
.nav-sidebar > * {{
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    min-width: 60px !important;
    flex-grow: 0 !important;
}}

.nav-sidebar .gap {{
    gap: 0 !important;
}}

.nav-sidebar .svelte-vt1mxs {{
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    min-width: 60px !important;
    max-width: 60px !important;
    flex-grow: 0 !important;
}}

.nav-sidebar .block {{
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}}

.nav-sidebar .padded {{
    padding: 0 !important;
}}

.nav-sidebar .svelte-12cmxck {{
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}}

/* Navigation buttons container */
.nav-buttons-container {{
    display: flex;
    flex-direction: column;
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100%;
}}

/* Navigation button wrapper */
.nav-button-wrapper {{
    width: 100%;
    margin: 0;
    padding: 0;
}}

/* Navigation buttons */
.nav-button {{
    background-color: #2c2c2c !important;
    color: white !important;
    border: none !important;
    border-radius: 0 !important;
    border-left: 2px solid transparent !important;
    padding: 15px 10px !important;
    margin: 0 !important;
    width: 100% !important;
    text-align: center !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
}}

.nav-button:hover {{
    background-color: #2c2c2c !important;
}}

.nav-button.active {{
    background-color: #2c2c2c !important;
    border-left: 2px solid white !important;
}}

/* SVG icon styling */
.nav-button svg {{
    width: 24px !important;
    height: 24px !important;
    transition: all 0.3s ease !important;
}}

.nav-button svg path {{
    fill: #999999 !important;
    transition: fill 0.3s ease !important;
}}

.nav-button.active svg path {{
    fill: white !important;
}}

.nav-button span {{
    display: none !important;
}}

/* Main content area */
.main-content {{
    margin-left: 60px !important;
    margin-right: 60px !important;
    padding: 0 !important;
    width: calc(100% - 60px) !important;
    max-width: none !important;
    min-height: 100vh !important;
    height: auto !important;
    overflow: visible !important;
    transition: margin-left 0.3s ease !important;
}}

.main-content > * {{
    max-width: none !important;
}}

/* When analyze sidebar is visible, shift content */
.main-content.analyze-active {{
    margin-left: 310px !important; /* 60px + 250px */
    margin-right: 16px !important;
    width: calc(100% - 326px) !important; /* 310 + 16 */
}}

/* Page containers */
.page-container {{
    width: 100% !important;
    max-width: none !important;
    height: auto !important;
    overflow-y: visible !important;
}}

.page-container > * {{
    max-width: none !important;
}}

/* Analyze page specific container */
.analyze-page-wrapper {{
    margin-left: 0 !important; /* Reset since main-content handles margin */
    width: 100% !important;
    height: 100% !important;
}}

/* Hide footer */
footer {{
    display: none !important;
}}

/* Ensure rounded corners globally */
button, .gr-button {{
    border-radius: 12px !important;
}}

/* ── Gradio toast notifications: smaller size ── */
.toast-wrap {{
    font-size: 12px !important;
    padding: 6px 12px !important;
    min-width: 160px !important;
    max-width: 320px !important;
    line-height: 1.4 !important;
}}
.toast-body,
.toast-body p {{
    font-size: 12px !important;
    margin: 0 !important;
}}
.toast-title {{
    font-size: 12px !important;
    font-weight: 600 !important;
}}
"""


def create_app():
    """Create the main Gradio application"""

    with gr.Blocks(
        theme=gr.themes.Monochrome(
            font=["ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=["ui-monospace", "Consolas", "monospace"],
        ),
        css=COMBINED_CSS,
        js=f"""
        function() {{
            // Execute Combined JS (Navigation, etc)
            ({COMBINED_JS}).apply(this);

            // Execute Submit JS
            ({SUBMIT_JS}).apply(this);

            // Execute Analyze JS
            ({ANALYZE_JS}).apply(this);

            // Execute Extract PID JS
            ({EXTRACT_PID_JS}).apply(this);

            // Execute Extract SID JS
            ({EXTRACT_SID_JS}).apply(this);

            // Execute Generation JS
            ({GENERATION_JS}).apply(this);

            // Execute Dashboard JS
            ({DASHBOARD_JS}).apply(this);

            // Execute Speed Test JS
            ({SPEEDTEST_JS}).apply(this);
        }}
        """,
        title="OneReason Eval Benchmark Auto Eval"
    ) as app:

        with gr.Row():
            # Left: Navigation sidebar
            with gr.Column(scale=1, elem_classes="nav-sidebar", min_width=180):
                # Both navigation buttons in one HTML block
                gr.HTML(f"""
                    <div class="nav-buttons-container">
                        <div class="nav-button-wrapper">
                            <button class="nav-button active" id="submit-nav-btn-real" onclick="document.getElementById('submit-nav-btn').click()">
                                {submit_icon_svg}
                                <span>提交任务</span>
                            </button>
                        </div>
                        <div class="nav-button-wrapper">
                            <button class="nav-button" id="analyze-nav-btn-real" onclick="document.getElementById('analyze-nav-btn').click()">
                                {analyze_icon_svg}
                                <span>结果分析</span>
                            </button>
                        </div>
                        <div class="nav-button-wrapper">
                            <button class="nav-button" id="extract-pid-nav-btn-real" onclick="document.getElementById('extract-pid-nav-btn').click()">
                                {transform_icon_svg}
                                <span>Extract PID</span>
                            </button>
                        </div>
                        <div class="nav-button-wrapper">
                            <button class="nav-button" id="generation-nav-btn-real" onclick="document.getElementById('generation-nav-btn').click()">
                                {generation_icon_svg}
                                <span>Generation</span>
                            </button>
                        </div>
                        <div class="nav-button-wrapper">
                            <button class="nav-button" id="extract-sid-nav-btn-real" onclick="document.getElementById('extract-sid-nav-btn').click()">
                                {transform_icon_svg}
                                <span>Extract SID</span>
                            </button>
                        </div>
                        <div class="nav-button-wrapper">
                            <button class="nav-button" id="dashboard-nav-btn-real" onclick="document.getElementById('dashboard-nav-btn').click()">
                                {dashboard_icon_svg}
                                <span>对比看板</span>
                            </button>
                        </div>
                        <div class="nav-button-wrapper">
                            <button class="nav-button" id="leaderboard-nav-btn-real" onclick="document.getElementById('leaderboard-nav-btn').click()">
                                {leaderboard_icon_svg}
                                <span>排行榜</span>
                            </button>
                        </div>
                        <div class="nav-button-wrapper">
                            <button class="nav-button" id="speedtest-nav-btn-real" onclick="document.getElementById('speedtest-nav-btn').click()">
                                {speedtest_icon_svg}
                                <span>测速</span>
                            </button>
                        </div>
                    </div>
                """)

                # Hidden Gradio buttons for event handling
                submit_nav_btn = gr.Button("提交任务", visible=False, elem_id="submit-nav-btn")
                analyze_nav_btn = gr.Button("结果分析", visible=False, elem_id="analyze-nav-btn")
                extract_pid_nav_btn = gr.Button("Extract PID", visible=False, elem_id="extract-pid-nav-btn")
                generation_nav_btn = gr.Button("Generation", visible=False, elem_id="generation-nav-btn")
                extract_sid_nav_btn = gr.Button("Extract SID", visible=False, elem_id="extract-sid-nav-btn")
                dashboard_nav_btn = gr.Button("对比看板", visible=False, elem_id="dashboard-nav-btn")
                leaderboard_nav_btn = gr.Button("排行榜", visible=False, elem_id="leaderboard-nav-btn")
                speedtest_nav_btn = gr.Button("测速", visible=False, elem_id="speedtest-nav-btn")

            # Analyze Sidebar (Sibling of Nav)
            with gr.Column(visible=False, elem_classes="analyze-sidebar-wrapper") as analyze_sidebar_container:
                analyze_sidebar, analyze_dir_checkboxes = create_analyze_sidebar()

            # Right: Main content area
            with gr.Column(scale=6, elem_classes="main-content") as main_content:
                # Submit page
                with gr.Column(visible=True, elem_classes=["page-container"]) as submit_page_container:
                    submit_page, submit_status_outputs = create_submit_ui()

                # Analyze page content
                with gr.Column(visible=False, elem_classes=["page-container", "analyze-page-wrapper"]) as analyze_page_container:
                    analyze_page = create_analyze_content(analyze_dir_checkboxes)
                
                # Extract PID page
                with gr.Column(visible=False, elem_classes=["page-container"]) as extract_pid_page_container:
                    extract_pid_page, extract_pid_components = create_extract_pid_ui()

                # Generation page
                with gr.Column(visible=False, elem_classes=["page-container"]) as generation_page_container:
                    generation_components = create_generation_ui()

                # Extract SID page
                with gr.Column(visible=False, elem_classes=["page-container"]) as extract_sid_page_container:
                    extract_sid_page, extract_sid_components = create_extract_sid_ui()

                # Dashboard page
                with gr.Column(visible=False, elem_classes=["page-container"]) as dashboard_page_container:
                    dashboard_components = create_dashboard_content(analyze_dir_checkboxes)

                # Leaderboard page
                with gr.Column(visible=False, elem_classes=["page-container"]) as leaderboard_page_container:
                    leaderboard_components = create_leaderboard_content()

                # Speed Test page
                with gr.Column(visible=False, elem_classes=["page-container"]) as speedtest_page_container:
                    speedtest_page, speedtest_components = create_speedtest_ui()

        # Navigation event handlers
        all_pages = [submit_page_container, analyze_page_container, extract_pid_page_container, generation_page_container, extract_sid_page_container, dashboard_page_container, leaderboard_page_container, speedtest_page_container]

        def _make_switch(visible_idx, show_sidebar=False):
            def switch():
                updates = [gr.update(visible=(i == visible_idx)) for i in range(len(all_pages))]
                updates.append(gr.update(visible=show_sidebar))  # analyze_sidebar
                cls = ["main-content", "analyze-active"] if show_sidebar else "main-content"
                updates.append(gr.update(elem_classes=cls))  # main_content
                return tuple(updates)
            return switch

        switch_to_submit = _make_switch(0)
        switch_to_analyze = _make_switch(1, show_sidebar=True)
        switch_to_extract_pid = _make_switch(2)
        switch_to_generation = _make_switch(3)
        switch_to_extract_sid = _make_switch(4)
        switch_to_dashboard = _make_switch(5, show_sidebar=True)
        switch_to_leaderboard = _make_switch(6)
        switch_to_speedtest = _make_switch(7)

        nav_outputs = all_pages + [analyze_sidebar_container, main_content]

        submit_nav_btn.click(
            fn=switch_to_submit,
            inputs=None,
            outputs=nav_outputs
        )

        analyze_nav_btn.click(
            fn=switch_to_analyze,
            inputs=None,
            outputs=nav_outputs
        ).then(
            fn=refresh_directories,
            inputs=None,
            outputs=analyze_dir_checkboxes
        )

        extract_pid_nav_btn.click(
            fn=switch_to_extract_pid,
            inputs=None,
            outputs=nav_outputs
        )

        generation_nav_btn.click(
            fn=switch_to_generation,
            inputs=None,
            outputs=nav_outputs
        )

        extract_sid_nav_btn.click(
            fn=switch_to_extract_sid,
            inputs=None,
            outputs=nav_outputs
        )

        dashboard_nav_btn.click(
            fn=switch_to_dashboard,
            inputs=None,
            outputs=nav_outputs
        ).then(
            fn=refresh_directories,
            inputs=None,
            outputs=analyze_dir_checkboxes
        )

        leaderboard_nav_btn.click(
            fn=switch_to_leaderboard,
            inputs=None,
            outputs=nav_outputs
        )

        speedtest_nav_btn.click(
            fn=switch_to_speedtest,
            inputs=None,
            outputs=nav_outputs
        )

        # Load initial data
        app.load(
            fn=get_queue_status,
            inputs=None,
            outputs=submit_status_outputs
        )

        app.load(
            fn=lambda: gr.update(choices=get_result_directories(), value=[]),
            inputs=None,
            outputs=analyze_dir_checkboxes
        )

        # Auto-refresh for submit page queue status
        app.load(
            fn=get_queue_status,
            inputs=None,
            outputs=submit_status_outputs,
            every=AUTO_REFRESH_INTERVAL
        )

        # Auto-refresh for SID2PID loading status (stop when done)
        def refresh_loading_status():
            """Refresh loading status, return cancel signal when done."""
            return _get_loading_status_html()

        app.load(
            fn=refresh_loading_status,
            inputs=None,
            outputs=extract_pid_components['loading_status'],
            every=LOADING_STATUS_REFRESH_INTERVAL
        )

        # Auto-refresh for PID2SID loading status (stop when done)
        def refresh_loading_status_sid():
            """Refresh loading status for Extract SID, return cancel signal when done."""
            return _get_loading_status_html_sid()

        app.load(
            fn=refresh_loading_status_sid,
            inputs=None,
            outputs=extract_sid_components['loading_status'],
            every=LOADING_STATUS_REFRESH_INTERVAL
        )

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("OneReason Eval Benchmark Auto Evaluation Web Interface")
    print("=" * 60)
    print("Starting server...")

    app = create_app()

    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 8890)),
        share=False,
        prevent_thread_lock=False,
        show_error=True,
        quiet=False,
        show_api=False,
        allowed_paths=[str(DASHBOARD_OUTPUT_DIR)],
    )
