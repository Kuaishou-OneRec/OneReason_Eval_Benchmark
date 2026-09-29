"""Gradio UI components for the dashboard page.

The dashboard tab reuses the analyze sidebar for directory selection.
This module only creates the content area (generate button + status + iframe).
"""

from __future__ import annotations

import gradio as gr

from .handlers import generate_dashboard_html


def create_dashboard_content(dir_checkboxes: gr.CheckboxGroup) -> dict:
    """Create the dashboard main content area and wire events.

    Args:
        dir_checkboxes: Shared directory CheckboxGroup from the analyze sidebar.

    Returns:
        Dict with component references: {generate_btn, status_text, iframe_html}
    """
    generate_btn = gr.Button(
        "生成看板",
        variant="primary",
        elem_classes=["dashboard-generate-btn"],
    )

    status_text = gr.Textbox(
        label="状态",
        value="请在左侧选择目录并点击「生成看板」",
        interactive=False,
        lines=1,
        max_lines=2,
        elem_classes=["dashboard-status"],
    )

    iframe_html = gr.HTML(
        value='<div class="dashboard-iframe-placeholder" style="text-align:center; padding:60px; color:#999;">选择目录后点击「生成看板」预览对比结果</div>',
    )

    # Wire generate button
    generate_btn.click(
        fn=generate_dashboard_html,
        inputs=[dir_checkboxes],
        outputs=[iframe_html, status_text],
    )

    return {
        "generate_btn": generate_btn,
        "status_text": status_text,
        "iframe_html": iframe_html,
    }
