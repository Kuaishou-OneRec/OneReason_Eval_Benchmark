"""Gradio UI components for the leaderboard page.

The leaderboard tab loads all evaluation data automatically (no directory
selection required).  This module creates the content area with a generate
button, version/think-mode filter dropdowns, status text, and HTML output.
"""
from __future__ import annotations

from typing import List, Tuple

import gradio as gr

from leaderboard.handlers import generate_leaderboard_html

from tasks_meta import get_all_versions, get_tasks_metadata


def _get_all_task_names_from_meta(version_filter: str = "全部") -> List[str]:
    versions = [version_filter] if version_filter != "全部" else get_all_versions()
    tasks: set = set()
    for v in versions:
        try:
            meta = get_tasks_metadata(v)
            tasks.update(meta.get("all_tasks", []))
        except Exception:
            pass
    return sorted(tasks)


def create_leaderboard_content() -> dict:
    """Create the leaderboard main content area and wire events.

    Returns:
        Dict with component references:
        {generate_btn, status_text, html_output, version_dropdown, think_dropdown}
    """
    with gr.Row():
        generate_btn = gr.Button(
            "生成排行榜",
            variant="primary",
            elem_classes=["leaderboard-generate-btn"],
            scale=1,
            min_width=120,
        )
        version_dropdown = gr.Dropdown(
            choices=["全部"] + get_all_versions(),
            value="全部",
            label="版本筛选",
            interactive=True,
            scale=2,
            min_width=120,
        )
        think_dropdown = gr.Dropdown(
            choices=["全部", "think", "nonthink"],
            value="全部",
            label="思考模式",
            interactive=True,
            scale=2,
            min_width=140,
        )

    _initial_tasks = _get_all_task_names_from_meta()
    task_dropdown = gr.Dropdown(
        choices=_initial_tasks,
        value=[],
        label="任务筛选（不选则显示全部）",
        multiselect=True,
        interactive=True,
    )

    status_text = gr.Textbox(
        label="状态",
        value="点击「生成排行榜」加载全量排行数据",
        interactive=False,
        lines=1,
        max_lines=2,
        elem_classes=["leaderboard-status"],
    )

    html_output = gr.HTML(
        value='<div class="leaderboard-placeholder" style="text-align:center; padding:60px; color:#999;">点击「生成排行榜」加载</div>',
    )

    def on_generate() -> Tuple:
        html, status = generate_leaderboard_html()
        tasks: List[str] = _get_all_task_names_from_meta()
        return html, status, gr.update(choices=tasks, value=[])

    generate_btn.click(
        fn=on_generate,
        inputs=[],
        outputs=[html_output, status_text, task_dropdown],
    )

    def on_filter(version: str, think: str, tasks: List[str]) -> Tuple:
        task_filter = tasks if tasks else None
        html, _status = generate_leaderboard_html(
            version_filter=version, think_filter=think, task_filter=task_filter
        )
        new_tasks = _get_all_task_names_from_meta(version)
        return html, gr.update(choices=new_tasks, value=[t for t in tasks if t in new_tasks])

    version_dropdown.change(
        fn=on_filter,
        inputs=[version_dropdown, think_dropdown, task_dropdown],
        outputs=[html_output, task_dropdown],
    )
    think_dropdown.change(
        fn=lambda v, t, tasks: on_filter(v, t, tasks)[0],
        inputs=[version_dropdown, think_dropdown, task_dropdown],
        outputs=[html_output],
    )
    task_dropdown.change(
        fn=lambda v, t, tasks: on_filter(v, t, tasks)[0],
        inputs=[version_dropdown, think_dropdown, task_dropdown],
        outputs=[html_output],
    )

    return {
        "generate_btn": generate_btn,
        "status_text": status_text,
        "html_output": html_output,
        "version_dropdown": version_dropdown,
        "think_dropdown": think_dropdown,
        "task_dropdown": task_dropdown,
    }
