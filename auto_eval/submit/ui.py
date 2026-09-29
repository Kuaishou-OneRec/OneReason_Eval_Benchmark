"""
Gradio UI components for task submission
"""
import os
import gradio as gr
import pandas as pd
from submit.handlers import (
    parse_paths_to_preview,
    parse_api_model_to_preview,
    submit_tasks,
    submit_api_tasks,
    get_queue_status,
    delete_selected_tasks,
)
from auto_eval.tasks_meta import get_tasks_metadata
from config import AUTO_REFRESH_INTERVAL, REFRESH_ICON, DELETE_ICON, VERSION
from utils import load_svg_icon

DEFAULT_VERSION = "v3.1"

ALL_TASKS = get_tasks_metadata(DEFAULT_VERSION)["all_tasks"]
DEFAULT_SELECTED_TASKS = []

# Available closed-source models (display name -> config key in llm_config.json)
CLOSED_SOURCE_MODELS = [
    ("GPT 5.2", "gpt"),
    ("GPT 5.2 (low thinking)", "gpt-5.2-low"),
    ("GPT 5.2 (medium thinking)", "gpt-5.2-medium"),
    ("GPT 5.2 (high thinking)", "gpt-5.2-high"),
    ("GPT 5.5", "gpt-5.5"),
    ("Claude Opus 4", "claude"),
    ("DeepSeek V3.2 Thinking", "deepseek"),
    ("Gemini 3 Flash", "gemini-3-flash"),
    ("Gemini 3.1 Pro", "gemini-3.1-pro"),
    ("Gemini 3.1 Flash Lite", "gemini-3.1-flash-lite"),
    ("Gemini 2.5 Pro", "gemini-2.5-pro"),
    ("Gemini 2.5 Flash", "gemini-2.5-flash"),
]


def generate_task_pills_html(all_tasks: list, selected_tasks: list) -> str:
    """
    Generate HTML for task pills.

    Args:
        all_tasks: List of all available tasks
        selected_tasks: List of currently selected tasks

    Returns:
        HTML string for task pills
    """
    pills_html = '<div class="task-pills-wrapper">'
    for task in all_tasks:
        is_selected = task in selected_tasks
        selected_class = "" if is_selected else "hidden-task"
        pills_html += f'''
            <div class="task-pill {selected_class}" id="pill_{task}" onclick="toggleTask('{task}')">
                <span class="task-pill-text">{task}</span>
            </div>
        '''
    pills_html += '</div>'
    return pills_html


def create_submit_ui():
    """
    Create the task submission UI components.

    Returns:
        Gradio components for the submit page
    """
    with gr.Column() as submit_page:
        gr.Markdown(
            "<h1 style='text-align: center; font-family: \"Arial Rounded MT\", \"Comic Sans MS\", sans-serif; color: #171717; margin-top: 20px;'>OneReason Eval Benchmark Auto Eval</h1>"
        )

        with gr.Row():
            new_task_btn = gr.Button("➕ 新建评估任务", variant="primary", size="lg", visible=True)

        with gr.Column(visible=False, elem_classes="submit-area") as submit_area:
            gr.Markdown("## <span style='color: #171717;'>📝 提交新任务</span>")

            with gr.Row(elem_classes="username-row"):
                with gr.Column(scale=10):
                    username = gr.Textbox(
                        label="用户名",
                        placeholder="example_user",
                        elem_classes=["rounded-textbox"]
                    )
                with gr.Column(scale=1, min_width=150):
                    gr.Markdown("填写用户名或完整邮箱")

            # Model type selector (open-source vs closed-source)
            with gr.Row(elem_classes="think-switches-row"):
                gr.HTML("""
                    <div class="think-mode-label-container">
                        <span class="think-mode-label">模型类型</span>
                    </div>
                """)
                model_type_radio = gr.Radio(
                    choices=[
                        ("开源模型", "open_source"),
                        ("闭源模型", "closed_source"),
                    ],
                    value="open_source",
                    label="",
                    info="",
                    elem_id="model-type-radio"
                )

            # Open-source model path input (visible by default)
            with gr.Column(visible=True) as open_source_area:
                model_paths_text = gr.Textbox(
                    label="模型路径",
                    placeholder=os.environ.get("BENCHMARK_MODEL_PATH", "models/model"),
                    lines=6,
                    elem_classes=["rounded-textbox"]
                )
                tensor_parallel_size_dropdown = gr.Dropdown(
                    choices=[1, 2, 4, 8],
                    value=1,
                    label="Tensor Parallel Size (TP)",
                    info="Tensor Parallel Size (TP), 默认 1, 不用改",
                    elem_id="tensor-parallel-size-dropdown"
                )

            # Closed-source model selector (hidden by default)
            with gr.Column(visible=False) as closed_source_area:
                api_model_dropdown = gr.Dropdown(
                    choices=CLOSED_SOURCE_MODELS,
                    value="claude",
                    label="闭源模型",
                    info="选择要评测的闭源模型",
                    elem_id="api-model-dropdown"
                )

            version_state = gr.State(value=DEFAULT_VERSION)

            # Think mode selector (hidden for closed-source models)
            with gr.Column(visible=True) as think_rollout_area:
                # Think mode selector
                with gr.Row(elem_classes="think-switches-row"):
                    gr.HTML("""
                    <div class="think-mode-label-container">
                        <span class="think-mode-label">Think Mode</span>
                    </div>
                """)
                think_mode_radio = gr.Radio(
                    choices=[
                        ("Non Think", "Non Think"),
                    ],
                    value="Non Think",
                    label="",
                    info="",
                    elem_id="think-mode-radio"
                )

                # Rollout Mode selector
                with gr.Row(elem_classes="think-switches-row"):
                    gr.HTML("""
                    <style>
                        /* Container width control */
                        #rollout-mode-radio {
                            max-width: 1150px !important;
                            box-sizing: border-box !important;
                        }

                        /* Vertical layout for radio buttons */
                        #rollout-mode-radio fieldset {
                            display: flex !important;
                            flex-direction: column !important;
                            gap: 4px !important;
                            border: none !important;
                            padding: 0 !important;
                            width: fit-content !important;
                            max-width: 950px !important;
                            box-sizing: border-box !important;
                        }

                        /* Style each radio button */
                        #rollout-mode-radio label {
                            display: flex !important;
                            align-items: center !important;
                            padding: 6px 10px !important;
                            margin: 0 !important;
                            border: 1px solid #e5e7eb !important;
                            border-radius: 6px !important;
                            background: white !important;
                            cursor: pointer !important;
                            width: auto !important;
                            max-width: 450px !important;
                            box-sizing: border-box !important;
                            transition: all 0.2s !important;
                        }

                        #rollout-mode-radio label:hover {
                            background: #fafafa !important;
                            border-color: #9ca3af !important;
                        }

                        #rollout-mode-radio label:has(input:checked) {
                            background: #eff6ff !important;
                            border-color: #3b82f6 !important;
                        }

                        /* Style the radio button */
                        #rollout-mode-radio input[type="radio"] {
                            margin-right: 8px !important;
                        }
                    </style>
                    <div class="think-mode-label-container">
                        <span class="think-mode-label">Rollout Mode</span>
                    </div>
                """)
                    rollout_mode_radio = gr.Radio(
                    choices=[("Race", "Race")],
                    value="Race",
                    label="",
                    info="",
                    elem_id="rollout-mode-radio"
                )

            # Version selector
            with gr.Row(elem_classes="think-switches-row"):
                gr.HTML("""
                    <div class="think-mode-label-container">
                        <span class="think-mode-label">Version</span>
                    </div>
                """)
                version_radio = gr.Radio(
                    choices=[("datav3.1", "v3.1")],
                    value="v3.1",
                    label="",
                    info="",
                    elem_id="version-radio"
                )

            # Seed selector
            with gr.Row(elem_classes="think-switches-row"):
                gr.HTML("""
                    <div class="think-mode-label-container">
                        <span class="think-mode-label">Seed</span>
                    </div>
                """)
                seed_radio = gr.Radio(
                    choices=[("42", 42)],
                    value=42,
                    label="",
                    info="",
                    elem_id="seed-radio"
                )

            # Task pills area
            with gr.Column(elem_classes="task-pills-container"):
                gr.HTML('<div class="task-pills-label">评估任务</div>')
                selected_tasks = gr.State(value=DEFAULT_SELECTED_TASKS.copy())
                task_pills_html = gr.HTML(
                    value=generate_task_pills_html(ALL_TASKS, DEFAULT_SELECTED_TASKS),
                    elem_id="task-pills-area"
                )
                # Hidden textbox to receive clicked task from JS
                task_click_input = gr.Textbox(visible=False, elem_id="task-click-input")

            # Overwrite switch
            with gr.Row(elem_classes="switch-container"):
                overwrite_switch = gr.Checkbox(
                    label="Overwrite",
                    value=False,
                    elem_id="overwrite-switch",
                    elem_classes=["ios-switch"]
                )

            # CoT-quality metrics switch (only effective when
            # rollout_mode=Default AND Think mode is on, because the metrics
            # path requires num_return_thinking_sequences=1).
            with gr.Row(elem_classes="switch-container"):
                cot_metrics_switch = gr.Checkbox(
                    label="Compute CoT Metrics (Default + Think only)",
                    value=False,
                    elem_id="cot-metrics-switch",
                    elem_classes=["ios-switch"]
                )

            preview_df = gr.DataFrame(
                headers=["Model Path", "Think", "Output Suffix", "Task", "Status"],
                column_widths=["35%", "80px", "20%", "25%", "150px"],
                interactive=True,
                wrap=True,
                col_count=(5, "fixed"),
                row_count=(0, "dynamic"),
                elem_classes=["preview-df", "preview-table"],
                value=pd.DataFrame(columns=["Model Path", "Think", "Output Suffix", "Task", "Status"])
            )

            with gr.Row():
                cancel_btn = gr.Button("取消", elem_classes=["cancel-button"])
                submit_btn = gr.Button("提交任务", elem_classes=["submit-button"])

        # Queue status area
        with gr.Column(elem_classes="queue-area"):
            with gr.Row(elem_classes="queue-title-row"):
                # Using a single HTML block for both title and buttons to ensure perfect flex control
                # or relying on Gradio Row's flex behavior if we put them as separate components.
                # Since Gradio Row adds extra divs, let's try to be explicit with CSS on the Row class.
                # But to be safe, let's put the title in Markdown and buttons in HTML, 
                # and the CSS .queue-title-row { justify-content: space-between; } should handle it 
                # IF Gradio doesn't wrap them in full-width containers.
                
                # Actually, Gradio Row children often expand. 
                # Let's use a single HTML block for the header row to guarantee layout control.
                
                # Hidden buttons for logic (keep outside the visual row or hidden)
                refresh_btn_hidden = gr.Button(visible=False, elem_id="refresh-btn-hidden")
                delete_btn_hidden = gr.Button(visible=False, elem_id="delete-btn-hidden")

                # Load icons
                refresh_icon_svg = load_svg_icon(REFRESH_ICON)
                delete_icon_svg = load_svg_icon(DELETE_ICON)

                gr.HTML(f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                        <h2 style="margin: 0; font-size: 1.5em; font-weight: bold; color: #171717;">📊 任务队列状态</h2>
                        <div class="queue-actions" style="display: flex; gap: 8px; align-items: center;">
                            <button class="icon-button" onclick="document.getElementById('refresh-btn-hidden').click()" title="刷新">
                                {refresh_icon_svg}
                            </button>
                            <button class="icon-button" onclick="document.getElementById('delete-btn-hidden').click()" title="删除选中">
                                {delete_icon_svg}
                            </button>
                        </div>
                    </div>
                """)

            # Hidden input box for receiving JS returned data
            with gr.Row(visible=False):
                delete_task_ids_input = gr.Textbox(elem_id="delete-task-ids-input")

            with gr.Tabs():
                with gr.Tab(label="⏳ 等待中"):
                    with gr.Row():
                        with gr.Column():
                            pending_status_df = gr.HTML(
                                elem_classes=["queue-df", "pending-df", "deletable-df"],
                                elem_id="pending-df"
                            )
                with gr.Tab(label="▶️ 运行中"):
                    with gr.Row():
                        with gr.Column():
                            processing_status_df = gr.HTML(
                                elem_classes=["queue-df", "processing-df"],
                                elem_id="processing-df"
                            )
                with gr.Tab(label="✅ 已完成"):
                    with gr.Row():
                        with gr.Column():
                            done_status_df = gr.HTML(
                                elem_classes=["queue-df", "done-df", "deletable-df"],
                                elem_id="done-df"
                            )
                with gr.Tab(label="❌ 已失败"):
                    with gr.Row():
                        with gr.Column():
                            failed_status_df = gr.HTML(
                                elem_classes=["queue-df", "failed-df", "deletable-df"],
                                elem_id="failed-df"
                            )

        status_outputs = [pending_status_df, processing_status_df, done_status_df, failed_status_df]

        # Event handlers
        # "New Task" button
        new_task_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            inputs=None,
            outputs=[new_task_btn, submit_area]
        )

        # Model type switch handler
        def on_model_type_change(model_type):
            if model_type == "open_source":
                meta = get_tasks_metadata("v3.1")
                return (
                    gr.update(visible=True),   # open_source_area
                    gr.update(visible=False),  # closed_source_area
                    gr.update(visible=True),   # think_rollout_area
                    "v3.1",                    # version_radio
                    meta["default_selected"].copy(),
                    generate_task_pills_html(meta["all_tasks"], meta["default_selected"]),
                    pd.DataFrame(columns=["Model Path", "Think", "Output Suffix", "Task", "Status"]),
                )
            else:
                meta = get_tasks_metadata("v3.1")
                # For closed-source, default to ALL tasks (since show_in_web_ui may not be set)
                all_tasks = meta["all_tasks"]
                default = all_tasks if not meta["default_selected"] else meta["default_selected"]
                return (
                    gr.update(visible=False),  # open_source_area
                    gr.update(visible=True),   # closed_source_area
                    gr.update(visible=False),  # think_rollout_area
                    "v3.1",                    # version_radio
                    default.copy(),
                    generate_task_pills_html(all_tasks, default),
                    pd.DataFrame(columns=["Model Path", "Think", "Output Suffix", "Task", "Status"]),
                )

        model_type_radio.change(
            fn=on_model_type_change,
            inputs=model_type_radio,
            outputs=[open_source_area, closed_source_area, think_rollout_area, version_radio, selected_tasks, task_pills_html, preview_df]
        )

        # Closed-source model dropdown change -> update preview
        api_model_dropdown.change(
            fn=lambda api_model, tasks, ver, seed: parse_api_model_to_preview(api_model, tasks, ver, int(seed)),
            inputs=[api_model_dropdown, selected_tasks, version_radio, seed_radio],
            outputs=preview_df
        )

        # "Cancel" button
        def reset_form(current_version):
            meta = get_tasks_metadata(current_version)
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                "open_source",
                "",
                "claude",
                1,
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                "Non Think",
                "Race",
                "v3.1",
                42,
                meta["default_selected"].copy(),
                generate_task_pills_html(meta["all_tasks"], meta["default_selected"]),
                pd.DataFrame(columns=["Model Path", "Think", "Output Suffix", "Task", "Status"])
            )

        cancel_btn.click(
            fn=reset_form,
            inputs=version_state,
            outputs=[submit_area, new_task_btn, model_type_radio, model_paths_text, api_model_dropdown,
                     tensor_parallel_size_dropdown,
                     open_source_area, closed_source_area, think_rollout_area,
                     think_mode_radio, rollout_mode_radio,
                     version_radio, seed_radio,
                     selected_tasks, task_pills_html, preview_df]
        )

        # Wrapper function to convert think_mode to boolean
        def preview_with_conversion(username, model_paths_text, think_mode, rollout_mode, selected_tasks, version, seed):
            enable_thinking = think_mode == "Think"
            return parse_paths_to_preview(username, model_paths_text, enable_thinking, rollout_mode, selected_tasks, version, int(seed))

        preview_inputs = [username, model_paths_text, think_mode_radio, rollout_mode_radio, selected_tasks, version_radio, seed_radio]

        # Path input change
        model_paths_text.change(
            fn=preview_with_conversion,
            inputs=preview_inputs,
            outputs=preview_df
        )

        # Think mode radio change
        think_mode_radio.change(
            fn=preview_with_conversion,
            inputs=preview_inputs,
            outputs=preview_df
        )

        # Rollout mode radio change
        rollout_mode_radio.change(
            fn=preview_with_conversion,
            inputs=preview_inputs,
            outputs=preview_df
        )

        # Version radio change - update task pills and preview
        def on_version_change(version):
            meta = get_tasks_metadata(version)
            pills_html = generate_task_pills_html(meta["all_tasks"], [])
            return [], pills_html

        def preview_after_task_toggle(username, model_paths_text, think_mode, rollout_mode,
                                     selected_tasks, version, seed, model_type, api_model):
            if model_type == "closed_source":
                return parse_api_model_to_preview(api_model, selected_tasks, version, int(seed))
            else:
                enable_thinking = think_mode == "Think"
                return parse_paths_to_preview(username, model_paths_text, enable_thinking, rollout_mode, selected_tasks, version, int(seed))

        version_radio.change(
            fn=on_version_change,
            inputs=version_radio,
            outputs=[selected_tasks, task_pills_html]
        ).then(
            fn=preview_after_task_toggle,
            inputs=[username, model_paths_text, think_mode_radio, rollout_mode_radio,
                    selected_tasks, version_radio, seed_radio, model_type_radio, api_model_dropdown],
            outputs=preview_df
        )

        # Seed radio change
        seed_radio.change(
            fn=preview_after_task_toggle,
            inputs=[username, model_paths_text, think_mode_radio, rollout_mode_radio,
                    selected_tasks, version_radio, seed_radio, model_type_radio, api_model_dropdown],
            outputs=preview_df
        )

        # Task pill click handler
        def toggle_task(task_name: str, current_tasks: list, version: str):
            version_tasks = get_tasks_metadata(version)["all_tasks"]
            if not task_name:
                return current_tasks, generate_task_pills_html(version_tasks, current_tasks)

            new_tasks = current_tasks.copy()
            if task_name in new_tasks:
                new_tasks.remove(task_name)
            else:
                new_tasks = [t for t in version_tasks if t in new_tasks or t == task_name]

            return new_tasks, generate_task_pills_html(version_tasks, new_tasks)

        task_click_input.change(
            fn=toggle_task,
            inputs=[task_click_input, selected_tasks, version_radio],
            outputs=[selected_tasks, task_pills_html]
        ).then(
            fn=preview_after_task_toggle,
            inputs=[username, model_paths_text, think_mode_radio, rollout_mode_radio,
                    selected_tasks, version_radio, seed_radio, model_type_radio, api_model_dropdown],
            outputs=preview_df
        )

        # "Submit" button - handles both open-source and closed-source
        def submit_with_conversion(username, preview_df, think_mode, rollout_mode,
                                   selected_tasks, overwrite, cot_metrics, version, seed,
                                   tensor_parallel_size, model_type, api_model):
            if model_type == "closed_source":
                return submit_api_tasks(username, api_model, selected_tasks, overwrite, version, int(seed))
            else:
                enable_thinking = think_mode == "Think"
                # CoT metrics only flow through when rollout_mode == "Default"
                # and Think is on — other rollouts use num_return_thinking != 1
                # and the metrics path is skipped with a warning.
                effective_cot_metrics = bool(cot_metrics) and enable_thinking and rollout_mode == "Default"
                return submit_tasks(username, preview_df, enable_thinking, rollout_mode, selected_tasks, overwrite, version, int(seed), compute_cot_metrics=effective_cot_metrics, tensor_parallel_size=int(tensor_parallel_size))

        submit_btn.click(
            fn=submit_with_conversion,
            inputs=[username, preview_df, think_mode_radio, rollout_mode_radio,
                    selected_tasks, overwrite_switch, cot_metrics_switch, version_radio, seed_radio,
                    tensor_parallel_size_dropdown, model_type_radio, api_model_dropdown],
            outputs=[submit_area, new_task_btn] + status_outputs
        )


        # "Refresh" button
        refresh_btn_hidden.click(
            fn=get_queue_status,
            inputs=None,
            outputs=status_outputs
        )

        # Delete button event
        delete_btn_hidden.click(
            fn=delete_selected_tasks,
            inputs=delete_task_ids_input,
            outputs=status_outputs,
            js="() => getSelectedTaskIds()"
        )

    return submit_page, status_outputs
