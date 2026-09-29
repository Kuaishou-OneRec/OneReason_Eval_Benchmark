"""
UI definition for the generation page
"""
import gradio as gr
from .styles import GENERATION_CSS, GENERATION_JS
from .handlers import submit_and_show_status, refresh_task_status

def create_generation_ui():
    """
    Create the generation UI components
    """
    with gr.Column(elem_classes="generation-container"):
        # Model Path
        model_path_input = gr.Textbox(
            label="模型路径",
            placeholder="/path/to/model",
            lines=2,
            elem_classes="model-path-input"
        )

        # Parameters
        with gr.Row(elem_classes="params-container"):
            with gr.Column():
                temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="temperature")
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="top_p")
                top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="top_k")

            with gr.Column():
                max_new_tokens = gr.Slider(minimum=1, maximum=4096, value=512, step=1, label="max_new_tokens")
                num_return_sequences = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="num_return_sequences")
                repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.0, step=0.1, label="repetition_penalty")

        # Prompt
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="请输入提示词，每行一个 prompt...\n例如：\n请介绍一下人工智能的发展历程。\n什么是深度学习？请简要说明。",
            lines=10,
            elem_id="prompt-input",
            elem_classes="prompt-input"
        )

        # Template Shortcuts
        gr.HTML("""
            <div class="template-shortcuts">
                <span class="template-link" onclick="appendTemplate('think')">chat_template 格式（think）</span>
                <span style="color: #ccc;">|</span>
                <span class="template-link" onclick="appendTemplate('no_think')">chat_template 格式（non think）</span>
                <span style="color: #ccc;">|</span>
                <span class="template-link" onclick="appendTemplate('sid')">sid 格式</span>
            </div>
        """)

        # Start Button
        start_btn = gr.Button("开始生成", elem_classes="start-btn")

        # Output Area with Refresh Button
        with gr.Row(elem_classes="result-row"):
            result_output = gr.HTML(label="生成结果", elem_classes="result-container")

        # Refresh Button (initially hidden)
        refresh_btn = gr.Button("🔄 刷新状态", elem_classes="refresh-btn", visible=False)

        # Hidden state to store current task_id
        task_id_state = gr.State(value="")

        # Event Handling
        # Submit task
        start_btn.click(
            fn=submit_and_show_status,
            inputs=[
                model_path_input, prompt_input,
                temperature, top_p, top_k, max_new_tokens, num_return_sequences, repetition_penalty
            ],
            outputs=[start_btn, result_output, refresh_btn, task_id_state]
        )

        # Refresh status
        refresh_btn.click(
            fn=refresh_task_status,
            inputs=[task_id_state],
            outputs=[result_output, refresh_btn]
        )

    return (
        model_path_input,
        temperature, top_p, top_k, max_new_tokens, num_return_sequences, repetition_penalty,
        prompt_input,
        start_btn,
        result_output,
        refresh_btn,
        task_id_state
    )
