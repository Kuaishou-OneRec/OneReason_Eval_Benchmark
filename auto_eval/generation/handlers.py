"""
Handlers for the generation page
"""
import json
import uuid
import gradio as gr
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from auto_eval.config import GENERATION_DIRS


def submit_generation_task(
    model_path: str,
    prompts_text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    num_return_sequences: int,
    repetition_penalty: float
) -> str:
    """
    Submit a generation task to the incoming queue.

    Args:
        model_path: Path to the model
        prompts_text: Multiple prompts separated by newlines
        ... other generation parameters

    Returns:
        task_id: The unique ID of the submitted task
    """
    # Parse models (split by newlines, filter empty lines)
    models = [m.strip() for m in model_path.strip().split('\n') if m.strip()]

    if not models:
        raise ValueError("Model path is required")

    # Parse prompts (split by newlines, filter empty lines)
    # Then convert literal \n (two chars) to actual newline char
    prompts = [p.strip().replace('\\n', '\n') for p in prompts_text.strip().split('\n') if p.strip()]

    if not prompts:
        raise ValueError("No prompts provided")

    # Generate unique task ID
    task_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Build task config (compatible with vLLM input format)
    task_config = {
        "task_id": task_id,
        "models": models,  # vLLM expects a list
        "prompts": prompts,
        "params": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": int(max_new_tokens),
            "num_return_sequences": int(num_return_sequences),
            "repetition_penalty": repetition_penalty
        },
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "max_retries": 3
    }

    # Save to incoming directory
    incoming_dir = GENERATION_DIRS["incoming"]
    incoming_dir.mkdir(parents=True, exist_ok=True)

    task_file = incoming_dir / f"{task_id}.json"
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump(task_config, f, ensure_ascii=False, indent=4)

    return task_id


def find_task_file(task_id: str) -> Optional[Tuple[Path, str]]:
    """
    Find the task file across all queue directories.

    Note: Check done/failed first to handle NFS cache issues where
    files may appear to still exist in processing after being moved.

    Returns:
        Tuple of (file_path, status) or None if not found
    """
    # Priority order: done, failed, processing, incoming
    # This handles NFS cache issues where moved files may still appear in source dir
    check_order = ["done", "failed", "processing", "incoming"]
    for status in check_order:
        dir_path = GENERATION_DIRS.get(status)
        if dir_path:
            task_file = dir_path / f"{task_id}.json"
            if task_file.exists():
                return task_file, status
    return None


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the current status and results of a task.

    Returns:
        Dict containing status, results (if completed), and error info (if failed)
    """
    result = find_task_file(task_id)

    if result is None:
        return {"status": "not_found", "message": f"Task {task_id} not found"}

    task_file, queue_status = result

    try:
        with open(task_file, 'r', encoding='utf-8') as f:
            task_config = json.load(f)

        return {
            "status": task_config.get("status", queue_status),
            "queue_status": queue_status,
            "task_id": task_id,
            "results": task_config.get("results"),
            "error_log": task_config.get("error_log"),
            "created_at": task_config.get("created_at"),
            "started_at": task_config.get("started_at"),
            "finished_at": task_config.get("finished_at"),
            "prompts": task_config.get("prompts", []),
            "params": task_config.get("params", {})
        }
    except Exception as e:
        return {"status": "error", "message": f"Error reading task file: {e}"}


def submit_and_show_status(
    model_path: str,
    prompts_text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    num_return_sequences: int,
    repetition_penalty: float
) -> Tuple[gr.update, gr.update, gr.update, str]:
    """
    Submit task and return initial status.
    Returns updates for: start_btn, result_output, refresh_btn, task_id_state
    """
    # Validate inputs
    if not model_path.strip():
        return (
            gr.update(value="开始生成", interactive=True),
            gr.update(value="<div class='error-message'>错误: 请输入模型路径</div>"),
            gr.update(visible=False),
            ""
        )

    if not prompts_text.strip():
        return (
            gr.update(value="开始生成", interactive=True),
            gr.update(value="<div class='error-message'>错误: 请输入至少一个 prompt</div>"),
            gr.update(visible=False),
            ""
        )

    # Submit task
    try:
        task_id = submit_generation_task(
            model_path, prompts_text,
            temperature, top_p, top_k, max_new_tokens, num_return_sequences, repetition_penalty
        )
    except Exception as e:
        return (
            gr.update(value="开始生成", interactive=True),
            gr.update(value=f"<div class='error-message'>提交任务失败: {e}</div>"),
            gr.update(visible=False),
            ""
        )

    # Return initial status with refresh button visible
    status_html = f"""
    <div class='status-message'>
        <div class='task-info'>
            <span class='task-label'>任务ID:</span> <code>{task_id}</code>
        </div>
        <div class='task-status'>状态: 等待处理...</div>
    </div>
    """
    return (
        gr.update(value="开始生成", interactive=True),
        gr.update(value=status_html),
        gr.update(visible=True),
        task_id
    )


def refresh_task_status(task_id: str) -> Tuple[gr.update, gr.update]:
    """
    Manually refresh task status.
    Returns updates for: result_output, refresh_btn
    """
    if not task_id:
        return (
            gr.update(value="<div class='error-message'>没有正在进行的任务</div>"),
            gr.update(visible=False)
        )

    status_info = get_task_status(task_id)
    status = status_info.get("status")
    queue_status = status_info.get("queue_status", "unknown")

    print(f"[DEBUG] Refresh - Task {task_id}: status={status}, queue_status={queue_status}")

    if status == "completed":
        # Task completed successfully
        results = status_info.get("results", {})
        params = status_info.get("params", {})

        results_data = {
            "results": results,
            "params": params,
        }

        try:
            result_html = format_generation_results(results_data)
        except Exception as e:
            print(f"[DEBUG] Error formatting results: {e}")
            import traceback
            traceback.print_exc()
            result_html = f"<div class='error-message'>格式化结果时出错: {e}</div>"

        return (
            gr.update(value=result_html),
            gr.update(visible=False)  # Hide refresh button after completion
        )

    elif status == "failed":
        error_log = status_info.get("error_log", {})
        error_html = format_error_message(task_id, error_log)
        return (
            gr.update(value=error_html),
            gr.update(visible=False)
        )

    elif status == "not_found":
        return (
            gr.update(value=f"<div class='error-message'>任务未找到: {task_id}</div>"),
            gr.update(visible=False)
        )

    else:
        # Still processing
        status_text = "处理中" if queue_status == "processing" else "等待中"
        status_html = f"""
        <div class='status-message'>
            <div class='task-info'>
                <span class='task-label'>任务ID:</span> <code>{task_id}</code>
            </div>
            <div class='task-status'>状态: {status_text}</div>
        </div>
        """
        return (
            gr.update(value=status_html),
            gr.update(visible=True)
        )


def format_error_message(task_id: str, error_log: Dict) -> str:
    """Format error message for display"""
    import html

    error_parts = [f"<div class='error-container'>"]
    error_parts.append(f"<h3 style='color: #d62728;'>任务失败: {task_id}</h3>")

    if error_log:
        if 'stderr' in error_log:
            stderr = error_log.get('stderr', '')[:1000]  # Truncate
            error_parts.append(f"<h4>错误信息:</h4><pre>{html.escape(stderr)}</pre>")

        if 'exception' in error_log:
            error_parts.append(f"<p><strong>异常类型:</strong> {html.escape(str(error_log.get('exception_type', 'Unknown')))}</p>")
            error_parts.append(f"<p><strong>异常信息:</strong> {html.escape(str(error_log.get('exception', '')))}</p>")

        if 'sentinel_error' in error_log:
            sentinel_err = error_log['sentinel_error']
            error_parts.append(f"<p><strong>退出码:</strong> {sentinel_err.get('exit_code', 'N/A')}</p>")

    error_parts.append("</div>")
    return "".join(error_parts)

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to rgba string"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def format_generation_results(results_data: Dict[str, Any]) -> str:
    """
    Format generation results for display with model comparison view.
    Groups results by sample_id (prompt index) and shows models side by side.
    """
    import html as html_module

    results = results_data.get("results", {})

    if not results:
        return "<p>无生成结果</p>"

    # Define model colors
    model_colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Get all model names and assign colors
    model_names = list(results.keys())
    model_color_map = {name: model_colors_list[i % len(model_colors_list)]
                       for i, name in enumerate(model_names)}

    # Reorganize data by sample_id (prompt index)
    # Find the number of samples (assuming all models have the same number of prompts)
    num_samples = len(next(iter(results.values()))) if results else 0

    # Build comparisons: [{sample_id, models: {model_name: {prompt, generation}}}]
    comparisons = []
    for sample_idx in range(num_samples):
        sample_data = {
            "sample_id": sample_idx,
            "models": {}
        }
        for model_name, samples in results.items():
            if sample_idx < len(samples):
                sample = samples[sample_idx]
                prompt_text = sample.get("prompt", "")
                generated_texts = sample.get("generated_texts", [])
                # Merge multiple generations with separator
                generation_text = "\n##########\n".join(generated_texts) if generated_texts else ""
                sample_data["models"][model_name] = {
                    "prompt": prompt_text,
                    "generation": generation_text
                }
        comparisons.append(sample_data)

    html_parts = []

    # 1. Header with model filter pills
    html_parts.append("<div class='gen-comparison-header'>")
    html_parts.append("<h2 style='margin: 0 20px 0 0;'>生成结果对比</h2>")
    html_parts.append("<div class='gen-model-filter'>")

    for model_name in model_names:
        model_color = model_color_map[model_name]
        light_color = hex_to_rgba(model_color, 0.2)
        safe_model_id = model_name.replace(' ', '_').replace('/', '_')
        html_parts.append(f"""
        <div class='gen-model-pill'
             style='background-color: {light_color}; border-left: 3px solid {model_color};'
             id='pill_{safe_model_id}'
             onclick="toggleGenModel('{safe_model_id}')"
             title="点击隐藏/显示此模型">
            <span>{model_name}</span>
        </div>
        """)

    html_parts.append("</div>")  # End filter
    html_parts.append("</div>")  # End header

    # Copy icon SVG
    copy_icon_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>'

    # 2. Sample comparisons
    for comp in comparisons:
        sample_id = comp["sample_id"]
        models = comp["models"]

        # Sample container (collapsible)
        html_parts.append(f"<details open class='gen-sample-details'>")
        html_parts.append(f"<summary class='gen-sample-summary'>Sample {sample_id}</summary>")
        html_parts.append("<div class='gen-sample-content'>")

        # Model cards in a flex row
        html_parts.append("<div class='gen-model-cards'>")

        for model_name, data in models.items():
            model_color = model_color_map[model_name]
            safe_model_id = model_name.replace(' ', '_').replace('/', '_')
            prompt_text = data["prompt"]
            generation_text = data["generation"]

            # Unique IDs
            prompt_id = f"gen_p_{sample_id}_{safe_model_id}"
            gen_id = f"gen_g_{sample_id}_{safe_model_id}"

            # Single quote for onclick
            sq = "'"

            # Prompt expand/collapse
            prompt_preview = prompt_text[:500] if len(prompt_text) > 500 else prompt_text
            prompt_expand = f'<span class="gen-expand-link" onclick="toggleGenText({sq}{prompt_id}{sq})">展开</span>' if len(prompt_text) > 500 else ''
            prompt_collapse = f'<span class="gen-collapse-link" onclick="toggleGenText({sq}{prompt_id}{sq})">收起</span>'

            # Generation expand/collapse
            gen_preview = generation_text[:500] if len(generation_text) > 500 else generation_text
            gen_expand = f'<span class="gen-expand-link" onclick="toggleGenText({sq}{gen_id}{sq})">展开</span>' if len(generation_text) > 500 else ''
            gen_collapse = f'<span class="gen-collapse-link" onclick="toggleGenText({sq}{gen_id}{sq})">收起</span>'

            html_parts.append(f"""
            <div class='gen-model-card' data-model='{safe_model_id}' style='border-top: 3px solid {model_color};'>
                <h4 class='gen-model-title' style='color: {model_color};'>{model_name}</h4>

                <details open>
                    <summary class='gen-section-title'>Prompt</summary>
                    <div class='gen-text-container'>
                        <button class="gen-copy-btn" onclick="copyGenText('{prompt_id}')" title="复制">
                            {copy_icon_svg}
                        </button>
                        <div class='gen-text-preview' id='{prompt_id}_preview'>
                            <pre>{html_module.escape(prompt_preview)}{'...' if len(prompt_text) > 500 else ''}</pre>
                            {prompt_expand}
                        </div>
                        <div class='gen-text-full' id='{prompt_id}_full' style='display: none;'>
                            <pre>{html_module.escape(prompt_text)}</pre>
                            {prompt_collapse}
                        </div>
                    </div>
                </details>

                <details open>
                    <summary class='gen-section-title'>Generation</summary>
                    <div class='gen-text-container'>
                        <button class="gen-copy-btn" onclick="copyGenText('{gen_id}')" title="复制">
                            {copy_icon_svg}
                        </button>
                        <div class='gen-text-preview' id='{gen_id}_preview'>
                            <pre>{html_module.escape(gen_preview)}{'...' if len(generation_text) > 500 else ''}</pre>
                            {gen_expand}
                        </div>
                        <div class='gen-text-full' id='{gen_id}_full' style='display: none;'>
                            <pre>{html_module.escape(generation_text)}</pre>
                            {gen_collapse}
                        </div>
                    </div>
                </details>
            </div>
            """)

        html_parts.append("</div>")  # End model cards
        html_parts.append("</div>")  # End sample content
        html_parts.append("</details>")  # End sample details

    return "".join(html_parts)
