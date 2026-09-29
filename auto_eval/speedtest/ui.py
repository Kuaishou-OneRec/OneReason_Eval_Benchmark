"""
Gradio UI for Speed Test page.
"""
import gradio as gr
from speedtest.handlers import (
    submit_speed_task,
    get_speedtest_queue_status,
    scan_speed_results,
    estimate_resources,
)
from auto_eval.tasks_meta import get_tasks_metadata

DEFAULT_VERSION = "v3.1"
_meta = get_tasks_metadata(DEFAULT_VERSION)
ALL_TASKS = _meta["all_tasks"]

GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_BATCH = "128"


def _get_task_beam_defaults() -> dict:
    """Read per-task num_beams from task configs in the registry."""
    from benchmark.tasks.v3_1.registry import TASK_REGISTRY
    defaults = {}
    for name, tr in TASK_REGISTRY.items():
        gen_cfg = tr.config.get("generation_config", {})
        beam = gen_cfg.get("num_beams")
        defaults[name] = beam if beam else 1
    return defaults


TASK_BEAM_DEFAULTS = _get_task_beam_defaults()


def _group_tasks_by_beam(task_names):
    """Group selected tasks by their default beam value."""
    groups = {}
    for t in task_names:
        beam = TASK_BEAM_DEFAULTS.get(t, 1)
        groups.setdefault(beam, []).append(t)
    return groups


def _build_config_preview(task_names):
    """Build HTML preview of how tasks will be grouped and submitted."""
    if not task_names:
        return ""
    groups = _group_tasks_by_beam(task_names)
    html = "<div style='font-size:13px; margin-top:8px;'>"
    html += f"<b>已选 {len(task_names)} 个任务，将按 beam 值拆分提交：</b><br/>"
    for beam, tasks in sorted(groups.items()):
        html += (
            f"<span style='color:#1976d2;'>[beam={beam}]</span> "
            f"{len(tasks)} 个任务: {', '.join(tasks)}<br/>"
        )
    html += "</div>"
    return html


def _build_results_html(results):
    """Build HTML table from speed test results."""
    if not results:
        return "<p style='color:#999; text-align:center;'>暂无测速结果</p>"

    html = """<table style='width:100%; border-collapse:collapse; font-size:13px;'>
    <thead><tr style='background:#f0f0f0;'>
        <th style='padding:8px; border:1px solid #ddd;'>模型</th>
        <th style='padding:8px; border:1px solid #ddd;'>任务</th>
        <th style='padding:8px; border:1px solid #ddd;'>Mode</th>
        <th style='padding:8px; border:1px solid #ddd;'>Beam</th>
        <th style='padding:8px; border:1px solid #ddd;'>Batch</th>
        <th style='padding:8px; border:1px solid #ddd;'>Wall Time</th>
        <th style='padding:8px; border:1px solid #ddd;'>Infer Time</th>
        <th style='padding:8px; border:1px solid #ddd;'>Decode tok/s</th>
        <th style='padding:8px; border:1px solid #ddd;'>Peak GPU MB</th>
        <th style='padding:8px; border:1px solid #ddd;'>Status</th>
    </tr></thead><tbody>"""

    for r in results:
        status_color = "#4caf50" if r["result"] == "OK" else "#f44336"
        html += f"""<tr>
            <td style='padding:6px; border:1px solid #ddd;'>{r['model_tag']}</td>
            <td style='padding:6px; border:1px solid #ddd;'>{r['task']}</td>
            <td style='padding:6px; border:1px solid #ddd;'>{r['mode']}</td>
            <td style='padding:6px; border:1px solid #ddd;'>{r['beam']}</td>
            <td style='padding:6px; border:1px solid #ddd;'>{r['batch']}</td>
            <td style='padding:6px; border:1px solid #ddd;'>{r['wall_time_s']:.1f}s</td>
            <td style='padding:6px; border:1px solid #ddd;'>{r['infer_time_s']:.1f}s</td>
            <td style='padding:6px; border:1px solid #ddd;'>{r['decode_tok_per_s']:.1f}</td>
            <td style='padding:6px; border:1px solid #ddd;'>{r['peak_gpu_mem_mb']:.0f}</td>
            <td style='padding:6px; border:1px solid #ddd; color:{status_color};'>{r['result']}</td>
        </tr>"""
    html += "</tbody></table>"
    return html


def _build_estimation_html(estimations, sample_count, target_count):
    """Build HTML table for resource estimation."""
    if not estimations:
        return "<p style='color:#999;'>请先选择一条测速结果</p>"

    html = f"<p><b>Benchmark 样本数:</b> {sample_count:,} | <b>目标数据量:</b> {target_count:,}</p>"
    html += """<table style='width:100%; border-collapse:collapse; font-size:13px;'>
    <thead><tr style='background:#e3f2fd;'>
        <th style='padding:8px; border:1px solid #ddd;'>GPU 数</th>
        <th style='padding:8px; border:1px solid #ddd;'>Wall Time 估算（保守）</th>
        <th style='padding:8px; border:1px solid #ddd;'>Infer Time 估算（乐观）</th>
    </tr></thead><tbody>"""

    for est in estimations:
        html += f"""<tr>
            <td style='padding:6px; border:1px solid #ddd; text-align:center;'><b>{est['num_gpus']}</b></td>
            <td style='padding:6px; border:1px solid #ddd; text-align:center;'>{est['wall_est_display']}</td>
            <td style='padding:6px; border:1px solid #ddd; text-align:center;'>{est['infer_est_display']}</td>
        </tr>"""
    html += "</tbody></table>"
    return html


def create_speedtest_ui():
    """Create the Speed Test page UI."""
    with gr.Column() as speedtest_page:
        gr.Markdown("<h1 style='text-align:center; margin-top:20px;'>Speed Test</h1>")

        with gr.Row():
            # ---- Left: Submit ----
            with gr.Column(scale=1):
                gr.Markdown("### 提交测速任务")
                model_path_input = gr.Textbox(label="模型路径", placeholder="/path/to/model")
                model_tag_input = gr.Textbox(label="模型标签", placeholder="如: 8b, 0.6b")
                task_input = gr.CheckboxGroup(
                    choices=ALL_TASKS,
                    label="任务（自动按 beam 值分组提交，beam 值来自任务配置）",
                    value=[],
                )
                config_preview = gr.HTML("")
                mode_input = gr.Dropdown(choices=["nothink", "think"], label="Mode", value="nothink")
                gpu_mode_input = gr.Radio(choices=["single", "multi"], label="GPU 模式", value="single")
                gpu_ids_input = gr.Textbox(label="GPU IDs", value="0", placeholder="单卡填 GPU ID；多卡填空格分隔的 IDs")

                submit_btn = gr.Button("提交测速任务", variant="primary")
                submit_status = gr.Markdown("")
                queue_status = gr.Markdown("")

            # ---- Middle: Results ----
            with gr.Column(scale=2):
                gr.Markdown("### 测速结果")
                refresh_btn = gr.Button("刷新结果")
                results_html = gr.HTML("<p style='color:#999; text-align:center;'>点击刷新查看结果</p>")

                results_state = gr.State([])

                gr.Markdown("### 选择基准结果用于估算")
                result_selector = gr.Dropdown(choices=[], label="选择测速结果", value=None)

            # ---- Right: Estimation ----
            with gr.Column(scale=1):
                gr.Markdown("### 资源估算")
                sample_count_input = gr.Number(label="Benchmark 样本数", value=574, precision=0)
                target_count_input = gr.Number(label="目标数据量", value=20000000, precision=0)
                estimate_btn = gr.Button("计算估算", variant="secondary")
                estimation_html = gr.HTML("")

        # ---- Event handlers ----
        task_input.change(
            fn=_build_config_preview,
            inputs=[task_input],
            outputs=[config_preview],
        )

        def on_submit(model_path, model_tag, tasks, mode, gpu_mode, gpu_ids):
            if not tasks:
                return "❌ 请选择至少一个任务", get_speedtest_queue_status()

            groups = _group_tasks_by_beam(tasks)
            messages = []
            for beam, beam_tasks in sorted(groups.items()):
                tasks_str = " ".join(beam_tasks)
                msg = submit_speed_task(
                    model_path, model_tag, tasks_str,
                    str(beam), DEFAULT_BATCH,
                    mode, gpu_mode, gpu_ids,
                )
                messages.append(f"[beam={beam}] {msg}")

            queue = get_speedtest_queue_status()
            return "\n\n".join(messages), queue

        submit_btn.click(
            fn=on_submit,
            inputs=[model_path_input, model_tag_input, task_input, mode_input, gpu_mode_input, gpu_ids_input],
            outputs=[submit_status, queue_status],
        )

        def on_refresh():
            results = scan_speed_results()
            html = _build_results_html(results)
            choices = [
                f"{r['model_tag']} | {r['task']} | beam={r['beam']} | batch={r['batch']} (wall={r['wall_time_s']:.1f}s, infer={r['infer_time_s']:.1f}s)"
                for r in results
            ]
            queue = get_speedtest_queue_status()
            return html, results, gr.update(choices=choices, value=None), queue

        refresh_btn.click(
            fn=on_refresh,
            inputs=None,
            outputs=[results_html, results_state, result_selector, queue_status],
        )

        def on_estimate(selector_value, results, sample_count, target_count):
            if not selector_value or not results:
                return "<p style='color:#999;'>请先选择一条测速结果</p>"
            try:
                idx = next(
                    i for i, r in enumerate(results)
                    if f"{r['model_tag']} | {r['task']} | beam={r['beam']} | batch={r['batch']}" in selector_value
                )
            except StopIteration:
                return "<p style='color:#f44336;'>未找到匹配的结果</p>"

            r = results[idx]
            estimations = estimate_resources(
                sample_count=int(sample_count),
                wall_time_s=r["wall_time_s"],
                infer_time_s=r["infer_time_s"],
                target_count=int(target_count),
                gpu_counts=GPU_COUNTS,
            )
            return _build_estimation_html(estimations, int(sample_count), int(target_count))

        estimate_btn.click(
            fn=on_estimate,
            inputs=[result_selector, results_state, sample_count_input, target_count_input],
            outputs=[estimation_html],
        )

    return speedtest_page, {"queue_status": queue_status}
