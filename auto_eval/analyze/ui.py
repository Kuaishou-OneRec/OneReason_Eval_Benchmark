"""
Gradio UI components for results analysis
"""
import gradio as gr
import pandas as pd
from typing import List, Tuple, Dict

from auto_eval.tasks_meta import get_all_versions
from analyze.handlers import (
    get_result_directories,
    load_eval_results,
    load_test_generated_samples,
    compare_samples,
    format_comparison_for_display,
    dataframe_to_html,
    get_loading_html
)
from analyze.plot import generate_radar_chart_and_table
from config import RESULT_ROOTS, EVAL_RESULTS_FILENAME, EVAL_RESULTS_SIDMODEL_FILENAME, COPY_ICON, REFRESH_ICON, EXTRA_VERSIONS
from utils import load_svg_icon


def refresh_directories():
    """Refresh the list of result directories"""
    directories = get_result_directories()
    return gr.update(choices=directories, value=[])


def generate_analysis(selected_dirs: List[str], show_sid: bool = True, show_pid: bool = False, grounding: bool = False):
    """
    Generate radar chart and summary table from selected directories

    Args:
        selected_dirs: List of selected directory names
        show_sid: Whether to show SID metrics (default: True)
        show_pid: Whether to show PID metrics (default: False)
        grounding: Whether to use sidmodel eval results (default: False)

    Returns:
        Tuple of (radar_image, summary_table, copy_btn, error_message, model_color_map)
    """
    print(f"[DEBUG] generate_analysis called: show_sid={show_sid!r} (type={type(show_sid).__name__}), "
          f"show_pid={show_pid!r} (type={type(show_pid).__name__}), "
          f"grounding={grounding!r} (type={type(grounding).__name__}), "
          f"selected_dirs count={len(selected_dirs) if selected_dirs else 0}")

    if not selected_dirs:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "",  # copy_btn - empty HTML (hidden by CSS)
            "<p class='error-message'>⚠️ 请先选择至少一个结果目录</p>",
            {}  # empty model_color_map
        )

    try:
        # Choose filename based on grounding mode
        filename = EVAL_RESULTS_SIDMODEL_FILENAME if grounding else EVAL_RESULTS_FILENAME

        # Build full paths to eval results files
        eval_result_paths = []
        for dir_name in selected_dirs:
            # Search across all result roots and versions; first match wins
            for results_root in RESULT_ROOTS:
                found = False
                for version in get_all_versions() + EXTRA_VERSIONS:
                    eval_file = results_root / version / dir_name / filename
                    if eval_file.exists():
                        eval_result_paths.append(str(eval_file))
                        found = True
                        break
                if found:
                    break

        if not eval_result_paths:
            return (
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                "",  # copy_btn - empty HTML (hidden by CSS)
                f"<p class='error-message'>❌ 所选目录中没有找到 {filename} 文件</p>",
                {}  # empty model_color_map
            )

        # Generate radar chart and table
        radar_chart_path, summary_table, model_color_map = generate_radar_chart_and_table(
            eval_result_paths, show_sid=show_sid, show_pid=show_pid, grounding=grounding
        )

        # Add "View" button column to summary table
        # We'll add this as a string column that will be rendered as a button in the UI
        summary_table['查看'] = '查看'

        # Convert to HTML
        summary_html = dataframe_to_html(summary_table)

        return (
            gr.update(value=radar_chart_path, visible=True),
            gr.update(value=summary_html, visible=True),
            f"""<button class="icon-button copy-btn" id="copy-table-btn" onclick="copyTableToClipboard()">
                {load_svg_icon(COPY_ICON)}
            </button>""",
            "",
            model_color_map
        )

    except Exception as e:
        import traceback
        error_msg = f"❌ 生成分析时出错: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "",  # copy_btn - empty HTML (hidden by CSS)
            f"<p class='error-message'>{error_msg}</p>",
            {}  # empty model_color_map
        )


def show_loading_modal(dataset_metric_combined: str):
    """
    Immediately show modal with loading indicator

    Args:
        dataset_metric_combined: Combined string of dataset|||metric

    Returns:
        Tuple of (overlay_visible, modal_visible, modal_content)
    """
    if not dataset_metric_combined:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=""),
            set() # Reset seen samples
        )

    # Show modal with loading animation
    return (
        gr.update(visible=True),   # Show overlay
        gr.update(visible=True),   # Show modal
        gr.update(value=get_loading_html()),  # Show loading content
        set() # Reset seen samples
    )


def show_sample_comparison(selected_dirs: List[str], dataset_metric_combined: str, model_color_map: Dict[str, str] = None, seen_samples: set = None):
    """
    Load and show sample comparison data in modal

    Args:
        selected_dirs: List of selected directory names
        dataset_metric_combined: Combined string of dataset|||metric
        model_color_map: Dictionary mapping model names to colors
        seen_samples: Set of already seen sample IDs

    Returns:
        Tuple of (modal_content, updated_seen_samples)
    """
    if model_color_map is None:
        model_color_map = {}
    if seen_samples is None:
        seen_samples = set()

    try:
        if not dataset_metric_combined:
            return gr.update(value=""), seen_samples

        # Parse combined value to get dataset and metric
        parts = dataset_metric_combined.split('|||')
        dataset_name = parts[0] if len(parts) > 0 else ''
        metric_name = parts[1] if len(parts) > 1 else ''

        if not dataset_name:
            return gr.update(value=""), seen_samples

        # Load test_generated.json for the dataset
        test_generated_data = load_test_generated_samples(selected_dirs, dataset_name)

        if not test_generated_data:
            return gr.update(value=f"<p class='error-message'>未找到 {dataset_name} 的测试数据</p>"), seen_samples

        # Compare samples with history tracking
        comparisons, updated_seen_samples = compare_samples(test_generated_data, seen_sample_ids=seen_samples)

        if not comparisons:
            return gr.update(value=f"<p class='error-message'>没有可比较的样本数据</p>"), updated_seen_samples

        # Format comparison for display with specific metric filter
        comparison_html = format_comparison_for_display(comparisons, model_color_map, metric_filter=metric_name)

        return gr.update(value=comparison_html), updated_seen_samples  # Update modal content and state

    except Exception as e:
        import traceback
        error_msg = f"<p class='error-message'>显示样本对比时出错: {str(e)}\n{traceback.format_exc()}</p>"
        print(error_msg)
        return gr.update(value=error_msg), seen_samples


def hide_modal():
    """Hide the modal overlay and content, and clear the dataset selector"""
    return (
        gr.update(visible=False),  # Hide overlay
        gr.update(visible=False),  # Hide modal
        ""  # Clear selected_dataset_hidden to allow re-triggering
    )


def create_analyze_sidebar():
    with gr.Column(elem_classes="analyze-sidebar") as sidebar:
        search_box = gr.Textbox(
            label=None,
            show_label=False,
            placeholder="搜索词 (逗号或空格分隔)...",
            elem_classes=["search-box", "rounded-textbox"],
            lines=1,
            max_lines=1
        )

        # Exclude box (same style as search box)
        exclude_box = gr.Textbox(
            label=None,
            show_label=False,
            placeholder="排除词 (逗号或空格分隔)...",
            elem_classes=["exclude-box", "rounded-textbox"],
            lines=1,
            max_lines=1
        )

        result_dir_checkboxes = gr.CheckboxGroup(
            choices=get_result_directories(),
            label=None,
            show_label=False, # Explicitly hide label
            interactive=True,
            elem_classes=["checkbox-group", "directory-list"],
            value=[],
        )

        def filter_directories(search_term, exclude_term):
            all_dirs = get_result_directories()
            # Filter by search terms (comma or space separated, must match ALL terms)
            if search_term:
                search_terms = [t.strip().lower() for t in search_term.replace(',', ' ').split() if t.strip()]
                filtered = [d for d in all_dirs if all(term in d.lower() for term in search_terms)]
            else:
                filtered = all_dirs
            # Exclude by exclude terms (comma or space separated)
            if exclude_term:
                exclude_terms = [t.strip().lower() for t in exclude_term.replace(',', ' ').split() if t.strip()]
                filtered = [d for d in filtered if not any(term in d.lower() for term in exclude_terms)]
            return gr.CheckboxGroup(choices=filtered)

        search_box.change(filter_directories, inputs=[search_box, exclude_box], outputs=[result_dir_checkboxes])
        exclude_box.change(filter_directories, inputs=[search_box, exclude_box], outputs=[result_dir_checkboxes])

    return sidebar, result_dir_checkboxes

def create_analyze_content(result_dir_checkboxes):
    with gr.Column(elem_classes="analyze-content", scale=6) as content:
        with gr.Row(elem_classes="header-row"):
            # Title
            # gr.Markdown(
            #     """<h1>OneReason Eval Benchmark Auto Eval</h1>""",
            #     elem_id="analyze-title"
            # )
            # Generate Button
            generate_analysis_btn = gr.Button("生成分析报告", elem_classes="generate-btn")
        
        error_display = gr.HTML(value="", visible=True)

        radar_image = gr.Image(
            label=None,
            show_label=False,
            interactive=False,
            elem_classes="radar-plot",
            visible=False  # Initially hidden
        )

        # SID/PID switches and copy button container
        with gr.Row(elem_classes="switch-and-copy-container"):
            with gr.Row(elem_classes="switch-container"):
                sid_switch = gr.Checkbox(
                    label="Show SID Metrics",
                    value=True,
                    elem_id="sid-switch",
                    elem_classes=["ios-switch"]
                )
                pid_switch = gr.Checkbox(
                    label="Show PID Metrics",
                    value=False,
                    elem_id="pid-switch",
                    elem_classes=["ios-switch"]
                )
                grounding_switch = gr.Checkbox(
                    label="Show Grounding Metrics",
                    value=False,
                    elem_id="grounding-switch",
                    elem_classes=["ios-switch"]
                )

            copy_btn = gr.HTML(
                value="",  # Empty initially - button will be added when analysis is generated
                elem_classes=["copy-btn-container"]
            )

        # Hidden input for dataset selection
        selected_dataset_hidden = gr.Textbox(visible=False, elem_classes="dataset-selector")

        # State to store model_color_map
        model_color_map_state = gr.State(value={})
        
        # State to store seen sample IDs for unique refresh
        seen_samples_state = gr.State(value=set())

        summary_table = gr.HTML(
            value="",
            visible=False,
            elem_classes="summary-table-container"
        )

        # Modal for sample comparison
        with gr.Column(visible=False, elem_classes="modal-overlay") as modal_overlay:
            close_modal_btn = gr.Button("×", elem_classes=["close-modal-btn"])

        with gr.Column(visible=False, elem_classes="modal-content") as modal_content:
            # Header moved to modal_html to support inline model filters

            modal_html = gr.HTML(value="")
            
            # Refresh button (Custom HTML: SVG + Clickable Text)
            hidden_refresh_btn = gr.Button(visible=False, elem_id="hidden-refresh-btn")
            
            refresh_icon_svg = load_svg_icon(REFRESH_ICON)
            
            refresh_display = gr.HTML(
                value=f"""
                <div class="refresh-container">
                    <div class="refresh-icon">{refresh_icon_svg}</div>
                    <div class="refresh-text" onclick="document.getElementById('hidden-refresh-btn').click()">换一批</div>
                </div>
                """
            )

        # Event handlers

        # Generate analysis
        generate_analysis_btn.click(
            fn=generate_analysis,
            inputs=[result_dir_checkboxes, sid_switch, pid_switch, grounding_switch],
            outputs=[radar_image, summary_table, copy_btn, error_display, model_color_map_state]
        )

        # SID switch change - regenerate table
        sid_switch.change(
            fn=generate_analysis,
            inputs=[result_dir_checkboxes, sid_switch, pid_switch, grounding_switch],
            outputs=[radar_image, summary_table, copy_btn, error_display, model_color_map_state]
        )

        # PID switch change - regenerate table
        pid_switch.change(
            fn=generate_analysis,
            inputs=[result_dir_checkboxes, sid_switch, pid_switch, grounding_switch],
            outputs=[radar_image, summary_table, copy_btn, error_display, model_color_map_state]
        )

        # Grounding switch change - regenerate table
        grounding_switch.change(
            fn=generate_analysis,
            inputs=[result_dir_checkboxes, sid_switch, pid_switch, grounding_switch],
            outputs=[radar_image, summary_table, copy_btn, error_display, model_color_map_state]
        )

        # Show sample comparison when hidden input changes (triggered by JS)
        # Step 1: Immediately show modal with loading indicator AND reset seen samples
        loading_event = selected_dataset_hidden.change(
            fn=show_loading_modal,
            inputs=[selected_dataset_hidden],
            outputs=[modal_overlay, modal_content, modal_html, seen_samples_state]
        )

        # Step 2: Load actual data and update content
        loading_event.then(
            fn=show_sample_comparison,
            inputs=[result_dir_checkboxes, selected_dataset_hidden, model_color_map_state, seen_samples_state],
            outputs=[modal_html, seen_samples_state]
        )

        # Refresh samples
        hidden_refresh_btn.click(
            fn=show_sample_comparison,
            inputs=[result_dir_checkboxes, selected_dataset_hidden, model_color_map_state, seen_samples_state],
            outputs=[modal_html, seen_samples_state]
        )

        # Close modal when clicking the close button
        close_modal_btn.click(
            fn=hide_modal,
            inputs=None,
            outputs=[modal_overlay, modal_content, selected_dataset_hidden]
        )

    return content
