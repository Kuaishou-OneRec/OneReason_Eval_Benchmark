"""
Gradio UI components for SID extraction (reverse lookup from PID)
"""
import gradio as gr
from typing import List, Tuple
from extract_sid.handlers import batch_convert_pids, get_loading_status, is_data_ready, EXPECTED_LOADING_TIME
from config import COPY_ICON
from utils import load_svg_icon


# Constants
INITIAL_INPUT_COUNT = 20
MAX_VISIBLE_HEIGHT = "150px"  # Fixed height before expansion
LOADING_STATUS_REFRESH_INTERVAL = 60  # seconds (refresh once per minute)
OUTPUT_PREVIEW_LENGTH = 300  # Characters to show before requiring expand


def _get_loading_status_html() -> str:
    """Generate HTML for loading status indicator with progress bar."""
    status = get_loading_status()

    if status["status"] == "loading":
        return f'''
        <div class="loading-status loading">
            <div class="loading-status-content">
                <span class="loading-spinner"></span>
                <span>PID2SID 映射加载中... (约{EXPECTED_LOADING_TIME}s)</span>
            </div>
        </div>
        '''
    elif status["status"] == "done":
        elapsed = int(status.get("elapsed", 0))
        return f'''
        <div class="loading-status done">
            <span class="status-icon">&#10003;</span>
            <span>PID2SID 映射已就绪 ({status["count"]:,} 条, 用时 {elapsed}s)</span>
        </div>
        '''
    elif status["status"] == "error":
        return f'''
        <div class="loading-status error">
            <span class="status-icon">&#10007;</span>
            <span>加载失败: {status["message"][:50]}</span>
        </div>
        '''
    else:
        return '<div class="loading-status idle">PID2SID 映射未加载</div>'


def _generate_output_html(index: int, content: str = None) -> str:
    """
    Generate the HTML for the output box, including label and copy button.
    Supports expand/collapse for long content.

    Args:
        index: The row index (0-based)
        content: The content to display. If None, shows placeholder.
    """
    import html
    copy_icon_svg = load_svg_icon(COPY_ICON)
    sq = "'"  # Single quote for onclick handlers

    if content:
        content_id = f"sid_output_{index}"
        escaped_content = html.escape(content)
        needs_expand = len(content) > OUTPUT_PREVIEW_LENGTH

        if needs_expand:
            # Preview version (truncated)
            preview_text = content[:OUTPUT_PREVIEW_LENGTH] + "..."
            escaped_preview = html.escape(preview_text)
            expand_link = f'<span class="sid-expand-link" onclick="toggleSidText({sq}{content_id}{sq})">展开</span>'
            collapse_link = f'<span class="sid-collapse-link" onclick="toggleSidText({sq}{content_id}{sq})" style="display:none;">收起</span>'

            display_content = f'''
            <div id="{content_id}_preview" class="custom-output-content">
                <pre>{escaped_preview}</pre>
                {expand_link}
            </div>
            <div id="{content_id}_full" class="custom-output-content" style="display:none;">
                <pre>{escaped_content}</pre>
                {collapse_link}
            </div>
            '''
        else:
            display_content = f'<div class="custom-output-content"><pre>{escaped_content}</pre></div>'

        copy_btn = f'''
        <div class="copy-btn-container">
            <button class="copy-btn" onclick="copySidToClipboard(this)" data-content="{escaped_content}" title="复制">
                {copy_icon_svg}
            </button>
        </div>
        '''
    else:
        display_content = '<div class="custom-output-content empty">转换后的 SID codes 将显示在这里</div>'
        copy_btn = ''  # No copy button when empty

    return f'''
    <div class="custom-output-wrapper">
        <label class="extract-sid-label">输出 {index + 1}</label>
        <div class="custom-output-box">
            {copy_btn}
            {display_content}
        </div>
    </div>
    '''


def create_extract_sid_ui():
    """
    Create the SID extraction UI components (reverse lookup from PID).

    Returns:
        Tuple of (ui_component, state_components)
    """

    with gr.Column(elem_classes=["extract-sid-page"]) as extract_sid_page:
        # Title on top
        gr.Markdown("# Extract SID", elem_classes=["extract-sid-title"])

        # Header row: Loading status (left) + Convert button (right)
        with gr.Row(elem_classes=["extract-sid-header-row"]):
            # Loading status indicator
            loading_status = gr.HTML(
                value=_get_loading_status_html(),
                elem_classes=["extract-sid-loading-status"]
            )

            convert_btn = gr.Button(
                "开始转换",
                variant="primary",
                elem_classes=["extract-sid-convert-btn"]
            )

        # State to track number of visible rows
        num_visible_rows = gr.State(INITIAL_INPUT_COUNT)

        # Main content area
        MAX_ROWS = 20
        main_container = gr.Column(elem_classes=["extract-sid-main-container"])

        with main_container:
            input_boxes = []
            output_boxes = []
            card_rows = []

            # Pre-create MAX_ROWS
            for i in range(MAX_ROWS):
                # Row container (Grid Layout: Card)
                card_row = gr.Row(
                    elem_classes=["extract-sid-card-row"],
                    visible=(i < INITIAL_INPUT_COUNT)
                )

                with card_row:
                    # The Card (Grid Layout: Input + Output)
                    with gr.Column(elem_classes=["extract-sid-card"], scale=1):
                        # Left: Input (Gradio Textbox)
                        text_input = gr.Textbox(
                            label=f"输入 {i+1}",
                            placeholder="输入 PID (支持单个、逗号分隔或换行分隔)...",
                            lines=4,
                            elem_classes=["extract-sid-input-box"],
                            show_label=True
                        )
                        input_boxes.append(text_input)

                        # Right: Output (Custom HTML)
                        output_html = gr.HTML(
                            value=_generate_output_html(i),
                            elem_classes=["extract-sid-output-col"]
                        )
                        output_boxes.append(output_html)

                card_rows.append((card_row))

    def handle_convert(*input_texts):
        """
        Convert all input PIDs to SID codes and return HTML outputs.
        """
        # Filter out None values
        valid_texts = [text if text else "" for text in input_texts]

        # Convert PIDs to SID codes
        sid_results = batch_convert_pids(valid_texts)

        # Format as HTML
        html_outputs = []
        for i, sid_str in enumerate(sid_results):
            content = sid_str if sid_str else None
            if not content and valid_texts[i]:  # If input exists but no SID found
                content = "未找到有效的 SID codes"

            html_outputs.append(_generate_output_html(i, content))

        # Pad with empty outputs
        while len(html_outputs) < MAX_ROWS:
            i = len(html_outputs)
            html_outputs.append(_generate_output_html(i))

        return html_outputs

    # Connect convert button
    convert_btn.click(
        fn=lambda: gr.update(value="转换中...", interactive=False),
        inputs=None,
        outputs=convert_btn
    ).then(
        fn=handle_convert,
        inputs=input_boxes,
        outputs=output_boxes
    ).then(
        fn=lambda: gr.update(value="开始转换", interactive=True),
        inputs=None,
        outputs=convert_btn
    )

    return extract_sid_page, {
        'input_boxes': input_boxes,
        'output_boxes': output_boxes,
        'card_rows': card_rows,
        'convert_btn': convert_btn,
        'loading_status': loading_status,
        'num_visible_rows': num_visible_rows
    }
