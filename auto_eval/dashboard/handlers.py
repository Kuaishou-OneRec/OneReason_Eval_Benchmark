"""Business logic for generating dashboard HTML from selected directories."""

from __future__ import annotations

import html
import time
from datetime import datetime
from pathlib import Path

from auto_eval.config import RESULT_ROOTS, DASHBOARD_OUTPUT_DIR
from auto_eval.tasks_meta import get_all_versions
from .data import build_race_score_card_data, build_section_data
from .parser import build_model_specs
from .renderer import render_dashboard
from .sections import SECTION_SPECS


# Maximum age of dashboard files before cleanup (24 hours)
_CLEANUP_MAX_AGE_SECONDS = 24 * 60 * 60


def _cleanup_old_dashboards() -> int:
    """Remove dashboard HTML files older than 24 hours.

    Returns:
        Number of files removed.
    """
    removed = 0
    if not DASHBOARD_OUTPUT_DIR.exists():
        return removed
    cutoff = time.time() - _CLEANUP_MAX_AGE_SECONDS
    for html_file in DASHBOARD_OUTPUT_DIR.glob("dashboard_*.html"):
        try:
            if html_file.stat().st_mtime < cutoff:
                html_file.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def generate_dashboard_html(selected_dirs: list[str]) -> tuple[str, str]:
    """Generate a dashboard HTML file from selected result directories.

    Args:
        selected_dirs: List of result directory names (e.g. "results_v3_step1000_think").

    Returns:
        Tuple of (iframe_html, status_text).
        iframe_html contains an <iframe> with srcdoc embedding the generated HTML,
        or a placeholder message on error.
        status_text describes what happened.
    """
    if not selected_dirs:
        return (
            '<div style="text-align:center; padding:60px; color:#999;">'
            '请先选择至少一个结果目录</div>',
            "请先选择至少一个结果目录",
        )

    # Build model specs from directory names
    models = build_model_specs(selected_dirs, RESULT_ROOTS)
    if not models:
        return (
            '<div style="text-align:center; padding:60px; color:#c00;">'
            '所选目录均无法解析或不存在，请检查目录名称格式</div>',
            f"错误：{len(selected_dirs)} 个目录均无法解析",
        )

    # Build section data for each section spec
    sections = [build_section_data(spec, models) for spec in SECTION_SPECS]
    race_score_cards = build_race_score_card_data(models)

    # Build source description
    source_description = f"选中 {len(selected_dirs)} 个目录，解析成功 {len(models)} 个"

    # Render full HTML
    html_content = render_dashboard(
        title="对比看板",
        source_description=source_description,
        models=models,
        sections=sections,
        race_score_cards=race_score_cards,
    )

    # Save a copy for direct download
    DASHBOARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"dashboard_{timestamp}.html"
    output_path = DASHBOARD_OUTPUT_DIR / output_filename
    output_path.write_text(html_content, encoding="utf-8")
    _cleanup_old_dashboards()

    # Embed HTML via srcdoc
    escaped_html = html.escape(html_content)
    iframe_html = (
        f'<iframe srcdoc="{escaped_html}" '
        f'style="width:100%; height:calc(100vh - 120px); border:none;"></iframe>'
    )

    status_text = (
        f"看板生成成功：{len(models)} 个模型目录，"
        f"{sum(len(s.rows) for s in sections)} 行指标"
    )

    return iframe_html, status_text
