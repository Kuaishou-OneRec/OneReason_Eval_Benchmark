"""Dashboard module — interactive comparison dashboard for evaluation results."""

from .handlers import generate_dashboard_html
from .styles import DASHBOARD_CSS, DASHBOARD_JS
from .ui import create_dashboard_content

__all__ = [
    "DASHBOARD_CSS",
    "DASHBOARD_JS",
    "create_dashboard_content",
    "generate_dashboard_html",
]
