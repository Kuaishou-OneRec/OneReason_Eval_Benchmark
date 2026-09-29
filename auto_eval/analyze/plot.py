"""
Plot generation module - rewritten from original plot.py for web interface
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple
import tempfile
import re
from pathlib import Path

from auto_eval.config import PLOT_DPI, PLOT_FIGSIZE, PLOT_OUTPUT_DIR
from auto_eval.tasks_meta import get_task_plot_metadata, get_all_versions


# ========================================
# Build plot metadata dynamically from registries
# ========================================
def _build_plot_metadata():
    """Merge analysis metadata from all version registries."""
    combined = {
        "task_metrics": {},
        "radar_task_metrics": {},
        "task_ranges": {},
        "affected_tasks": [],
    }
    for v in get_all_versions():
        meta = get_task_plot_metadata(v)
        for key in combined:
            if isinstance(combined[key], dict):
                combined[key].update(meta[key])
            elif isinstance(combined[key], list):
                combined[key].extend(meta[key])
    return combined


_PLOT_META = _build_plot_metadata()
TASK_METRICS = _PLOT_META["task_metrics"]
RADAR_TASK_METRICS = _PLOT_META["radar_task_metrics"]
TASK_RANGES = _PLOT_META["task_ranges"]

# Display name mapping for long metric names
METRIC_DISPLAY_NAMES = {
    'macro_wip_double_weighted_f1': 'llm_f1',
    'micro_wip_double_weighted_f1': 'micro_llm_f1',
    'macro_wip_double_weighted_core_f1': 'llm_core_f1',
}

# Fallback metrics
FALLBACK_METRICS = {
    'pass@1': 'accuracy',
}

# Task name aliases - if original task name not found, try alias
TASK_ALIASES = {
    'rec_reason': 'reco_reason',
    'item_understand': 'sid_caption',
    'video': 'sid_user_doc',
    'product': 'goods'
}


def extract_model_name(file_path: str) -> str:
    """Extract model name from file path"""
    path = Path(file_path)
    # Find directory containing step and lr information
    for part in path.parts:
        if 'step' in part and 'lr' in part:
            return part
    # If not found, use parent directory name
    name = path.parent.name
    # Remove 'results_' prefix if present
    if name.startswith('results_'):
        name = name[8:]
    return name


def load_model_data(file_paths: List[str], strip_sidmodel_prefix: bool = False) -> Dict:
    """Load data from multiple models

    Args:
        file_paths: List of paths to eval result JSON files
        strip_sidmodel_prefix: If True, replace regular metrics with their sidmodel_
                               prefixed versions (stripping the prefix) for grounding mode.
                               e.g. sidmodel_pass@32 overwrites pass@32
    """
    models_data = {}

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model_name = extract_model_name(file_path)
        # Get first (and only) model's data
        model_key = list(data.keys())[0]
        model_data = data[model_key]

        if strip_sidmodel_prefix:
            cleaned_data = {}
            for task_key, task_val in model_data.items():
                if isinstance(task_val, dict):
                    cleaned_task_val = {}
                    for section_key, section_val in task_val.items():
                        if isinstance(section_val, dict):
                            # First pass: keep non-sidmodel keys
                            base = {k: v for k, v in section_val.items()
                                    if not k.startswith('sidmodel_')}
                            # Second pass: sidmodel_ keys overwrite after stripping prefix
                            for k, v in section_val.items():
                                if k.startswith('sidmodel_'):
                                    base[k[len('sidmodel_'):]] = v
                            cleaned_task_val[section_key] = base
                        else:
                            cleaned_task_val[section_key] = section_val
                    cleaned_data[task_key] = cleaned_task_val
                else:
                    cleaned_data[task_key] = task_val
            model_data = cleaned_data

        models_data[model_name] = model_data

    return models_data


def extract_metrics(models_data: Dict, use_radar_metrics: bool = False) -> Tuple[Dict, Dict]:
    """
    Extract specified metrics with fallback support

    Args:
        models_data: Model evaluation data
        use_radar_metrics: If True, use RADAR_TASK_METRICS (single metric per task)
                          If False, use TASK_METRICS (multiple metrics per task)

    Returns:
        Tuple of (models_metrics, task_metrics_dict)
        - For table: task_metrics_dict maps (task, metric) tuples to metric names
        - For radar: task_metrics_dict maps task names to metric names
    """
    results = {}
    task_metrics_dict = dict(RADAR_TASK_METRICS) if use_radar_metrics else {}

    if use_radar_metrics:
        # Radar chart mode: use single metric per task
        for model_name, model_data in models_data.items():
            model_metrics = {}

            for task, metric in RADAR_TASK_METRICS.items():
                # Determine actual task key: use original or alias
                actual_task = task
                if task not in model_data and task in TASK_ALIASES:
                    alias = TASK_ALIASES[task]
                    if alias in model_data:
                        actual_task = alias

                if actual_task in model_data and 'test' in model_data[actual_task]:
                    # First try to get original metric
                    if metric in model_data[actual_task]['test']:
                        model_metrics[task] = model_data[actual_task]['test'][metric]
                    # If original metric doesn't exist, try fallback metric
                    elif metric in FALLBACK_METRICS and FALLBACK_METRICS[metric] in model_data[actual_task]['test']:
                        fallback_metric = FALLBACK_METRICS[metric]
                        model_metrics[task] = model_data[actual_task]['test'][fallback_metric]
                    else:
                        model_metrics[task] = 0  # If neither metric exists, set to 0
                else:
                    model_metrics[task] = 0  # If task doesn't exist, set to 0

            results[model_name] = model_metrics
    else:
        # Table mode: support multiple metrics per task
        # Define affected tasks and metrics for SID/PID switching
        AFFECTED_TASKS = _PLOT_META["affected_tasks"]
        AFFECTED_METRICS = ['pass@1', 'pass@16', 'recall@16', 'pass@32', 'recall@32', 'pass@64', 'recall@64', 'pass@128', 'recall@128', 'pass@256', 'recall@256']

        for model_name, model_data in models_data.items():
            model_metrics = {}

            for task, metrics_list in TASK_METRICS.items():
                # Determine actual task key: use original or alias
                actual_task = task
                if task not in model_data and task in TASK_ALIASES:
                    alias = TASK_ALIASES[task]
                    if alias in model_data:
                        actual_task = alias

                for metric in metrics_list:
                    task_metric_key = (task, metric)

                    if actual_task in model_data and 'test' in model_data[actual_task]:
                        # First try to get original metric
                        if metric in model_data[actual_task]['test']:
                            model_metrics[task_metric_key] = model_data[actual_task]['test'][metric]
                        # If original metric doesn't exist, try fallback metric
                        elif metric in FALLBACK_METRICS and FALLBACK_METRICS[metric] in model_data[actual_task]['test']:
                            fallback_metric = FALLBACK_METRICS[metric]
                            model_metrics[task_metric_key] = model_data[actual_task]['test'][fallback_metric]
                        else:
                            model_metrics[task_metric_key] = None  # If neither metric exists
                    else:
                        model_metrics[task_metric_key] = None  # If task doesn't exist

                    # Build task_metrics_dict for table
                    task_metrics_dict[task_metric_key] = metric

                    # For affected tasks and metrics, also extract PID metrics
                    if task in AFFECTED_TASKS and metric in AFFECTED_METRICS:
                        pid_metric = f'pid_{metric}'
                        pid_task_metric_key = (task, pid_metric)

                        if actual_task in model_data and 'test' in model_data[actual_task]:
                            if pid_metric in model_data[actual_task]['test']:
                                model_metrics[pid_task_metric_key] = model_data[actual_task]['test'][pid_metric]
                            else:
                                model_metrics[pid_task_metric_key] = None  # N/A if not found
                        else:
                            model_metrics[pid_task_metric_key] = None

                        # Add PID metric to dict
                        task_metrics_dict[pid_task_metric_key] = pid_metric

            results[model_name] = model_metrics

    return results, task_metrics_dict


def normalize_metrics(models_metrics: Dict, task_metrics: Dict) -> Tuple[Dict, Dict]:
    """Normalize metrics using fixed ranges"""
    normalized_data = {}
    tasks = list(task_metrics.keys())

    # Normalize data
    for model_name, metrics in models_metrics.items():
        normalized_metrics = {}

        for task in tasks:
            value = metrics[task]
            min_val = TASK_RANGES[task]['min']
            max_val = TASK_RANGES[task]['max']

            # For pass@1 and pass@32 larger is better
            clamped_value = max(min_val, min(value, max_val))
            if max_val > min_val:
                normalized_value = (clamped_value - min_val) / (max_val - min_val)
            else:
                normalized_value = 1.0 if value > 0 else 0.0

            normalized_metrics[task] = max(0, min(1, normalized_value))  # Ensure in [0,1] range

        normalized_data[model_name] = normalized_metrics

    return normalized_data, TASK_RANGES


def normalize_metrics_dynamic(models_metrics: Dict, task_metrics: Dict) -> Tuple[Dict, Dict]:
    """Normalize metrics using dynamic ranges (min/max values with 10% padding)"""
    normalized_data = {}
    tasks = list(task_metrics.keys())

    # Calculate dynamic ranges for each task
    dynamic_ranges = {}
    for task in tasks:
        # Collect values from models, excluding those with "qwen3" in name
        values = [metrics[task] for model_name, metrics in models_metrics.items()
                  if 'qwen3' not in model_name.lower()]

        # If no models remain after filtering, fall back to all models
        if not values:
            values = [metrics[task] for metrics in models_metrics.values()]

        min_val = min(values)
        max_val = max(values)

        # Add 10% padding
        range_span = max_val - min_val
        if range_span > 0:
            padding = range_span * 0.1
            dynamic_min = min_val - padding
            dynamic_max = max_val + padding
        else:
            # If all values are the same, add small padding
            dynamic_min = min_val - 0.05 if min_val > 0.05 else 0
            dynamic_max = max_val + 0.05

        # Ensure min is not below 0
        dynamic_min = max(0, dynamic_min)

        dynamic_ranges[task] = {'min': dynamic_min, 'max': dynamic_max}

    # Normalize data using dynamic ranges
    for model_name, metrics in models_metrics.items():
        normalized_metrics = {}

        for task in tasks:
            value = metrics[task]
            min_val = dynamic_ranges[task]['min']
            max_val = dynamic_ranges[task]['max']

            # For pass@1 and pass@32, larger is better
            clamped_value = max(min_val, min(value, max_val))
            if max_val > min_val:
                normalized_value = (clamped_value - min_val) / (max_val - min_val)
            else:
                normalized_value = 1.0 if value > 0 else 0.0

            normalized_metrics[task] = max(0, min(1, normalized_value))  # Ensure in [0,1] range

        normalized_data[model_name] = normalized_metrics

    return normalized_data, dynamic_ranges


def get_display_ranges(models_metrics: Dict, task_metrics: Dict) -> Dict:
    """Calculate dynamic ranges for display"""
    tasks = list(task_metrics.keys())
    display_ranges = {}

    for task in tasks:
        values = [metrics[task] for metrics in models_metrics.values()]
        min_val = min(values)
        max_val = max(values)

        # Slightly expand range for better display
        range_span = max_val - min_val
        if range_span > 0:
            padding = range_span * 0.1  # 10% padding
            display_min = max(0, min_val - padding)
            display_max = max_val + padding
        else:
            display_min = 0
            display_max = max_val + 0.1 if max_val > 0 else 1

        display_ranges[task] = {'min': display_min, 'max': display_max}

    return display_ranges


def create_radar_chart(normalized_data_fixed: Dict, normalized_data_dynamic: Dict,
                       models_metrics: Dict, task_metrics: Dict,
                       task_ranges_fixed: Dict, task_ranges_dynamic: Dict,
                       style_config: Dict,
                       save_path: str = None) -> str:
    """
    Create dual radar charts (fixed range and dynamic range)

    Args:
        normalized_data_fixed: Normalized metrics data using fixed ranges
        normalized_data_dynamic: Normalized metrics data using dynamic ranges
        models_metrics: Original metrics data
        task_metrics: Task to metric mapping
        task_ranges_fixed: Fixed task value ranges
        task_ranges_dynamic: Dynamic task value ranges
        style_config: Dict containing colors, line_styles, markers, and sorted_model_names
        save_path: Optional path to save the chart

    Returns:
        Path to saved PNG file
    """
    tasks = list(task_metrics.keys())
    num_tasks = len(tasks)

    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    angles += angles[:1]  # Close the shape

    # Create figure with GridSpec layout
    # Row 0: Legend (height_ratios=[0.25, 0.75]) - Increased height for vertical legend
    # Row 1: Two radar charts
    fig = plt.figure(figsize=(40, 24)) # Increased total height
    gs = fig.add_gridspec(2, 2, height_ratios=[0.25, 0.75], hspace=0.4) # Increased hspace for more separation

    # Legend subplot (spans both columns in first row)
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')

    # Left subplot: radar chart with fixed ranges
    ax_radar_fixed = fig.add_subplot(gs[1, 0], projection='polar')
    # Right subplot: radar chart with dynamic ranges
    ax_radar_dynamic = fig.add_subplot(gs[1, 1], projection='polar')

    # Extract style configuration
    colors = style_config['colors']
    line_styles = style_config['line_styles']
    markers = style_config['markers']
    sorted_model_names = style_config['sorted_model_names']

    # Plot each model in sorted order on both radar charts
    legend_handles = []
    legend_labels = []
    for i, model_name in enumerate(sorted_model_names):
        # Fixed range radar chart (plot all models)
        metrics_fixed = normalized_data_fixed[model_name]
        values_fixed = [metrics_fixed[task] for task in tasks]
        values_fixed += values_fixed[:1]  # Close the shape

        line, = ax_radar_fixed.plot(angles, values_fixed,
                color=colors[i % len(colors)],
                linewidth=3,
                linestyle=line_styles[i % len(line_styles)],
                marker=markers[i % len(markers)],
                markersize=15,
                label=model_name.replace('results_', ''))
        ax_radar_fixed.fill(angles, values_fixed,
                color=colors[i % len(colors)],
                alpha=0.15)

        # Dynamic range radar chart (skip models with "qwen3" in name)
        if 'qwen3' not in model_name.lower():
            metrics_dynamic = normalized_data_dynamic[model_name]
            values_dynamic = [metrics_dynamic[task] for task in tasks]
            values_dynamic += values_dynamic[:1]  # Close the shape

            ax_radar_dynamic.plot(angles, values_dynamic,
                    color=colors[i % len(colors)],
                    linewidth=3,
                    linestyle=line_styles[i % len(line_styles)],
                    marker=markers[i % len(markers)],
                    markersize=8,
                    label=model_name.replace('results_', ''))
            ax_radar_dynamic.fill(angles, values_dynamic,
                    color=colors[i % len(colors)],
                    alpha=0.15)

        # Collect legend info (only once)
        legend_handles.append(line)
        legend_labels.append(model_name.replace('results_', ''))

    # Set labels for fixed range radar chart
    ax_radar_fixed.set_xticks(angles[:-1])
    task_labels_fixed = []
    for task in tasks:
        min_val = task_ranges_fixed[task]['min']
        max_val = task_ranges_fixed[task]['max']
        task_labels_fixed.append(f"{task}\n[{min_val:.4f}-{max_val:.4f}]")
    ax_radar_fixed.set_xticklabels(task_labels_fixed, fontsize=24, fontweight='bold')
    ax_radar_fixed.set_ylim(0, 1)
    ax_radar_fixed.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar_fixed.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=24, fontweight='bold')
    ax_radar_fixed.grid(True, color='#333333', alpha=0.5, linewidth=0.8)
    ax_radar_fixed.set_title('Fixed Range Normalization', fontsize=30, fontweight='bold', pad=80)

    # Set labels for dynamic range radar chart
    ax_radar_dynamic.set_xticks(angles[:-1])
    task_labels_dynamic = []
    for task in tasks:
        min_val = task_ranges_dynamic[task]['min']
        max_val = task_ranges_dynamic[task]['max']
        task_labels_dynamic.append(f"{task}\n[{min_val:.4f}-{max_val:.4f}]")
    ax_radar_dynamic.set_xticklabels(task_labels_dynamic, fontsize=24, fontweight='bold')
    ax_radar_dynamic.set_ylim(0, 1)
    ax_radar_dynamic.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar_dynamic.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=24, fontweight='bold')
    ax_radar_dynamic.grid(True, color='#333333', alpha=0.5, linewidth=0.8)
    ax_radar_dynamic.set_title('Dynamic Range Normalization (±10%)', fontsize=30, fontweight='bold', pad=80)

    # Add legend to the separate legend subplot
    # Use single column for legend
    ax_legend.legend(handles=legend_handles, labels=legend_labels, loc='center',
                     ncol=1, fontsize=28, frameon=True, fancybox=True, shadow=True)

    # Add title to the figure
    # fig.suptitle('Model Performance Comparison', fontsize=28, fontweight='bold', y=0.98)

    # Save to file
    if not save_path:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=str(PLOT_OUTPUT_DIR))
        save_path = temp_file.name
        temp_file.close()

    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    return save_path




def generate_line_style_svg(color: str, linestyle: str, marker: str, 
                           width: int = 60, height: int = 20) -> str:
    """
    Generate SVG icon showing line style and marker
    
    Args:
        color: Line color (hex or named color)
        linestyle: Matplotlib linestyle ('-', '--', '-.', ':', or tuple)
        marker: Matplotlib marker ('o', 's', '^', 'v', 'D', 'p', '*', etc.)
        width: SVG width in pixels
        height: SVG height in pixels
    
    Returns:
        SVG string
    """
    # Convert matplotlib linestyle to SVG stroke-dasharray
    linestyle_map = {
        '-': 'none',
        '--': '8,4',
        '-.': '8,4,2,4',
        ':': '2,2',
    }
    
    # Handle tuple linestyles (convert to closest standard style)
    if isinstance(linestyle, tuple):
        # Map complex linestyles to simpler ones
        linestyle = '--'  # Default to dashed for complex styles
    
    stroke_dasharray = linestyle_map.get(linestyle, 'none')
    
    # Line coordinates (horizontal line in the middle)
    y_center = height / 2
    line_start_x = 5
    line_end_x = width - 5
    line_center_x = width / 2
    
    # Generate marker SVG based on type
    marker_svg = ''
    marker_size = 6
    
    if marker == 'o':  # Circle
        marker_svg = f'<circle cx="{line_center_x}" cy="{y_center}" r="{marker_size}" fill="{color}" stroke="{color}" stroke-width="1.5"/>'
    elif marker == 's':  # Square
        marker_svg = f'<rect x="{line_center_x - marker_size}" y="{y_center - marker_size}" width="{marker_size * 2}" height="{marker_size * 2}" fill="{color}" stroke="{color}" stroke-width="1.5"/>'
    elif marker == '^':  # Triangle up
        points = f"{line_center_x},{y_center - marker_size} {line_center_x - marker_size},{y_center + marker_size} {line_center_x + marker_size},{y_center + marker_size}"
        marker_svg = f'<polygon points="{points}" fill="{color}" stroke="{color}" stroke-width="1.5"/>'
    elif marker == 'v':  # Triangle down
        points = f"{line_center_x},{y_center + marker_size} {line_center_x - marker_size},{y_center - marker_size} {line_center_x + marker_size},{y_center - marker_size}"
        marker_svg = f'<polygon points="{points}" fill="{color}" stroke="{color}" stroke-width="1.5"/>'
    elif marker == 'D':  # Diamond
        points = f"{line_center_x},{y_center - marker_size} {line_center_x + marker_size},{y_center} {line_center_x},{y_center + marker_size} {line_center_x - marker_size},{y_center}"
        marker_svg = f'<polygon points="{points}" fill="{color}" stroke="{color}" stroke-width="1.5"/>'
    elif marker == 'p':  # Pentagon
        # Simplified pentagon (5 points)
        angle_offset = -90  # Start from top
        points_list = []
        for i in range(5):
            angle = (angle_offset + i * 72) * np.pi / 180
            x = line_center_x + marker_size * np.cos(angle)
            y = y_center + marker_size * np.sin(angle)
            points_list.append(f"{x},{y}")
        points = " ".join(points_list)
        marker_svg = f'<polygon points="{points}" fill="{color}" stroke="{color}" stroke-width="1.5"/>'
    elif marker == '*':  # Star
        # 5-pointed star
        points_list = []
        for i in range(10):
            angle = (i * 36 - 90) * np.pi / 180
            r = marker_size if i % 2 == 0 else marker_size * 0.4
            x = line_center_x + r * np.cos(angle)
            y = y_center + r * np.sin(angle)
            points_list.append(f"{x},{y}")
        points = " ".join(points_list)
        marker_svg = f'<polygon points="{points}" fill="{color}" stroke="{color}" stroke-width="1.5"/>'
    elif marker == 'h':  # Hexagon
        points_list = []
        for i in range(6):
            angle = (i * 60 - 90) * np.pi / 180
            x = line_center_x + marker_size * np.cos(angle)
            y = y_center + marker_size * np.sin(angle)
            points_list.append(f"{x},{y}")
        points = " ".join(points_list)
        marker_svg = f'<polygon points="{points}" fill="{color}" stroke="{color}" stroke-width="1.5"/>'
    elif marker in ['H', '+', 'x', '<', '>', 'd', '|', '_', 'P', 'X', '.', ',']:
        # For other markers, use circle as fallback
        marker_svg = f'<circle cx="{line_center_x}" cy="{y_center}" r="{marker_size}" fill="{color}" stroke="{color}" stroke-width="1.5"/>'
    else:
        # Default to circle
        marker_svg = f'<circle cx="{line_center_x}" cy="{y_center}" r="{marker_size}" fill="{color}" stroke="{color}" stroke-width="1.5"/>'
    
    # Build complete SVG
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <line x1="{line_start_x}" y1="{y_center}" x2="{line_end_x}" y2="{y_center}" 
          stroke="{color}" stroke-width="2.5" stroke-dasharray="{stroke_dasharray}" 
          stroke-linecap="round"/>
    {marker_svg}
</svg>'''
    
    return svg


def sort_model_names_by_step(model_names: List[str]) -> List[str]:
    """
    Sort model names alphabetically

    Args:
        model_names: List of model names

    Returns:
        Sorted list of model names
    """
    return sorted(model_names)


def create_summary_table(models_metrics: Dict, task_metrics: Dict, style_config: Dict,
                        show_sid: bool = True, show_pid: bool = False) -> pd.DataFrame:
    """
    Create summary table with original values and SVG line style indicators in headers
    Supports multiple metrics per task (displayed as multiple rows)
    Supports SID/PID switching for affected tasks

    Args:
        models_metrics: Original metrics data (keys are (task, metric) tuples for table mode)
        task_metrics: Task to metric mapping (keys are (task, metric) tuples, values are metric names)
        style_config: Dict containing colors, line_styles, markers, and sorted_model_names
        show_sid: Whether to show SID metrics (default: True)
        show_pid: Whether to show PID metrics (default: False)

    Returns:
        DataFrame with summary table
    """
    # Extract style configuration
    colors = style_config['colors']
    line_styles = style_config['line_styles']
    markers = style_config['markers']
    sorted_model_names = style_config['sorted_model_names']

    # Define affected tasks and metrics for SID/PID switching
    AFFECTED_TASKS = _PLOT_META["affected_tasks"]
    AFFECTED_METRICS = ['pass@1', 'pass@16', 'recall@16', 'pass@32', 'recall@32', 'pass@64', 'recall@64', 'pass@128', 'recall@128', 'pass@256', 'recall@256']

    # Build table data
    # Iterate through tasks in TASK_METRICS order to maintain consistent ordering
    table_data = []

    for task, metrics_list in TASK_METRICS.items():
        for metric in metrics_list:
            task_metric_key = (task, metric)

            # Check if this is an affected task and metric
            is_affected = (task in AFFECTED_TASKS and metric in AFFECTED_METRICS)

            if is_affected:
                # Handle affected tasks based on switch states
                if show_sid and show_pid:
                    # Show combined: metric / pid_metric
                    display_metric = METRIC_DISPLAY_NAMES.get(metric, metric)
                    metric_display = f"{display_metric} / pid_{display_metric}"
                    row = [task, metric_display]

                    for model_name in sorted_model_names:
                        sid_value = models_metrics[model_name][task_metric_key]
                        pid_value = models_metrics[model_name].get((task, f'pid_{metric}'), None)

                        sid_display = f"{sid_value:.4f}" if sid_value is not None else "-"
                        pid_display = f"{pid_value:.4f}" if pid_value is not None else "-"
                        value_display = f"{sid_display} / {pid_display}"
                        row.append(value_display)

                    table_data.append(row)

                elif show_sid and not show_pid:
                    # Show SID only
                    row = [task, METRIC_DISPLAY_NAMES.get(metric, metric)]

                    for model_name in sorted_model_names:
                        sid_value = models_metrics[model_name][task_metric_key]
                        row.append(f"{sid_value:.4f}" if sid_value is not None else "-")

                    table_data.append(row)

                elif not show_sid and show_pid:
                    # Show PID only
                    display_metric = METRIC_DISPLAY_NAMES.get(metric, metric)
                    metric_display = f"pid_{display_metric}"
                    row = [task, metric_display]

                    for model_name in sorted_model_names:
                        pid_value = models_metrics[model_name].get((task, f'pid_{metric}'), None)
                        row.append(f"{pid_value:.4f}" if pid_value is not None else "-")

                    table_data.append(row)
                # else: both switches off, skip this metric

            else:
                # Not affected, always show original metric
                row = [task, METRIC_DISPLAY_NAMES.get(metric, metric)]

                for model_name in sorted_model_names:
                    original_value = models_metrics[model_name][task_metric_key]
                    row.append(f"{original_value:.4f}" if original_value is not None else "-")

                table_data.append(row)

    # Create column names with SVG indicators for model columns
    columns = ["Task", "Metric"]
    for i, model_name in enumerate(sorted_model_names):
        color = colors[i % len(colors)]
        linestyle = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]

        # Generate SVG indicator
        svg_indicator = generate_line_style_svg(color, linestyle, marker)

        # Create column name with SVG above model name
        # Use a special marker to indicate this column has SVG
        display_name = model_name.replace('results_', '')
        column_name = f"__SVG__{svg_indicator}__NAME__{display_name}"
        columns.append(column_name)

    # Create DataFrame
    df = pd.DataFrame(table_data, columns=columns)

    return df


def generate_radar_chart_and_table(eval_result_paths: List[str],
                                   show_sid: bool = True,
                                   show_pid: bool = False,
                                   grounding: bool = False) -> Tuple[str, pd.DataFrame, Dict[str, str]]:
    """
    Generate radar chart and summary table from eval_results.json files

    Args:
        eval_result_paths: List of paths to eval_results.json files
        show_sid: Whether to show SID metrics in summary table (default: True)
        show_pid: Whether to show PID metrics in summary table (default: False)
        grounding: Whether to strip sidmodel_ prefix from metrics (default: False)

    Returns:
        Tuple of (radar_chart_png_path, summary_dataframe, model_color_map)
    """
    # Load data
    models_data = load_model_data(eval_result_paths, strip_sidmodel_prefix=grounding)

    # Extract metrics for radar chart (single metric per task)
    models_metrics_radar, task_metrics_radar = extract_metrics(models_data, use_radar_metrics=True)

    # Extract metrics for table (multiple metrics per task)
    models_metrics_table, task_metrics_table = extract_metrics(models_data, use_radar_metrics=False)

    # Normalize with fixed ranges (for radar)
    normalized_data_fixed, task_ranges_fixed = normalize_metrics(models_metrics_radar, task_metrics_radar)

    # Normalize with dynamic ranges (for radar)
    normalized_data_dynamic, task_ranges_dynamic = normalize_metrics_dynamic(models_metrics_radar, task_metrics_radar)

    # Define unified style configuration for both radar chart and table
    # Sort model names
    model_names = list(models_metrics_radar.keys())
    sorted_model_names = sort_model_names_by_step(model_names)

    # Create style configuration dict
    colors = [
        '#FF0000', '#0000FF', '#00AA00', '#FF8C00', '#8B008B',
        '#00CED1', '#FFD700', '#FF1493', '#32CD32', '#8A2BE2',
        '#FF6347', '#4682B4', '#DC143C', '#00FF7F', '#FF4500',
        '#9370DB', '#20B2AA', '#F0E68C', '#FF69B4', '#87CEEB'
    ]

    style_config = {
        'colors': colors,
        'line_styles': [
            '-', '--', '-.', ':',
            '-', '--', '-.', ':',
            (0, (3, 1, 1, 1)), (0, (5, 5)),
            (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1)),
        ],
        'markers': [
            'o', 's', '^', 'D', 'v', 'p', '*', 'h',
            'H', '+', 'x', '<', '>', 'd', '|', '_',
            'P', 'X', '.', ','
        ],
        'sorted_model_names': sorted_model_names
    }

    # Create model to color mapping
    model_color_map = {model: colors[i % len(colors)] for i, model in enumerate(sorted_model_names)}

    # Create dual radar charts (using radar metrics)
    radar_chart_path = create_radar_chart(normalized_data_fixed, normalized_data_dynamic,
                                         models_metrics_radar, task_metrics_radar,
                                         task_ranges_fixed, task_ranges_dynamic,
                                         style_config)

    # Create summary table (using table metrics with multiple metrics per task)
    summary_table = create_summary_table(models_metrics_table, task_metrics_table, style_config,
                                        show_sid=show_sid, show_pid=show_pid)

    return radar_chart_path, summary_table, model_color_map
