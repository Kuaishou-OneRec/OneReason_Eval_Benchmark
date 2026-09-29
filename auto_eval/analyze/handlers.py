"""
Business logic for results analysis
"""
import html
import json
import random
import re
from typing import List, Dict, Tuple, Optional
import gradio as gr

from auto_eval.config import RESULT_ROOTS, EVAL_RESULTS_FILENAME, TEST_GENERATED_FILENAME, MAX_SAMPLES_TO_COMPARE, EXTRA_VERSIONS
from auto_eval.utils import safe_json_load
from auto_eval.tasks_meta import get_all_versions


def get_result_directories() -> List[str]:
    """
    Get list of result directories from all RESULT_ROOTS.

    Searches each root in order and deduplicates directory names so that a
    directory present in multiple roots appears only once in the returned list.

    Returns:
        List of directory names (not full paths), sorted descending.
    """
    try:
        seen = set()
        directories = []
        for results_root in RESULT_ROOTS:
            if not results_root.exists():
                print(f"[WARNING] Results root does not exist: {results_root}")
                continue
            for version in get_all_versions() + EXTRA_VERSIONS:
                version_dir = results_root / version
                if not version_dir.exists():
                    continue
                for d in version_dir.iterdir():
                    if d.is_dir() and d.name.startswith('results_') and d.name not in seen:
                        if (d / EVAL_RESULTS_FILENAME).exists():
                            seen.add(d.name)
                            directories.append(d.name)
        # Sort by name descending
        directories.sort(reverse=True)
        return directories
    except Exception as e:
        print(f"[ERROR] Error reading result directories: {e}")
        return []


def load_eval_results(directory_names: List[str]) -> Dict[str, Dict]:
    """
    Load eval_results.json from selected directories

    Args:
        directory_names: List of directory names (not full paths)

    Returns:
        Dictionary mapping directory name to eval results data
    """
    results = {}
    missing_files = []

    for dir_name in directory_names:
        # Search across all result roots and versions; first match wins
        eval_file = None
        for results_root in RESULT_ROOTS:
            for version in get_all_versions() + EXTRA_VERSIONS:
                candidate = results_root / version / dir_name / EVAL_RESULTS_FILENAME
                if candidate.exists():
                    eval_file = candidate
                    break
            if eval_file is not None:
                break

        if eval_file is None:
            missing_files.append(dir_name)
            continue

        data = safe_json_load(eval_file)
        if data:
            results[dir_name] = data
        else:
            missing_files.append(dir_name)

    if missing_files:
        gr.Warning(f"⚠️ 以下目录缺少 {EVAL_RESULTS_FILENAME}:\n" + '\n'.join(f"- {f}" for f in missing_files))

    return results


def load_test_generated_samples(directory_names: List[str], dataset_name: str) -> Dict[str, Dict]:
    """
    Load test_generated.json samples for a specific dataset from selected directories
    
    Directory structure: <result_root>/<version>/dir_name/model_name/dataset_name/test_generated.json
    
    Args:
        directory_names: List of directory names (dir_name)
        dataset_name: Dataset name (e.g., 'gsm8k', 'math_500')
    
    Returns:
        Dictionary mapping directory name to test_generated data
    """
    results = {}
    missing_files = []

    for dir_name in directory_names:
        # Search across all result roots and versions; first match wins
        dir_path = None
        for results_root in RESULT_ROOTS:
            for version in get_all_versions() + EXTRA_VERSIONS:
                candidate = results_root / version / dir_name
                if candidate.exists():
                    dir_path = candidate
                    break
            if dir_path is not None:
                break

        if dir_path is None:
            missing_files.append(f"{dir_name} (directory not found)")
            continue

        # Find model_name subdirectory (usually only one, e.g., 'converted')
        model_dirs = [d for d in dir_path.iterdir() if d.is_dir()]

        if not model_dirs:
            missing_files.append(f"{dir_name} (no model directory found)")
            continue
        
        # Use the first model directory (usually only one)
        model_name = model_dirs[0].name
        
        # Construct path: dir_name/model_name/dataset_name/test_generated.json
        test_file = dir_path / model_name / dataset_name / TEST_GENERATED_FILENAME
        
        if test_file.exists():
            data = safe_json_load(test_file)
            if data:
                results[dir_name] = data
            else:
                missing_files.append(f"{dir_name} (failed to load {test_file})")
        else:
            missing_files.append(f"{dir_name} (no {dataset_name} data found)")

    if missing_files:
        gr.Warning(f"⚠️ 以下目录缺少 {dataset_name}:\n" + '\n'.join(f"- {f}" for f in missing_files))

    return results


def _get_sample_key(sample_data: dict, fallback_key: str) -> str:
    """从 metadata 中按优先级提取 sample key"""
    metadata = sample_data.get('metadata', {})
    for field in ['user_id', 'uid', 'pid', 'source_pid']:
        if field in metadata and metadata[field] is not None:
            return str(metadata[field])
    return fallback_key


def compare_samples(test_generated_data: Dict[str, Dict], max_samples: int = MAX_SAMPLES_TO_COMPARE, seen_sample_ids: set = None) -> Tuple[List[Dict], set]:
    """
    Compare samples across different models

    Args:
        test_generated_data: Dictionary mapping directory name to test_generated data
        max_samples: Maximum number of samples to compare
        seen_sample_ids: Set of sample IDs already seen in this session

    Returns:
        Tuple of (List of sample comparison dictionaries, Updated set of seen sample IDs)
    """
    if seen_sample_ids is None:
        seen_sample_ids = set()

    if not test_generated_data:
        return [], seen_sample_ids

    # Rekey samples using metadata fields (user_id > uid > pid > source_pid)
    for dir_name, data in test_generated_data.items():
        if 'samples' in data:
            new_samples = {}
            for orig_key, sample_data in data['samples'].items():
                new_key = _get_sample_key(sample_data, orig_key)
                new_samples[new_key] = sample_data
            data['samples'] = new_samples

    # Get sample IDs from all models
    all_sample_ids = set()
    for dir_name, data in test_generated_data.items():
        if 'samples' in data:
            all_sample_ids.update(data['samples'].keys())

    # Find common sample IDs across all models
    common_sample_ids = set(all_sample_ids)
    for dir_name, data in test_generated_data.items():
        if 'samples' in data:
            model_sample_ids = set(data['samples'].keys())
            common_sample_ids = common_sample_ids.intersection(model_sample_ids)

    if not common_sample_ids:
        gr.Warning("⚠️ 所选模型没有共同的样本ID")
        return [], seen_sample_ids

    # Filter out already seen samples
    available_sample_ids = list(common_sample_ids - seen_sample_ids)
    
    # If no available samples (all seen), reset history and use all common samples
    if not available_sample_ids:
        available_sample_ids = list(common_sample_ids)
        seen_sample_ids = set() # Reset history
        
    # Randomly select samples if there are more than max_samples
    selected_sample_ids = available_sample_ids
    if len(selected_sample_ids) > max_samples:
        selected_sample_ids = random.sample(selected_sample_ids, max_samples)
    else:
        # Sort for consistent display if taking all
        selected_sample_ids.sort(key=lambda x: int(x) if x.isdigit() else x)

    # Update seen samples
    seen_sample_ids.update(selected_sample_ids)

    # Build comparison data
    comparisons = []

    for sample_id in selected_sample_ids:
        comparison = {
            'sample_id': sample_id,
            'models': {}
        }

        for dir_name, data in test_generated_data.items():
            if 'samples' in data and sample_id in data['samples']:
                sample_data = data['samples'][sample_id]

                # Store all available metrics
                model_data = {
                    'prompt': sample_data.get('prompt', 'N/A'),
                    'generations': sample_data.get('generations', []),
                    'ground_truth': sample_data.get('ground_truth', 'N/A'),
                    'parsed_answer': (sample_data.get('parsed_answers') or [None])[0] or sample_data.get('parsed_answer', 'N/A'),
                    'metadata': sample_data.get('metadata', {}),
                }

                # Add all metric candidates to the model data
                metric_candidates = ['bertscore_f1', 'pass@1', 'pass@32', 'accuracy_at_pos1', 'recall@32', 'auc', 'wuauc', 'wip_double_weighted_f1', 'mean_emb_sim@32', 'max_emb_sim@32']
                for metric in metric_candidates:
                    if metric in sample_data:
                        model_data[metric] = sample_data.get(metric)

                comparison['models'][dir_name] = model_data

        comparisons.append(comparison)

    return comparisons, seen_sample_ids


def hex_to_rgba(hex_color, alpha=0.2):
    """将十六进制颜色转换为RGBA格式"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def extract_segments(text: str) -> List[str]:
    """
    Extract segments like <s_a_412><s_b_2128><s_c_188> from text.
    Matches sequences of <s_[abc]_\d+> that appear at least 3 times consecutively.
    """
    # Regex to find sequences of <s_a_...><s_b_...><s_c_...> tags
    pattern = r'<s_a_\d+><s_b_\d+><s_c_\d+>'
    matches = re.findall(pattern, text)
    return matches


def highlight_matches(text: str, segments_to_highlight: List[str]) -> str:
    """
    Highlight occurrences of segments in text.
    """
    if not segments_to_highlight:
        return html.escape(text)
    
    escaped_text = html.escape(text)
    
    for segment in segments_to_highlight:
        escaped_segment = html.escape(segment)
        # Use a unique placeholder to avoid re-replacing already highlighted parts
        # or parts of the span tag itself (though unlikely with this content)
        highlighted = f'<span style="background-color: yellow; color: red; font-weight: bold;">{escaped_segment}</span>'
        escaped_text = escaped_text.replace(escaped_segment, highlighted)
        
    return escaped_text


def safe_truncate_html(html_text: str, max_length: int) -> str:
    """
    Safely truncate HTML text without breaking entities or tags.
    """
    if len(html_text) <= max_length:
        return html_text
    
    # Truncate at max_length
    truncated = html_text[:max_length]
    
    # Check if we're in the middle of an HTML entity (&...;)
    # Find the last '&' before the cut point
    last_ampersand = truncated.rfind('&')
    if last_ampersand != -1:
        # Check if there's a ';' after it in the original text
        remaining = html_text[last_ampersand:max_length + 10]  # Look ahead a bit
        semicolon_pos = remaining.find(';')
        if semicolon_pos != -1 and semicolon_pos < 10:  # Entity should be short
            # We're potentially in the middle of an entity, cut before it
            truncated = html_text[:last_ampersand]
    
    # Check if we're in the middle of an HTML tag (<...>)
    # Find the last '<' before the cut point
    last_open = truncated.rfind('<')
    last_close = truncated.rfind('>')
    
    if last_open > last_close:
        # We have an unclosed tag, cut before it
        truncated = html_text[:last_open]
    
    return truncated


def format_comparison_for_display(comparisons: List[Dict], model_color_map: Dict[str, str] = None, metric_filter: str = None) -> str:
    """
    Format comparison data for HTML display with color-coded model names

    Args:
        comparisons: List of sample comparison dictionaries
        model_color_map: Dictionary mapping model names to their colors
        metric_filter: Optional metric name to filter and display only that metric

    Returns:
        HTML string for display
    """
    if not comparisons:
        return "<p>没有可比较的样本数据</p>"

    html_parts = []

    # 1. Add Header and Model Filter Pills
    html_parts.append("<div class='comparison-header-container'>")
    # Add metric filter information to title if provided
    title = "样本对比详情"
    if metric_filter:
        title += f" - 指标: {metric_filter}"
    html_parts.append(f"<h2 style='margin: 0;'>{title}</h2>")
    
    # Get all unique models from the first comparison (assuming all have same models)
    # We need sorted models to match the display order
    from .plot import sort_model_names_by_step
    first_comp = comparisons[0]
    model_names = list(first_comp['models'].keys())
    sorted_model_names = sort_model_names_by_step(model_names)
    
    html_parts.append("<div class='model-filter-container'>")
    for model_name in sorted_model_names:

        display_name = model_name.replace('results_', '')
        model_color = model_color_map[display_name]
            
        # Pill HTML - clickable to toggle
        # We use data-model attribute to identify which elements to hide
        light_model_color = hex_to_rgba(model_color, 0.2)
        html_parts.append(f"""
        <div class='model-pill' style='background-color: {light_model_color} !important; color: black; cursor: pointer;' id='pill_{model_name}' onclick="toggleModel('{model_name}')" title="点击隐藏/显示此模型">
            <span class='model-pill-text'>{display_name}</span>
        </div>
        """)
    html_parts.append("</div>") # End filter container
    html_parts.append("</div>") # End header container

    for comp in comparisons:
        sample_id = comp['sample_id']
        models = comp['models']

        # Sample header
        # Sample header (Collapsible)
        html_parts.append(f"<details open class='sample-details'>")
        html_parts.append(f"<summary class='sample-summary'><h3 style='display: inline-block; margin: 0; color: #333;'>样本 ID: {sample_id}</h3></summary>")
        html_parts.append(f"<div style='margin-bottom: 30px; padding: 20px; border-top: none;'>")

        # Model cards
        html_parts.append("<div style='display: flex; gap: 20px; flex-wrap: wrap;'>")

        # Get sorted model names to match radar chart order
        from .plot import sort_model_names_by_step
        model_names = list(models.keys())
        sorted_model_names = sort_model_names_by_step(model_names)

        for model_name in sorted_model_names:
            sample_data = models[model_name]

            # Determine correctness based on metric_filter or fallback logic
            is_correct = None
            metric_used = "unknown"
            is_score_metric = False  # Flag to indicate if metric is a score (not binary)

            # Use metric_filter to determine which metric to display
            if metric_filter and metric_filter in sample_data:
                is_correct = sample_data[metric_filter]
                metric_used = metric_filter
                # Determine if it's a score metric (continuous value) vs binary metric
                score_metrics = [
                    'bertscore_f1',
                    'auc', 'wuauc',
                    'wip_double_weighted_f1', 'wip_double_weighted_core_f1',
                ]
                # recall@k is also a continuous score metric
                if metric_filter in score_metrics or metric_filter.startswith('recall@'):
                    is_score_metric = True
                print(f"[INFO] {model_name} - Sample {sample_id}: using metric {metric_filter} = {is_correct}")
            else:
                # Metric not found in sample data
                print(f"[WARNING] {model_name} - Sample {sample_id}: metric '{metric_filter}' not found")
                metric_used = metric_filter or "unknown"
                is_correct = None

            # Use color from model_color_map if available, otherwise use default colors
            display_name = model_name.replace('results_', '')
            model_color = model_color_map[display_name]

            # Generate correctness/score display
            if is_correct is None:
                # Metric not found or N/A
                correctness_color = "#999"  # Gray for N/A
                correctness_text = f"⚠️ {metric_used}: N/A"
            elif is_score_metric:
                # For score metrics like bertscore, show score with neutral color
                correctness_color = "#007bff"  # Blue for scores
                if isinstance(is_correct, (int, float)):
                    correctness_text = f"📊 {metric_used}: {is_correct:.4f}"
                else:
                    correctness_text = f"📊 {metric_used}: N/A"
            else:
                # For binary metrics, show correct/incorrect with green/red
                correctness_color = "#28a745" if is_correct else "#dc3545"
                correctness_text = f"✅ 正确 ({metric_used})" if is_correct else f"❌ 错误 ({metric_used})"

            # Prepare text content
            prompt_text = sample_data['prompt']

            raw_generation_list = sample_data['generations'] if sample_data['generations'] else []
            raw_parsed_answer = sample_data.get('parsed_answer', 'N/A')
            raw_ground_truth = sample_data['ground_truth']
            
            # Highlight matching segments if correct and metric is pass@k
            display_generation_text = ""
            display_ground_truth_text = html.escape(str(raw_ground_truth))
            
            if is_correct and metric_used.startswith('pass@') and not metric_used.endswith('@1'):
                # Extract target segments from ground truth
                target_segments = extract_segments(raw_ground_truth)
                
                if target_segments:
                    # First pass: Identify which segments are actually matched in the generations
                    matched_segments = set()
                    highlighted_generations = []
                    
                    for gen in raw_generation_list:
                        # Find segments present in this generation
                        gen_matches = []
                        for segment in target_segments:
                            if segment in gen:
                                matched_segments.add(segment)
                                gen_matches.append(segment)
                        
                        # Highlight this generation with its specific matches
                        if gen_matches:
                            highlighted_generations.append(highlight_matches(gen, gen_matches))
                        else:
                            highlighted_generations.append(html.escape(gen))
                    
                    # Highlight ground truth ONLY with segments that were matched
                    if matched_segments:
                        display_ground_truth_text = highlight_matches(raw_ground_truth, list(matched_segments))
                    
                    display_generation_text = '\n'.join(highlighted_generations) if highlighted_generations else 'N/A'
                else:
                    # No segments found, fallback to normal
                    display_generation_text = html.escape('\n'.join(raw_generation_list)) if raw_generation_list else 'N/A'
            else:
                # Normal display
                display_generation_text = html.escape('\n'.join(raw_generation_list)) if raw_generation_list else 'N/A'

            # Generate unique IDs for this sample's content
            prompt_id = f"prompt_{sample_id}_{model_name.replace(' ', '_')}"
            gen_id = f"gen_{sample_id}_{model_name.replace(' ', '_')}"
            parsed_id = f"parsed_{sample_id}_{model_name.replace(' ', '_')}"
            gt_id = f"gt_{sample_id}_{model_name.replace(' ', '_')}"

            # SVG copy icon
            copy_icon_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>'

            # Build expand/collapse links (inside pre tags)
            sq = "'"  # single quote for onclick attributes
            prompt_expand_link = f'\n<span class="expand-text-link" onclick="toggleText({sq}{prompt_id}{sq})">展开</span>' if len(prompt_text) > 500 else ''
            prompt_collapse_link = f'\n<span class="collapse-text-link" onclick="toggleText({sq}{prompt_id}{sq})" style="display:none;">收起</span>'

            # Note: We use display_generation_text which is already escaped and highlighted
            # We need to be careful not to double escape or break HTML
            # Since we manually handled escaping/highlighting, we should NOT escape again in the f-string below
            
            # For length check, we should use the raw text length
            raw_gen_len = len('\n'.join(raw_generation_list))
            gen_expand_link = f'\n<span class="expand-text-link" onclick="toggleText({sq}{gen_id}{sq})">展开</span>' if raw_generation_list and raw_gen_len > 500 else ''
            gen_collapse_link = f'\n<span class="collapse-text-link" onclick="toggleText({sq}{gen_id}{sq})" style="display:none;">收起</span>'

            parsed_expand_link = f'\n<span class="expand-text-link" onclick="toggleText({sq}{parsed_id}{sq})">展开</span>' if raw_parsed_answer != 'N/A' and len(raw_parsed_answer) > 300 else ''
            parsed_collapse_link = f'\n<span class="collapse-text-link" onclick="toggleText({sq}{parsed_id}{sq})" style="display:none;">收起</span>'

            gt_expand_link = f'\n<span class="expand-text-link" onclick="toggleText({sq}{gt_id}{sq})">展开</span>' if len(raw_ground_truth) > 300 else ''
            gt_collapse_link = f'\n<span class="collapse-text-link" onclick="toggleText({sq}{gt_id}{sq})" style="display:none;">收起</span>'

            # Model card
            html_parts.append(f"""
            <div class='model-card' data-model='{model_name}' style='flex: 1; min-width: 300px; border: 1px solid #ccc; border-radius: 16px; padding: 15px; background-color: #f9f9f9;'>
                <h4 style='margin-top: 0; color: {model_color};'>{model_name.replace('results_', '')}</h4>
                <p style='color: {correctness_color}; font-weight: bold;'>{correctness_text}</p>

                <details open>
                    <summary style='cursor: pointer; font-weight: bold; margin-bottom: 10px; margin-top: 10px;'>Prompt</summary>
                    <div class='text-container'>
                        <button class="copy-icon-btn" onclick="copyText('{prompt_id}')" title="复制">
                            {copy_icon_svg}
                        </button>
                        <div class='text-preview' id='{prompt_id}_preview'>
                            <pre>{html.escape(prompt_text[:500])}{'...' if len(prompt_text) > 500 else ''}{prompt_expand_link}</pre>
                        </div>
                        <div class='text-full' id='{prompt_id}_full' style='display: none;'>
                            <pre>{html.escape(prompt_text)}{prompt_collapse_link}</pre>
                        </div>
                    </div>
                </details>

                <details open>
                    <summary style='cursor: pointer; font-weight: bold; margin-bottom: 10px; margin-top: 10px;'>Generation</summary>
                    <div class='text-container'>
                        <button class="copy-icon-btn" onclick="copyText('{gen_id}')" title="复制">
                            {copy_icon_svg}
                        </button>
                        <div class='text-preview' id='{gen_id}_preview'>
                            <pre>{safe_truncate_html(display_generation_text, 500)}{'...' if raw_gen_len > 500 else ''}{gen_expand_link}</pre>
                        </div>
                        <div class='text-full' id='{gen_id}_full' style='display: none;'>
                            <pre>{display_generation_text}{gen_collapse_link}</pre>
                        </div>
                    </div>
                </details>
            """)

            # Only show Parsed Answer if it exists
            if raw_parsed_answer != 'N/A':
                html_parts.append(f"""
                <details open>
                    <summary style='cursor: pointer; font-weight: bold; margin-bottom: 10px; margin-top: 10px;'>Parsed Answer</summary>
                    <div class='text-container'>
                        <button class="copy-icon-btn" onclick="copyText('{parsed_id}')" title="复制">
                            {copy_icon_svg}
                        </button>
                        <div class='text-preview' id='{parsed_id}_preview'>
                            <pre>{html.escape(raw_parsed_answer[:300])}{'...' if len(raw_parsed_answer) > 300 else ''}{parsed_expand_link}</pre>
                        </div>
                        <div class='text-full' id='{parsed_id}_full' style='display: none;'>
                            <pre>{html.escape(raw_parsed_answer)}{parsed_collapse_link}</pre>
                        </div>
                    </div>
                </details>
                """)

            html_parts.append(f"""
                <details open>
                    <summary style='cursor: pointer; font-weight: bold; margin-bottom: 10px; margin-top: 10px;'>Ground Truth</summary>
                    <div class='text-container'>
                        <button class="copy-icon-btn" onclick="copyText('{gt_id}')" title="复制">
                            {copy_icon_svg}
                        </button>
                        <div class='text-preview' id='{gt_id}_preview'>
                            <pre>{safe_truncate_html(display_ground_truth_text, 300)}{'...' if len(raw_ground_truth) > 300 else ''}{gt_expand_link}</pre>
                        </div>
                        <div class='text-full' id='{gt_id}_full' style='display: none;'>
                            <pre>{display_ground_truth_text}{gt_collapse_link}</pre>
                        </div>
                    </div>
                </details>
            </div>
            """)

        html_parts.append("</div>")  # End model cards
        html_parts.append("</div>")  # End sample container
        html_parts.append("</details>")  # End sample details

    return "".join(html_parts)


def get_loading_html() -> str:
    """
    Get loading indicator HTML

    Returns:
        HTML string with loading message
    """
    return """
    <div class="loading-container">
        <div class="loading-text">正在加载样本数据...</div>
    </div>
    """


def dataframe_to_html(df) -> str:
    """
    Convert pandas DataFrame to custom HTML table with SVG line style indicators
    
    Args:
        df: Pandas DataFrame to convert
        
    Returns:
        HTML string
    """
    if df.empty:
        return ""
    
    html = ['<table class="custom-summary-table">']
    
    # Header
    html.append('<thead><tr>')
    for col in df.columns:
        if col == '查看':
            html.append('<th>操作</th>')
        elif col.startswith('__SVG__'):
            # Parse column name with SVG indicator
            # Format: __SVG__<svg>...</svg>__NAME__<model_name>
            parts = col.split('__NAME__')
            if len(parts) == 2:
                svg_part = parts[0].replace('__SVG__', '')
                model_name = parts[1]
                # Create header with SVG above model name
                html.append(f'<th><div class="line-indicator">{svg_part}</div><div class="model-name">{model_name}</div></th>')
            else:
                # Fallback if parsing fails
                html.append(f'<th>{col}</th>')
        else:
            html.append(f'<th>{col}</th>')
    html.append('</tr></thead>')
    
    # Body
    html.append('<tbody>')
    
    # Track current task for grouping and styling
    current_task = None
    task_group_index = 0
    task_row_count = 0
    
    for _, row in df.iterrows():
        task_name = row.get('Task', '')
        
        # Check if we're starting a new task group
        if task_name != current_task:
            current_task = task_name
            task_group_index += 1
            task_row_count = 0
        
        # Determine row class based on task group
        row_class = 'task-group-even' if task_group_index % 2 == 0 else 'task-group-odd'
        is_first_row = (task_row_count == 0)
        
        # Add class for first row of a task group (for border)
        if is_first_row:
            row_class += ' task-group-first'
        
        html.append(f'<tr class="{row_class}">')
        
        for col in df.columns:
            val = row[col]
            if col == '查看':
                # Use the Task column as the dataset name and Metric column for filtering
                dataset_name = row.get('Task', '')
                metric_name = row.get('Metric', '')
                html.append(f'<td><button class="view-btn" onclick="selectDataset(\'{dataset_name}\', \'{metric_name}\')">查看详情</button></td>')
            elif col == 'Task':
                # Only show task name in first row of each task group
                if is_first_row:
                    html.append(f'<td class="task-name-cell"><strong>{val}</strong></td>')
                else:
                    html.append('<td class="task-name-cell"></td>')
            else:
                html.append(f'<td>{val}</td>')
        html.append('</tr>')
        task_row_count += 1
        
    html.append('</tbody></table>')
    
    return "".join(html)
