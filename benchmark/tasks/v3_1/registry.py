"""The 10 competition tasks for data v3.1."""
from typing import Dict, Any, Optional
from benchmark.tasks.base_loader import BaseLoader
from benchmark.tasks.task_registration import TaskRegistration
from .config import VIDEO_CONFIG, AD_CONFIG, LIVE_CONFIG, PRODUCT_CONFIG, CHALLENGE_SID2CAPTION_VIDEO_CONFIG, CHALLENGE_SID2CAPTION_PRODUCT_CONFIG, CHALLENGE_SID2CAPTION_AD_CONFIG
from benchmark.tasks.v2_0.tongshi_choice.config import TONGSHI_CHOICE_CONFIG
from benchmark.tasks.v2_0.evolution_select.config import EVOLUTION_SELECT_CONFIG
from benchmark.tasks.v2_0.evolution_topic_gen.config import EVOLUTION_TOPIC_GEN_CONFIG

TASK_REGISTRY: Dict[str, TaskRegistration] = {
    'challenge_common_sense': TaskRegistration(
        name="challenge_common_sense",
        config=TONGSHI_CHOICE_CONFIG,
        evaluator_class="benchmark.tasks.v2_0.tongshi_choice.evaluator.TongshiChoiceEvaluator",
        category="general",
        show_in_web_ui=True,
        metrics=['pass@1'],
        radar_metric='pass@1',
        radar_range_min=0.0,
        radar_range_max=1.0,
    ),
    'challenge_itemic_pattern_caption_video': TaskRegistration(
        name="challenge_itemic_pattern_caption_video",
        config=CHALLENGE_SID2CAPTION_VIDEO_CONFIG,
        evaluator_class="benchmark.tasks.v3_1.challenge_sid2caption_evaluator.ChallengeSid2CaptionEvaluator",
        category="perception",
        show_in_web_ui=True,
        metrics=['macro_wip_double_weighted_f1', 'macro_wip_double_weighted_core_f1'],
        radar_metric='macro_wip_double_weighted_f1',
        radar_range_min=0.0,
        radar_range_max=0.8,
    ),
    'challenge_itemic_pattern_caption_product': TaskRegistration(
        name="challenge_itemic_pattern_caption_product",
        config=CHALLENGE_SID2CAPTION_PRODUCT_CONFIG,
        evaluator_class="benchmark.tasks.v3_1.challenge_sid2caption_evaluator.ChallengeSid2CaptionEvaluator",
        category="perception",
        show_in_web_ui=True,
        metrics=['macro_wip_double_weighted_f1', 'macro_wip_double_weighted_core_f1'],
        radar_metric='macro_wip_double_weighted_f1',
        radar_range_min=0.0,
        radar_range_max=0.8,
    ),
    'challenge_itemic_pattern_caption_ad': TaskRegistration(
        name="challenge_itemic_pattern_caption_ad",
        config=CHALLENGE_SID2CAPTION_AD_CONFIG,
        evaluator_class="benchmark.tasks.v3_1.challenge_sid2caption_evaluator.ChallengeSid2CaptionEvaluator",
        category="perception",
        show_in_web_ui=True,
        metrics=['macro_wip_double_weighted_f1', 'macro_wip_double_weighted_core_f1'],
        radar_metric='macro_wip_double_weighted_f1',
        radar_range_min=0.0,
        radar_range_max=0.8,
    ),
    'challenge_evolution_action_select': TaskRegistration(
        name="challenge_evolution_action_select",
        config=EVOLUTION_SELECT_CONFIG,
        evaluator_class="benchmark.tasks.v2_0.evolution_select.evaluator.EvolutionSelectEvaluator",
        category="evolution",
        show_in_web_ui=True,
        metrics=['precision', 'recall', 'f1'],
        radar_metric='f1',
        radar_range_min=0.0,
        radar_range_max=1.0,
    ),
    'challenge_evolution_topic_gen': TaskRegistration(
        name="challenge_evolution_topic_gen",
        config=EVOLUTION_TOPIC_GEN_CONFIG,
        evaluator_class="benchmark.tasks.v2_0.evolution_topic_gen.evaluator.EvolutionTopicGenEvaluator",
        category="evolution",
        show_in_web_ui=True,
        metrics=['action_f1_score', 'logic_f1_score', 'combined_f1_score'],
        radar_metric='combined_f1_score',
        radar_range_min=0.0,
        radar_range_max=1.0,
    ),
    'challenge_recommendation_video_full': TaskRegistration(
        name="challenge_recommendation_video_full",
        config=VIDEO_CONFIG,
        evaluator_class="benchmark.tasks.v2_0.recommendation.evaluator.RecommendationEvaluator",
        category="recommendation",
        show_in_web_ui=True,
        metrics=['pass@32', 'recall@32', 'pass@64', 'recall@64'],
        radar_metric='pass@32',
        radar_range_min=0.0,
        radar_range_max=0.8,
        affected_by_sid_pid=True,
    ),
    'challenge_recommendation_product': TaskRegistration(
        name="challenge_recommendation_product",
        config=PRODUCT_CONFIG,
        evaluator_class="benchmark.tasks.v2_0.recommendation.evaluator.RecommendationEvaluator",
        category="recommendation",
        show_in_web_ui=True,
        metrics=['pass@32', 'recall@32', 'pass@64', 'recall@64'],
        radar_metric='pass@32',
        radar_range_min=0.0,
        radar_range_max=0.8,
        affected_by_sid_pid=True,
    ),
    'challenge_recommendation_ad': TaskRegistration(
        name="challenge_recommendation_ad",
        config=AD_CONFIG,
        evaluator_class="benchmark.tasks.v2_0.recommendation.evaluator.RecommendationEvaluator",
        category="recommendation",
        show_in_web_ui=True,
        metrics=['pass@32', 'recall@32', 'pass@64', 'recall@64'],
        radar_metric='pass@32',
        radar_range_min=0.0,
        radar_range_max=0.8,
        affected_by_sid_pid=True,
    ),
    'challenge_recommendation_live': TaskRegistration(
        name="challenge_recommendation_live",
        config=LIVE_CONFIG,
        evaluator_class="benchmark.tasks.v2_0.recommendation.evaluator.RecommendationEvaluator",
        category="recommendation",
        show_in_web_ui=True,
        metrics=['pass@32', 'recall@32', 'pass@64', 'recall@64'],
        radar_metric='pass@32',
        radar_range_min=0.0,
        radar_range_max=0.8,
        affected_by_sid_pid=True,
    ),
}

def get_loader(task_name: str, data_dir: str, tokenizer: Optional[Any] = None, enable_thinking: Optional[bool] = None, custom_chat_template: Optional[str] = None):
    """Get loader instance for a task"""
    if task_name not in TASK_REGISTRY:
        available_tasks = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {available_tasks}"
        )

    reg = TASK_REGISTRY[task_name]

    return BaseLoader(
        task_config=reg.config,
        data_dir=data_dir,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        task_name=task_name,
        custom_chat_template=custom_chat_template
    )


def get_evaluator(task_name: str):
    """Get evaluator class for a task"""
    if task_name not in TASK_REGISTRY:
        available_tasks = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {available_tasks}"
        )

    from benchmark.tasks.task_registration import resolve_evaluator_class
    return resolve_evaluator_class(TASK_REGISTRY[task_name].evaluator_class)


def get_task_config(task_name: str) -> Dict[str, Any]:
    """Get task configuration"""
    if task_name not in TASK_REGISTRY:
        available_tasks = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {available_tasks}"
        )

    return TASK_REGISTRY[task_name].config


def get_all_tasks() -> list:
    """Get list of all registered task names"""
    return list(TASK_REGISTRY.keys())


def get_tasks_by_category(category: str) -> list:
    """Get tasks filtered by category"""
    return [
        name for name, reg in TASK_REGISTRY.items()
        if reg.category == category
    ]

# ========================================
# Backward Compatibility
# ========================================

TaskTable = {name: reg.config for name, reg in TASK_REGISTRY.items()}
