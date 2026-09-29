"""
Shared TaskRegistration dataclass used by all version registries.

This is the single source of truth for task metadata, including
both evaluation config and analysis/plot metadata.
"""

import importlib
from dataclasses import dataclass, field
from typing import Type, Dict, Any, List, Optional


@dataclass
class TaskRegistration:
    """Task registration information"""
    name: str
    config: Dict[str, Any]
    evaluator_class: str  # Lazy string path, e.g. "benchmark.tasks.v2_0.recommendation.evaluator.RecommendationEvaluator"
    category: str  # "general", "recommendation", "caption", "evolution", "perception", "derivation", etc.
    show_in_web_ui: bool = False
    # ---- Analysis / plot metadata ----
    metrics: Optional[List[str]] = None       # Table metrics list, e.g. ['pass@32', 'recall@32']
    radar_metric: str = ""                    # Primary metric for radar chart, e.g. 'pass@32'
    radar_range_min: float = 0.0              # Radar chart normalization min
    radar_range_max: float = 1.0              # Radar chart normalization max
    affected_by_sid_pid: bool = False          # Whether affected by SID/PID toggle


def resolve_evaluator_class(cls_path: str) -> Type:
    """Lazily resolve an evaluator class from its dotted string path."""
    module_path, class_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
