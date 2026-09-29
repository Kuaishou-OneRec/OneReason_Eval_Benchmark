"""
vLLM 批量推理接口

支持单模型和多模型并行评估。
"""

from .interface import (
    VllmInferenceInterface,
    MultiModelEvaluator,
    GPUScheduler,
    ModelConfig,
    evaluate_models_from_file
)

__all__ = [
    'VllmInferenceInterface',
    'MultiModelEvaluator',
    'GPUScheduler',
    'ModelConfig',
    'evaluate_models_from_file'
]
