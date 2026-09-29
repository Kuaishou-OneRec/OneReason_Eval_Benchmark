from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model loading and initialization parameters"""
    model_path: str = field(
        metadata={"help": "Model path or HuggingFace model name (e.g., Qwen/Qwen2-7B)", "required": True}
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "PT checkpoint path (optional, for loading .pt format models, will auto-convert to HuggingFace format)"}
    )
    dtype: str = field(
        default='bfloat16',
        metadata={"help": "Model data type: auto, float16, bfloat16, float32"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code"}
    )


@dataclass
class InfrastructureConfig:
    """Hardware and distributed computing configuration"""
    # Parallelism
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size (default 1, single GPU per worker)"}
    )
    # GPU allocation
    num_gpus: Optional[int] = field(
        default=None,
        metadata={"help": "Number of GPUs to use (default uses all visible GPUs)"}
    )
    gpu_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of GPU IDs to use (e.g., [0,2,4], default uses all visible GPUs)"}
    )
    # Ray cluster
    ray_address: Optional[str] = field(
        default="auto",
        metadata={"help": "Ray cluster address: 'auto' (auto-detect), 'local' (single machine), or 'ray://head_ip:10001' (specific cluster address)"}
    )
    allow_cross_node_tensor_parallel: bool = field(
        default=False,
        metadata={"help": "Allow tensor parallelism across different nodes (not recommended due to network latency)"}
    )


@dataclass
class InferenceConfig:
    """Inference execution parameters"""
    # Batch processing
    worker_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for each worker to process prompts"}
    )


@dataclass
class PromptConfig:
    """Prompt formatting and template parameters"""
    # Thinking mode (affects both template and generation)
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Enable thinking mode for apply_chat_template (overrides task config if set)"}
    )
    force_thinking: bool = field(
        default=False,
        metadata={"help": "Force thinking mode - output <think> tag when enable_thinking is True (overrides task config if set)"}
    )
    # Chat templates
    soft_switch: bool = field(
        default=False,
        metadata={"help": "Use soft switch chat template (qwen3_soft_switch.jinja2) instead of default (qwen3.jinja2)"}
    )


@dataclass
class BenchmarkConfig:
    """Benchmark execution and evaluation parameters"""
    # Task selection
    task_types: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Task name list (e.g., item_understand user_summary)"}
    )
    sample_ratio: float = field(
        default=1.0,
        metadata={"help": "Evaluation ratio in (0, 1]. Uses the first ceil(N * ratio) samples."}
    )
    splits: List[str] = field(
        default_factory=lambda: ['test'],
        metadata={"help": "Dataset split list"}
    )
    # Data I/O
    data_dir: str = field(
        default='./data',
        metadata={"help": "Data directory path"}
    )
    output_dir: str = field(
        default='./results',
        metadata={"help": "Output directory for results"}
    )
    overwrite: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite existing results"}
    )
    # Evaluation control
    select_k: Optional[str] = field(
        default=None,
        metadata={"help": "Strategy for selecting k generations for metric computation: 'first_k' (use first k, default) or 'top_k_by_logprobs' (select top k by logprobs)"}
    )
    # Reproducibility
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed for reproducibility (set to None to disable)"}
    )
    # Version selection
    benchmark_version: Optional[str] = field(
        default=None,
        metadata={"help": "Benchmark version to use (e.g., 'v1.0', 'v2.0'). If not set, uses merged task table (v2.0 priority on conflicts)."}
    )
