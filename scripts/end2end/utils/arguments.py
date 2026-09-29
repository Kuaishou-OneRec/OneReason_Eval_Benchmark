"""CLI argument dataclasses for End2End evaluation."""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class End2EndModelConfig:
    model_path: str = field(
        metadata={"help": "Parent directory containing converted_encoder/, converted_decoder/, converted_projection/"},
    )
    model_name: str = field(
        default="end2end",
        metadata={"help": "Model name for result directory naming"},
    )
    num_query_tokens: int = field(
        default=4,
        metadata={"help": "Number of learnable query tokens"},
    )
    nframe: int = field(
        default=64,
        metadata={"help": "Max number of video frames to sample"},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "System prompt for decoder"},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Use flash attention 2"},
    )

    @property
    def encoder_path(self) -> str:
        return os.path.join(self.model_path, "converted_encoder")

    @property
    def decoder_path(self) -> str:
        return os.path.join(self.model_path, "converted_decoder")

    @property
    def projection_path(self) -> str:
        return os.path.join(self.model_path, "converted_projection")


@dataclass
class InferenceConfig:
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Max new tokens to generate"},
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature"},
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p sampling"},
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to sample"},
    )


@dataclass
class PromptConfig:
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Enable thinking mode for apply_chat_template"},
    )
    custom_chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "Custom chat template filename under benchmark/tasks/templates/"},
    )


@dataclass
class BenchmarkConfig:
    task_types: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Task name list"},
    )
    sample_ratio: float = field(
        default=1.0,
        metadata={"help": "Evaluation ratio in (0, 1]. Uses the first ceil(N * ratio) samples."},
    )
    splits: List[str] = field(
        default_factory=lambda: ["test"],
        metadata={"help": "Dataset split list"},
    )
    data_dir: str = field(
        default="./data",
        metadata={"help": "Data directory path"},
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "Output directory"},
    )
    overwrite: bool = field(
        default=False,
        metadata={"help": "Overwrite existing results"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed"},
    )
    version: Optional[str] = field(
        default="v4.0",
        metadata={"help": "Benchmark version"},
    )
