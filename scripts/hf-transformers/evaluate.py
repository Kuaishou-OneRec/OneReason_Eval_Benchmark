"""
Benchmark Generation Task Evaluation Script

Usage:
    # Evaluate all tasks
    python evaluate.py \
        --model_path Qwen/Qwen2-7B \
        --data_dir ./data \
        --output_dir ./results
    
    # Use PT checkpoint
    python evaluate.py \
        --model_path /path/to/model/config \
        --checkpoint_path /path/to/checkpoint.pt \
        --torch_dtype bfloat16 \
        --data_dir ./data \
        --output_dir ./results \
        --batch_size 4
"""
from transformers import HfArgumentParser
import torch

from benchmark import Benchmark
from benchmark.console import *
from utils.generator import HfTransformersGenerator
from utils.arguments import (
    ModelConfig,
    InfrastructureConfig,
    InferenceConfig,
    PromptConfig,
    BenchmarkConfig
)


def main():
    parser = HfArgumentParser([
        ModelConfig,
        InfrastructureConfig,
        InferenceConfig,
        PromptConfig,
        BenchmarkConfig
    ])
    model_config, infra_config, inference_config, prompt_config, benchmark_config = \
        parser.parse_args_into_dataclasses()

    # 1. Initialize Benchmark
    benchmark = Benchmark(
        model_path=model_config.model_path,
        task_types=benchmark_config.task_types,
        splits=benchmark_config.splits,
        data_dir=benchmark_config.data_dir,
        enable_thinking=prompt_config.enable_thinking,
        force_thinking=prompt_config.force_thinking,
        soft_switch=prompt_config.soft_switch,
        seed=benchmark_config.seed
    )
    # Benchmark.print_benchmark_table()

    # 2. Initialize generator

    dtype_map = {
        'auto': None,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    torch_dtype = dtype_map[model_config.dtype]

    generator = HfTransformersGenerator(
        model_name_or_path=model_config.model_path,
        checkpoint_path=model_config.checkpoint_path,
        device=infra_config.device,
        batch_size=inference_config.batch_size,
        torch_dtype=torch_dtype,
        worker_batch_size=inference_config.worker_batch_size,
        task_types=benchmark_config.task_types,  # Pass task types for potential future optimizations
    )

    # 3. Generate text
    benchmark.run(
        generator=generator,
        output_dir=benchmark_config.output_dir,
        overwrite=benchmark_config.overwrite,
        sample_ratio=benchmark_config.sample_ratio,
        enable_thinking=prompt_config.enable_thinking,  # Pass to generator.generate()
        force_thinking=prompt_config.force_thinking     # Pass to generator.generate()
    )

    # 4. Release GPU memory occupied by model
    console.print("\nReleasing model GPU memory...", style=warning_style)
    del generator
    import gc
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    console.print("✓ GPU memory release completed\n", style=success_style)

    # 5. Calculate evaluation metrics
    eval_results_path = f"{benchmark_config.output_dir}/eval_results.json"
    Benchmark.evaluate_dev(
        generation_results_dir=benchmark_config.output_dir,
        output_path=eval_results_path,
        data_dir=benchmark_config.data_dir,
        overwrite=benchmark_config.overwrite,
        select_k=benchmark_config.select_k,
        task_types=benchmark_config.task_types
    )


if __name__ == "__main__":
    main()
