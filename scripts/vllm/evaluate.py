"""
Benchmark Generation Task Evaluation Script (vLLM Version)

Uses vLLM for high-speed inference with significant performance improvements over HuggingFace Transformers.

Usage:
    # Evaluate all tasks (single GPU)
    python evaluate.py \
        --model_path Qwen/Qwen2-7B \
        --data_dir ./data \
        --output_dir ./results
    
    # Use multiple GPUs (tensor parallelism)
    python evaluate.py \
        --model_path Qwen/Qwen2-7B \
        --tensor_parallel_size 2 \
        --data_dir ./data \
        --output_dir ./results
    
    # Use PT checkpoint (will auto-convert)
    python evaluate.py \
        --model_path /path/to/model/config \
        --checkpoint_path /path/to/checkpoint.pt \
        --dtype bfloat16 \
        --data_dir ./data \
        --output_dir ./results
    
    # Adjust GPU memory utilization and context length
    python evaluate.py \
        --model_path Qwen/Qwen2-7B \
        --gpu_memory_utilization 0.95 \
        --max_model_len 8192 \
        --data_dir ./data \
        --output_dir ./results
"""
from transformers import HfArgumentParser

from benchmark import Benchmark
from benchmark.console import *
from utils.generator import VllmGenerator
from utils.arguments import (
    ModelConfig,
    InfrastructureConfig,
    InferenceConfig,
    GenerationConfig,
    PromptConfig,
    BenchmarkConfig
)


def main():
    parser = HfArgumentParser([
        ModelConfig,
        InfrastructureConfig,
        InferenceConfig,
        GenerationConfig,
        PromptConfig,
        BenchmarkConfig
    ])
    model_config, infra_config, inference_config, generation_config, prompt_config, benchmark_config = \
        parser.parse_args_into_dataclasses()

    # 1. Initialize Benchmark
    benchmark = Benchmark(
        model_path=model_config.model_path,
        task_types=benchmark_config.task_types,
        splits=benchmark_config.splits,
        data_dir=benchmark_config.data_dir,
        enable_thinking=prompt_config.enable_thinking,
        custom_chat_template=prompt_config.custom_chat_template,
        seed=benchmark_config.seed,
        benchmark_version=benchmark_config.version,
    )
    # Benchmark.print_benchmark_table()

    # 2. Initialize vLLM generator
    generator = VllmGenerator(
        model_name_or_path=model_config.model_path,
        checkpoint_path=model_config.checkpoint_path,
        trust_remote_code=model_config.trust_remote_code,
        dtype=model_config.dtype,
        max_model_len=model_config.max_model_len,
        max_logprobs=model_config.max_logprobs,
        tensor_parallel_size=infra_config.tensor_parallel_size,
        gpu_memory_utilization=infra_config.gpu_memory_utilization,
        force_enable_optimizations=inference_config.force_enable_optimizations,
        force_disable_optimizations=inference_config.force_disable_optimizations,
        worker_batch_size=inference_config.worker_batch_size,
        task_types=benchmark_config.task_types,
        seed=benchmark_config.seed
    )

    # 3. Generate text
    benchmark.run(
        generator=generator,
        output_dir=benchmark_config.output_dir,
        overwrite=benchmark_config.overwrite,
        # Generation parameters
        enable_thinking=prompt_config.enable_thinking,
        compute_cot_metrics=prompt_config.compute_cot_metrics,
        num_beams=generation_config.num_beams,
        num_return_sequences=generation_config.num_return_sequences,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        top_k=generation_config.top_k,
        presence_penalty=generation_config.presence_penalty,
        num_return_thinking_sequences=generation_config.num_return_thinking_sequences,
        race_mode=generation_config.race_mode,
        sample_size=benchmark_config.sample_size,
        sample_ratio=benchmark_config.sample_ratio,
    )

    # 4. Release GPU memory occupied by vLLM
    console.print("\nReleasing vLLM GPU memory...", style=warning_style)
    del generator
    import gc
    gc.collect()

    # Clear CUDA cache
    import torch
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
        task_types=benchmark_config.task_types,
        benchmark_version=benchmark_config.version,
        select_k=benchmark_config.select_k,
    )


if __name__ == "__main__":
    main()
