"""
Benchmark Generation Task Evaluation Script (Ray + HuggingFace Transformers Multi-Node Multi-GPU Version)

Uses Ray to manage multiple HuggingFace Transformers instances for distributed data-parallel inference across multiple machines.

Features:
- Multi-node multi-GPU support via Ray cluster
- Automatic GPU detection across all cluster nodes
- Tensor parallelism with same-node constraint (recommended)
- Data parallelism across workers

Prerequisites:
1. Start Ray cluster on all machines:
   
   On head node:
   $ ray start --head --port=6379 --dashboard-host=0.0.0.0
   
   On worker nodes:
   $ ray start --address=<head_node_ip>:6379
   
   Check cluster status:
   $ ray status

2. Ensure shared storage (e.g., NFS) is mounted on all nodes with same paths

Usage Examples:

    # 1. Auto-connect to Ray cluster (use all cluster GPUs)
    python evaluate.py \
        --model_path Qwen/Qwen2-7B \
        --ray_address auto \
        --data_dir ./data \
        --output_dir ./results
    
    # 2. Connect to specific Ray cluster
    python evaluate.py \
        --model_path Qwen/Qwen2-7B \
        --ray_address ray://127.0.0.1:10001 \
        --data_dir ./data \
        --output_dir ./results
    
    # 3. Local mode (single machine, backward compatible)
    python evaluate.py \
        --model_path Qwen/Qwen2-7B \
        --ray_address local \
        --num_gpus 8 \
        --data_dir ./data \
        --output_dir ./results
    
    # 4. Use tensor parallelism (e.g., 2 GPUs per worker)
    #    For 2 nodes with 8 GPUs each (16 total), creates 8 workers
    python evaluate.py \
        --model_path Qwen/Qwen2-7B \
        --tensor_parallel_size 2 \
        --ray_address auto \
        --data_dir ./data \
        --output_dir ./results
    
    # 5. Limit number of GPUs to use
    python evaluate.py \
        --model_path Qwen/Qwen2-7B \
        --num_gpus 12 \
        --ray_address auto \
        --data_dir ./data \
        --output_dir ./results
    
    # 6. Use PT checkpoint
    python evaluate.py \
        --model_path /path/to/model/config \
        --checkpoint_path /path/to/checkpoint.pt \
        --dtype bfloat16 \
        --ray_address auto \
        --data_dir ./data \
        --output_dir ./results
    
    # 7. Allow cross-node tensor parallelism (NOT recommended, for testing only)
    python evaluate.py \
        --model_path Qwen/Qwen2-7B \
        --tensor_parallel_size 4 \
        --allow_cross_node_tensor_parallel \
        --ray_address auto \
        --data_dir ./data \
        --output_dir ./results

Performance Tips:
- For best performance, keep tensor_parallel_size GPUs on the same node
- Use data parallelism (tensor_parallel_size=1) when model fits on single GPU
- Adjust worker_batch_size based on GPU memory
- Monitor cluster with: ray status, ray dashboard (port 8265)
"""
from transformers import HfArgumentParser
import torch

from benchmark import Benchmark
from benchmark.console import *
from utils.generator import RayHfTransformersGenerator
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

    # 2. Initialize Ray + HuggingFace Transformers generator (Multi-Node Support)
    generator = RayHfTransformersGenerator(
        model_name_or_path=model_config.model_path,
        checkpoint_path=model_config.checkpoint_path,
        trust_remote_code=model_config.trust_remote_code,
        dtype=model_config.dtype,
        tensor_parallel_size=infra_config.tensor_parallel_size,
        num_gpus=infra_config.num_gpus,
        gpu_ids=infra_config.gpu_ids,
        ray_address=infra_config.ray_address,
        allow_cross_node_tensor_parallel=infra_config.allow_cross_node_tensor_parallel,
        worker_batch_size=inference_config.worker_batch_size,
        task_types=benchmark_config.task_types
    )

    # 3. Generate text
    benchmark.run(
        generator=generator,
        output_dir=benchmark_config.output_dir,
        overwrite=benchmark_config.overwrite,
        sample_ratio=benchmark_config.sample_ratio,
        enable_thinking=prompt_config.enable_thinking,
        force_thinking=prompt_config.force_thinking
    )

    # 4. Release GPU memory occupied by model
    console.print("\nReleasing model GPU memory...", style=warning_style)
    generator.cleanup()
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
