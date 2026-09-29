"""End2End visual model evaluation entry point.

Usage:
    python scripts/end2end/evaluate.py \
        --model_path /path/to/stepXXXX \
        --task_types sid2caption_video \
        --data_dir ./data \
        --output_dir ./results

model_path should contain converted_encoder/, converted_decoder/, converted_projection/.
"""

from transformers import HfArgumentParser
import torch

from benchmark import Benchmark
from benchmark.console import *
from utils.generator import End2EndGenerator
from utils.arguments import End2EndModelConfig, InferenceConfig, PromptConfig, BenchmarkConfig


def main():
    parser = HfArgumentParser([End2EndModelConfig, InferenceConfig, PromptConfig, BenchmarkConfig])
    model_config, inference_config, prompt_config, benchmark_config = parser.parse_args_into_dataclasses()

    # 1. Initialize Benchmark (use decoder_path for tokenizer / apply_chat_template)
    benchmark = Benchmark(
        model_path=model_config.decoder_path,
        task_types=benchmark_config.task_types,
        splits=benchmark_config.splits,
        data_dir=benchmark_config.data_dir,
        enable_thinking=prompt_config.enable_thinking,
        custom_chat_template=prompt_config.custom_chat_template,
        seed=benchmark_config.seed,
        benchmark_version=benchmark_config.version,
    )

    # 2. Initialize End2End generator
    generator = End2EndGenerator(
        encoder_path=model_config.encoder_path,
        decoder_path=model_config.decoder_path,
        projection_path=model_config.projection_path,
        model_name=model_config.model_name,
        num_query_tokens=model_config.num_query_tokens,
        max_new_tokens=inference_config.max_new_tokens,
        temperature=inference_config.temperature,
        top_p=inference_config.top_p,
        do_sample=inference_config.do_sample,
        nframe=model_config.nframe,
        system_prompt=model_config.system_prompt,
        use_flash_attn=model_config.use_flash_attn,
    )

    # 3. Run generation
    benchmark.run(
        generator=generator,
        output_dir=benchmark_config.output_dir,
        overwrite=benchmark_config.overwrite,
        sample_ratio=benchmark_config.sample_ratio,
    )

    # 4. Release GPU memory
    console.print("\nReleasing GPU memory...", style=warning_style)
    del generator
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    console.print("GPU memory release completed\n", style=success_style)

    # 5. Evaluate
    eval_results_path = f"{benchmark_config.output_dir}/eval_results.json"
    Benchmark.evaluate_dev(
        generation_results_dir=benchmark_config.output_dir,
        output_path=eval_results_path,
        data_dir=benchmark_config.data_dir,
        overwrite=benchmark_config.overwrite,
        task_types=benchmark_config.task_types,
        benchmark_version=benchmark_config.version,
    )


if __name__ == "__main__":
    main()
