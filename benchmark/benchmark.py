import os
import json
import copy
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

from benchmark.console import *
from benchmark.generation_runner import GenerationRunner
from benchmark.base_generator  import Generator
from benchmark.tasks import (
    BenchmarkTable,
    MergedTaskTable,
    LATEST_BENCHMARK_VERSION,
    check_benchmark_version,
    check_task_types,
    check_splits,
)
from benchmark.tasks.tasks import get_loader, get_evaluator, get_task_config


class DataLoaderWrapper:
    """Wrapper for unified data loading interface"""
    def __init__(self, model_path: str, benchmark_version: str, data_dir: str, enable_thinking: Optional[bool] = None, custom_chat_template: Optional[str] = None):
        self.model_path = model_path
        self._tokenizer = self._create_tokenizer(model_path) if model_path else None

        self.benchmark_version = benchmark_version
        self.data_dir = data_dir
        self.enable_thinking = enable_thinking
        self._loader_cache = {}
        self.custom_chat_template = custom_chat_template
    
    def _create_tokenizer(self, model_path: str):
        """Create tokenizer from model path"""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=False
            )
            console.print(f"[green]Tokenizer loaded from: {model_path}[/green]")
            return tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {model_path}: {e}")
    
    def load_data(
        self,
        task_name: str,
        split: str = "test",
        sample_size: Optional[Any] = None,
        sample_ratio: float = 1.0,
    ):
        """Load data using new loader system"""
        if task_name not in self._loader_cache:
            self._loader_cache[task_name] = get_loader(
                task_name=task_name,
                data_dir=self.data_dir,
                tokenizer=self._tokenizer,
                enable_thinking=self.enable_thinking,
                custom_chat_template=self.custom_chat_template,
                benchmark_version=self.benchmark_version,
            )

        loader = self._loader_cache[task_name]
        return loader.load_data(split=split, sample_size=sample_size, sample_ratio=sample_ratio)

    def load_data_with_thinking(
        self,
        task_name: str,
        split: str = "test",
        enable_thinking: bool = True,
        sample_size: Optional[Any] = None,
        sample_ratio: float = 1.0,
    ):
        """Load data with explicit enable_thinking override (for race mode)"""
        cache_key = f"{task_name}__thinking_{enable_thinking}"
        if cache_key not in self._loader_cache:
            self._loader_cache[cache_key] = get_loader(
                task_name=task_name,
                data_dir=self.data_dir,
                tokenizer=self._tokenizer,
                enable_thinking=enable_thinking,
                custom_chat_template=self.custom_chat_template,
                benchmark_version=self.benchmark_version,
            )

        loader = self._loader_cache[cache_key]
        return loader.load_data(split=split, sample_size=sample_size, sample_ratio=sample_ratio)



class Benchmark:
    """
    Benchmark Generation Task Evaluation Framework
    
    Usage Example:
        from benchmark import Benchmark
        from your_generator import YourGenerator
        
        benchmark = Benchmark(
            data_dir="./data"
        )
        
        generator = YourGenerator("your-model-path")
        
        benchmark.run(
            generator=generator,
            output_dir="./results"
        )
    """
    _AVERAGED_GENERATION_RUN_TASKS = {"challenge_itemic_pattern_grounding"}
    _AVERAGED_GENERATION_RUN_COUNT = 7
    
    @staticmethod
    def set_seed(seed: int = 42):
        """Set random seed for reproducibility"""
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def __init__(
        self,
        model_path: Optional[str] = None,
        task_types: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        custom_chat_template: Optional[str] = None,
        seed: Optional[int] = 42,
        benchmark_version: Optional[str] = None,
    ):
        """Initialize evaluation framework

        Args:
            benchmark_version: Benchmark version to use (e.g., 'v1.0', 'v2.0').
                If None, uses merged task table (v2.0 takes priority on conflicts).
        """
        if seed is not None:
            self.set_seed(seed)
            console.print(f"[green]Random seed set to: {seed}[/green]")
        if benchmark_version is not None:
            self.benchmark_version = check_benchmark_version(benchmark_version)
        else:
            self.benchmark_version = None
        self.data_dir = data_dir
        self.task_types = check_task_types(task_types, self.benchmark_version)
        splits_version = self.benchmark_version or LATEST_BENCHMARK_VERSION
        self.splits = check_splits(splits, splits_version)
        self.data_loader = DataLoaderWrapper(
            model_path=model_path,
            benchmark_version=self.benchmark_version,
            data_dir=data_dir,
            enable_thinking=enable_thinking,
            custom_chat_template=custom_chat_template
        )
    
    @staticmethod
    def print_benchmark_table():
        """Print all available benchmark versions and tasks"""
        for benchmark_version in BenchmarkTable:
            console.print(
                head_print(f"Benchmark Dataset Version: {benchmark_version}"),
                style=head_style,
                justify="center",
            )

            task_types_list = list(BenchmarkTable[benchmark_version].keys())
            total_task_types = len(task_types_list)
            
            for task_idx, task_type in enumerate(task_types_list, start=1):
                console.print(
                    f"\nTask Type [{task_idx}/{total_task_types}]: {task_type}\n", 
                    style=subhead_style, 
                    justify="center"
                )
                 
                task_config = BenchmarkTable[benchmark_version][task_type]
                
                console.print(
                    f"Dataset Name: {task_config.get('name', task_type)}",
                    style=row_style,
                    justify="center",
                )
                console.print(
                    f"Source: {task_config.get('source', 'N/A')}",
                    style=row_style,
                    justify="center",
                )
                console.print(
                    f"Splits: {task_config.get('splits', [])}",
                    style=row_style,
                    justify="center",
                )
                console.print(
                    f"Sample Size: {task_config.get('sample_size', 'N/A')}",
                    style=row_style,
                    justify="center",
                )
                console.print(
                    f"Description: {task_config.get('description', 'N/A')}",
                    style=row_style,
                    justify="center",
                )
    
    @staticmethod
    def check_generator(generator):
        """Verify that generator implements required methods"""
        required_methods = ["__str__", "generate"]
        for method in required_methods:
            if not hasattr(generator, method):
                raise ValueError(f"Generator should have `{method}` method.")
            if method != "__str__" and not callable(getattr(generator, method, None)):
                raise ValueError(f"Generator.{method} should be callable.")

    @staticmethod
    def _should_average_generation_runs(
        task_name: str,
        gen_data: Dict[str, Any],
        samples: Dict[str, Dict[str, Any]],
    ) -> bool:
        """Return True when this task should average metrics over saved generation runs."""
        if task_name not in Benchmark._AVERAGED_GENERATION_RUN_TASKS:
            return False
        if not gen_data.get("race_mode"):
            return False
        if gen_data.get("evaluation_run_policy") != "average_last_n":
            return False
        return any(isinstance(sample.get("generation_runs"), list) for sample in samples.values())

    @staticmethod
    def _build_samples_for_generation_run(
        samples: Dict[str, Dict[str, Any]],
        run_index: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Replace each sample's active generations with one saved generation run."""
        run_samples = {}
        for sample_id, sample in samples.items():
            run_sample = dict(sample)
            sample_runs = sample.get("generation_runs") or []
            if run_index >= len(sample_runs):
                run_sample["generations"] = []
                run_sample.pop("logprobs", None)
            else:
                run_data = sample_runs[run_index]
                run_sample["generations"] = run_data.get("generations", [])
                if run_data.get("logprobs"):
                    run_sample["logprobs"] = run_data["logprobs"]
                else:
                    run_sample.pop("logprobs", None)
            run_samples[sample_id] = run_sample
        return run_samples

    @staticmethod
    def _average_metric_values(values: List[Any]) -> Any:
        """Average numeric metric values after dropping one min and max."""
        if not values:
            return None
        if all(isinstance(value, bool) for value in values):
            numeric_values = [int(value) for value in values]
            if len(numeric_values) > 2:
                numeric_values = sorted(numeric_values)[1:-1]
            return sum(numeric_values) / len(numeric_values)
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            numeric_values = [float(value) for value in values]
            if len(numeric_values) > 2:
                numeric_values = sorted(numeric_values)[1:-1]
            return sum(numeric_values) / len(numeric_values)
        if all(isinstance(value, list) for value in values):
            lengths = {len(value) for value in values}
            if len(lengths) == 1 and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for value in values
                for item in value
            ):
                return [
                    Benchmark._average_metric_values([value[idx] for value in values])
                    for idx in range(len(values[0]))
                ]
        return values[-1]

    @staticmethod
    def _average_metric_dicts(metric_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average all metric dictionaries key-wise."""
        averaged = {}
        keys = set()
        for metrics in metric_dicts:
            keys.update(metrics.keys())

        for key in keys:
            values = [metrics[key] for metrics in metric_dicts if key in metrics]
            if key in {"total_samples", "generations", "pid_generations"}:
                averaged[key] = values[-1]
            else:
                averaged[key] = Benchmark._average_metric_values(values)
        return averaged

    @staticmethod
    def _average_per_sample_metric_dicts(
        per_run_metrics: List[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Average per-sample metrics across generation runs where values are numeric."""
        averaged = {}
        sample_ids = set()
        for run_metrics in per_run_metrics:
            sample_ids.update(run_metrics.keys())

        for sample_id in sample_ids:
            sample_metric_dicts = [
                run_metrics[sample_id]
                for run_metrics in per_run_metrics
                if sample_id in run_metrics
            ]
            averaged[sample_id] = Benchmark._average_metric_dicts(sample_metric_dicts)
        return averaged

    @staticmethod
    def _evaluate_averaged_generation_runs(
        evaluator_class,
        task_name: str,
        task_dir: str,
        samples: Dict[str, Dict[str, Any]],
        task_config: Dict[str, Any],
        data_dir: str,
        overwrite: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Evaluate the last N saved generation runs and average their metrics."""
        run_counts = [
            len(sample.get("generation_runs") or [])
            for sample in samples.values()
            if isinstance(sample.get("generation_runs"), list)
        ]
        available_runs = min(run_counts) if run_counts else 0
        if available_runs <= 0:
            raise ValueError(f"No generation_runs found for averaged evaluation task: {task_name}")

        run_count = min(Benchmark._AVERAGED_GENERATION_RUN_COUNT, available_runs)
        start_index = available_runs - run_count
        console.print(
            f"[cyan]Averaging {task_name} metrics over the last {run_count} generation runs "
            f"after dropping one min and one max "
            f"(available={available_runs})[/cyan]"
        )

        metrics_by_run = []
        per_sample_metrics_by_run = []
        for run_index in range(start_index, available_runs):
            console.print(f"[cyan]Evaluating saved generation run {run_index + 1}/{available_runs}...[/cyan]")
            run_samples = Benchmark._build_samples_for_generation_run(samples, run_index)
            evaluator = evaluator_class(
                samples=run_samples,
                task_name=task_name,
                predictions_dir=task_dir,
                debug=False,
                task_config=copy.deepcopy(task_config),
                data_dir=data_dir,
                overwrite=True,
                cached_metrics={},
            )
            run_metrics, run_per_sample_metrics = evaluator.evaluate()
            metrics_by_run.append(run_metrics)
            per_sample_metrics_by_run.append(run_per_sample_metrics)

        averaged_metrics = Benchmark._average_metric_dicts(metrics_by_run)
        averaged_per_sample_metrics = Benchmark._average_per_sample_metric_dicts(per_sample_metrics_by_run)
        averaged_metrics["averaged_generation_runs"] = run_count
        averaged_metrics["averaged_generation_run_policy"] = "drop_min_max_last_n"
        return averaged_metrics, averaged_per_sample_metrics
    
    def run(
        self,
        generator: Generator,
        output_dir: str = "./results",
        overwrite: bool = False,
        **kwargs
    ):
        """Run benchmark evaluation"""
        self.check_generator(generator)
        console.print(f"\n\nStarting generation\n\n", style=head_style, justify="center")
        
        generation_runner = GenerationRunner(self.data_loader, overwrite=overwrite)
        total_tasks = 0
        completed_tasks = 0
        if self.benchmark_version is not None:
            task_table = BenchmarkTable[self.benchmark_version]
        else:
            task_table = MergedTaskTable
        for task_name in self.task_types:
            if task_name not in task_table:
                continue
            task_config = task_table[task_name]
            available_splits = task_config.get("splits", ["test"])
            for split in self.splits:
                if split in available_splits:
                    total_tasks += 1
        
        for task_name in self.task_types:
            if task_name not in task_table:
                console.print(f"Task does not exist: {task_name}")
                continue
            
            task_config = task_table[task_name]
            available_splits = task_config.get("splits", ["test"])
            
            # Iterate through all splits
            for split in self.splits:
                if split not in available_splits:
                    console.print(f"Split does not exist: {split} (task: {task_name})")
                    continue
                
                # Determine displayed sample size
                sample_size_param = kwargs.get('sample_size')
                if sample_size_param is not None:
                    if sample_size_param == "full":
                        display_sample_size = task_config.get('size', 'full')
                    else:
                        display_sample_size = int(sample_size_param)
                else:
                    display_sample_size = task_config.get('sample_size', 'N/A')
                
                console.print(
                    f"\nTask [{completed_tasks + 1}/{total_tasks}]: {task_name} | Split: {split} | Sample Size: {display_sample_size}\n",
                    style=subhead_style,
                    justify="center",
                )
                
                try:
                    task_gen_config = task_config.get("generation_config", {})
                    prompt_config = task_config.get("prompt_config", {})
                    
                    # Merge generation parameters (priority: user input > task config > Generator init parameters)
                    # Filter out None values from kwargs to avoid overwriting task config
                    valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                    merged_kwargs = {**task_gen_config, **prompt_config, **valid_kwargs}

                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

                    # Execute generation (without computing metrics)
                    generation_runner(
                        task_name=task_name,
                        split=split,
                        results_save_dir=output_dir,
                        generator=generator,
                        **merged_kwargs
                    )
                    
                    completed_tasks += 1
                
                    
                except Exception as e:
                    import traceback
                    console.print(f"✗ Task failed: {task_name}/{split}", style=err_style)
                    console.print(f"✗ Error type: {type(e).__name__}", style=err_style)
                    console.print(f"✗ Error message: {str(e)}", style=err_style)
                    console.print("✗ Full stack trace:", style=err_style)
                    console.print(traceback.format_exc(), style=dim_style)
        
        console.print(f"Total tasks: {total_tasks}")
        console.print(f"Completed tasks: {completed_tasks}")
        console.print(f"Failed tasks: {total_tasks - completed_tasks}")
        console.print(f"Results saved to: {output_dir}")
    
    @staticmethod
    def _evaluate_single_task(
        task_name: str,
        task_dir: str,
        generation_file: str,
        split: str,
        data_dir: str,
        overwrite: bool,
        valid_kwargs: Dict[str, Any],
        cached_metrics: Optional[Dict[str, Any]] = None,
        benchmark_version: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Evaluate a single task split"""
        
        # Read generation results
        with open(generation_file, 'r', encoding='utf-8') as f:
            gen_data = json.load(f)
        
        if "samples" not in gen_data:
            raise ValueError("Generation result file missing 'samples' field (new format).")
        
        samples = gen_data["samples"]
        
        # Load task configuration
        if benchmark_version is not None:
            task_table = BenchmarkTable[benchmark_version]
        else:
            task_table = MergedTaskTable
        if task_name not in task_table:
            console.print(f"⚠ Warning: Task '{task_name}' not found in {benchmark_version or 'any registered version'}, skipping...", style=warning_style)
            return None, None, None

        try:
            evaluator_class = get_evaluator(task_name=task_name, benchmark_version=benchmark_version)
            task_config = copy.deepcopy(get_task_config(task_name=task_name, benchmark_version=benchmark_version))
            task_config['evaluation_config'].update(valid_kwargs)
            console.print(f"Using {evaluator_class.__name__} for {task_name}")
            if Benchmark._should_average_generation_runs(task_name, gen_data, samples):
                metrics, per_sample_metrics = Benchmark._evaluate_averaged_generation_runs(
                    evaluator_class=evaluator_class,
                    task_name=task_name,
                    task_dir=task_dir,
                    samples=samples,
                    task_config=task_config,
                    data_dir=data_dir,
                    overwrite=overwrite,
                )
            else:
                evaluator = evaluator_class(
                    samples=samples,
                    task_name=task_name,
                    predictions_dir=task_dir,
                    debug=False,  # Enable debug mode for detailed info
                    task_config=task_config,
                    data_dir=data_dir,
                    overwrite=overwrite,
                    cached_metrics=cached_metrics
                )
                metrics, per_sample_metrics = evaluator.evaluate()

            # Compute MFU metrics if hardware info and token stats are available
            try:
                from benchmark.tasks.mfu_evaluator import compute_mfu_from_generation_data
                mfu_metrics = compute_mfu_from_generation_data(gen_data)
                if mfu_metrics:
                    metrics.update(mfu_metrics)
                    # Display MFU for each stage
                    if "mfu" in mfu_metrics:
                        mfu_list = mfu_metrics["mfu"]
                        if len(mfu_list) == 1:
                            console.print(f"✓ MFU: {mfu_list[0]:.2%}", style=success_style)
                        else:
                            mfu_values = [f"Stage{i+1}: {mfu:.2%}" for i, mfu in enumerate(mfu_list)]
                            console.print(f"✓ MFU (multi-stage): {', '.join(mfu_values)}", style=success_style)
            except Exception as e:
                console.print(f"⚠ Warning: MFU calculation failed: {e}", style=warning_style)

            # Update samples with per-sample metrics
            for sample_id, sample_metrics in per_sample_metrics.items():
                if sample_id in samples:
                    samples[sample_id].update(sample_metrics)
            
            # Write updated data back to generation result file
            gen_data["samples"] = samples
            with open(generation_file, 'w', encoding='utf-8') as f:
                json.dump(gen_data, f, indent=2, ensure_ascii=False)
            console.print(f"Updated sample metrics to: {generation_file}")
            
            return gen_data, metrics, samples
        
        except Exception as e:
            console.print(f"✗ Error evaluating {task_name}: {e}", style=err_style)
            console.print(f"Skipping task {task_name}", style=warning_style)
            return None, None, None
    
    @staticmethod
    def _create_debug_file(generation_file: str, gen_data: Dict[str, Any], samples: Dict[str, Any], overwrite: bool = False) -> None:
        """Create debug file with first 100 samples"""

        debug_file = f"{generation_file}.debug"
        if overwrite or not os.path.exists(debug_file):
            sorted_ids = sorted(samples.keys())
            debug_sample_ids = sorted_ids[:100]
            debug_samples = {id: samples[id] for id in debug_sample_ids}
            
            debug_data = {
                "model_name": gen_data.get("model_name", ""),
                "task_name": gen_data.get("task_name", ""),
                "split": gen_data.get("split", ""),
                "total_time": gen_data.get("total_time", 0),
                "avg_time_per_sample": gen_data.get("avg_time_per_sample", 0),
                "samples": debug_samples,
            }
            
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
            console.print(f"Created debug file: {debug_file}")
    
    @staticmethod
    def _calculate_model_total_time(model_results: Dict[str, Any]) -> float:
        """Calculate total time for all tasks of a model"""
        model_total_time = 0
        for task_name, task_results in model_results.items():
            if task_name.startswith("_"):
                continue
            for split, split_metrics in task_results.items():
                model_total_time += split_metrics.get("total_time", 0)
        return model_total_time
    
    @staticmethod
    def _save_results_as_json(eval_results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results as JSON"""
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        console.print(f"\n\n✓ Results Saved to {output_path}\n\n", style=success_style, justify="center")
    
    @staticmethod
    def _load_existing_results(output_path: str, task_types: List[str] = None) -> dict:
        """Load existing evaluation results from JSON file for incremental update"""
        eval_results = {}

        if os.path.exists(output_path) and output_path.endswith('.json'):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    eval_results = json.load(f)
                console.print(f"✓ Loaded existing results from {output_path}", style=success_style, justify="center")
                if task_types is not None:
                    console.print(f"  Will update only specified tasks: {', '.join(task_types)}", style=success_style, justify="center")
            except Exception as e:
                console.print(f"⚠ Warning: Failed to load existing results: {e}", style=err_style, justify="center")
                console.print(f"  Starting with empty results", style=err_style, justify="center")
                eval_results = {}

        return eval_results

    @staticmethod
    def evaluate_dev(
        generation_results_dir: str,
        output_path: str = "./eval_results.json",
        data_dir: str = None,
        overwrite: bool = False,
        task_types: List[str] = None,
        benchmark_version: Optional[str] = None,
        **kwargs
    ):
        """Batch evaluate generated results and generate report"""
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        console.print(f"\n\nMetric Calculation\n", style=head_style, justify="center")
        console.print(f"Result Directory: {generation_results_dir}\n\n", style=head_style, justify="center")

        if not os.path.exists(generation_results_dir):
            console.print(f"✗ Error: Result Directory Not Found: {generation_results_dir}", style=err_style, justify="center")
            return

        eval_results = Benchmark._load_existing_results(output_path, task_types)
        
        for model_name in os.listdir(generation_results_dir):
            model_dir = os.path.join(generation_results_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            if model_name not in eval_results:
                eval_results[model_name] = {}
            
            all_tasks = [t for t in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, t))]

            if task_types is not None:
                all_tasks = [t for t in all_tasks if t in task_types]

            total_tasks_count = len(all_tasks)
            for task_idx, task_name in enumerate(all_tasks, start=1):
                task_dir = os.path.join(model_dir, task_name)

                console.print(f"\nTask [{task_idx}/{total_tasks_count}]: {task_name}\n", style=subhead_style, justify="center")

                if task_name not in eval_results[model_name]:
                    eval_results[model_name][task_name] = {}
                for filename in os.listdir(task_dir):
                    if not filename.endswith('_generated.json'):
                        continue
                    
                    split = filename.replace('_generated.json', '')
                    generation_file = os.path.join(task_dir, filename)

                    cached_metrics = eval_results.get(model_name, {}).get(task_name, {}).get(split, {})

                    # Evaluate single task
                    gen_data, metrics, samples = Benchmark._evaluate_single_task(
                        task_name=task_name,
                        task_dir=task_dir,
                        generation_file=generation_file,
                        split=split,
                        data_dir=data_dir,
                        overwrite=overwrite,
                        valid_kwargs=valid_kwargs,
                        cached_metrics=cached_metrics,
                        benchmark_version=benchmark_version,
                    )
                    
                    if gen_data is None:
                        continue

                    Benchmark._create_debug_file(generation_file, gen_data, samples, overwrite)
                    eval_results[model_name][task_name][split] = {
                        **metrics,
                        "total_time": gen_data.get("total_time", 0),
                        "avg_time_per_sample": gen_data.get("avg_time_per_sample", 0),
                    }
            
            model_total_time = Benchmark._calculate_model_total_time(eval_results[model_name])
            eval_results[model_name]["_total_time"] = model_total_time
            console.print(f"\n✓ Total time: {model_total_time:.2f}s ({model_total_time/60:.2f}min)\n", style=success_style)
        
        Benchmark._save_results_as_json(eval_results, output_path)
