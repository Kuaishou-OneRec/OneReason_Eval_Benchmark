"""
Generation Runner

Responsible for:
1. Loading test data via data loader
2. Calling Generator to produce model outputs  
3. Saving generation results to JSON files

Note: Does NOT compute evaluation metrics (handled by task-specific evaluators)
"""

import json
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

from benchmark.console import *
from benchmark.base_generator import Generator



class GenerationRunner:
    """
    Generation task runner

    Orchestrates the generation phase of evaluation:
    - Loads test data via data loader
    - Calls generator to produce model outputs
    - Saves generation results to disk

    Evaluation metrics are computed separately by task-specific evaluators.
    """

    _RACE_NOTHINK_ONLY_TASKS = {
        "challenge_common_sense",
        "challenge_evolution_action_select",
        "challenge_evolution_topic_gen",
        "challenge_itemic_pattern_caption_video",
        "challenge_itemic_pattern_caption_product",
        "challenge_itemic_pattern_caption_ad",
    }
    _RACE_AVERAGED_TASKS = set()
    _RACE_AVERAGED_RUNS = 7

    def __init__(
        self,
        data_loader,
        overwrite: bool = False,
        mfu_debug: bool = False,
    ):
        """
        Args:
            data_loader: Data loader wrapper (must have load_data() and benchmark_version attributes)
            overwrite: Whether to overwrite existing results
        """
        self.data_loader = data_loader
        self.overwrite = overwrite
        self.benchmark_version = data_loader.benchmark_version
        self.mfu_debug = mfu_debug
    
    def __call__(
        self,
        task_name: str,
        split: str,
        results_save_dir: str,
        generator: Generator,
        **kwargs
    ) -> None:
        """
        Execute generation pipeline
        
        This method is responsible for generation and saving only,
        NOT for computing evaluation metrics.
        
        Args:
            task_name: Task name
            split: Dataset split
            results_save_dir: Results save directory
            generator: Generator instance
            **kwargs: Generation parameters
        
        Returns:
            None
        """
        model_name = str(generator)
        results_dir = os.path.join(
            results_save_dir,
            model_name,
            task_name
        )
        os.makedirs(results_dir, exist_ok=True)
        
        generation_file = os.path.join(results_dir, f"{split}_generated.json")
        
        # Check if generation results already exist
        if os.path.exists(generation_file) and not self.overwrite:
            console.print(f"Generation results already exist, skipping: {generation_file}")
            console.print("To regenerate, please set overwrite=True")
            
            return None
        
        start_time = time.time()

        # Extract sample_size parameter (don't pass to generator)
        sample_size_param = kwargs.pop('sample_size', None)
        sample_ratio_param = kwargs.pop('sample_ratio', 1.0)
        # Extract race_mode (don't pass to generator)
        race_mode = kwargs.pop('race_mode', False)

        # 1. Load data
        test_data = self.data_loader.load_data(
            task_name=task_name,
            split=split,
            sample_size=sample_size_param,
            sample_ratio=sample_ratio_param,
        )

        # 2. Extract prompts and references
        prompts = {id: data["prompt"] for id, data in test_data.items()}
        references = {id: data["ground_truth"] for id, data in test_data.items()}
        metadata = {id: data.get("metadata", {}) for id, data in test_data.items()}

        def generate_once():
            race_debug = {}
            if not race_mode:
                return (*generator.generate(prompts, metadata=metadata, references=references, **kwargs), race_debug)

            if task_name in self._RACE_NOTHINK_ONLY_TASKS:
                kwargs_nothink = {**kwargs, "enable_thinking": False}
                kwargs_nothink.pop("num_return_thinking_sequences", None)
                nothink_data = self.data_loader.load_data_with_thinking(
                    task_name=task_name,
                    split=split,
                    enable_thinking=False,
                    sample_size=sample_size_param,
                    sample_ratio=sample_ratio_param,
                )
                nothink_prompts_for_debug = {id: data["prompt"] for id, data in nothink_data.items()}
                generated, generated_logprobs = generator.generate(
                    nothink_prompts_for_debug,
                    metadata=metadata,
                    references=references,
                    **kwargs_nothink,
                )
                race_debug["nothink"] = {
                    "prompts": nothink_prompts_for_debug,
                    "generations": generated,
                }
                return generated, generated_logprobs, race_debug
            else:
                # Race mode: split budget equally between think and no-think
                total_n = kwargs.get("num_return_sequences", 32)
                half_n = total_n // 2

                # Think pass
                kwargs_think = {**kwargs, "enable_thinking": True, "num_return_sequences": half_n, "num_beams": half_n, "num_return_thinking_sequences": 1}
                # Load prompts formatted for think mode
                think_data = self.data_loader.load_data_with_thinking(
                    task_name=task_name,
                    split=split,
                    enable_thinking=True,
                    sample_size=sample_size_param,
                    sample_ratio=sample_ratio_param,
                )
                think_prompts = {id: data["prompt"] for id, data in think_data.items()}
                gen_think, logprobs_think = generator.generate(think_prompts, metadata=metadata, references=references, **kwargs_think)

                # No-think pass
                kwargs_nothink = {**kwargs, "enable_thinking": False, "num_return_sequences": half_n, "num_beams": half_n}
                kwargs_nothink.pop("num_return_thinking_sequences", None)
                nothink_data = self.data_loader.load_data_with_thinking(
                    task_name=task_name,
                    split=split,
                    enable_thinking=False,
                    sample_size=sample_size_param,
                    sample_ratio=sample_ratio_param,
                )
                nothink_prompts = {id: data["prompt"] for id, data in nothink_data.items()}
                gen_nothink, logprobs_nothink = generator.generate(nothink_prompts, metadata=metadata, references=references, **kwargs_nothink)

                # Merge results
                generated = {sid: gen_think.get(sid, []) + gen_nothink.get(sid, []) for sid in prompts}
                generated_logprobs = {sid: logprobs_think.get(sid, []) + logprobs_nothink.get(sid, []) for sid in prompts}
                race_debug["think"] = {
                    "prompts": think_prompts,
                    "generations": gen_think,
                }
                race_debug["nothink"] = {
                    "prompts": nothink_prompts,
                    "generations": gen_nothink,
                }
                return generated, generated_logprobs, race_debug

        # 3. Generate text
        generation_runs = None
        evaluation_run_times = None
        race_debug = {}
        if race_mode and task_name in self._RACE_AVERAGED_TASKS:
            generation_runs = []
            evaluation_run_times = []
            for run_idx in range(self._RACE_AVERAGED_RUNS):
                console.print(
                    f"[cyan]Race averaged evaluation run {run_idx + 1}/{self._RACE_AVERAGED_RUNS} for {task_name}[/cyan]"
                )
                run_start_time = time.time()
                run_generations, run_logprobs, race_debug = generate_once()
                evaluation_run_times.append(time.time() - run_start_time)
                generation_runs.append({
                    "generations": run_generations,
                    "logprobs": run_logprobs,
                })
            generations = generation_runs[-1]["generations"]
            logprobs = generation_runs[-1]["logprobs"]
        else:
            generations, logprobs, race_debug = generate_once()
        
        end_time = time.time()

        if race_mode:
            for sid in list(prompts.keys())[:5]:
                console.print(f"\n{'='*60}\n[bold]Sample ID:[/bold] {sid}")
                if race_debug:
                    if "think" in race_debug:
                        think_prompt = race_debug["think"]["prompts"].get(sid, prompts[sid])
                        console.print(f"[bold]Think Input:[/bold]\n{think_prompt}")
                        for i, gen in enumerate(race_debug["think"]["generations"].get(sid, [])):
                            console.print(f"[bold]Think Output[{i}]:[/bold]\n{gen}")

                    if "nothink" in race_debug:
                        nothink_prompt = race_debug["nothink"]["prompts"].get(sid, prompts[sid])
                        console.print(f"[bold]No-think Input:[/bold]\n{nothink_prompt}")
                        for i, gen in enumerate(race_debug["nothink"]["generations"].get(sid, [])):
                            console.print(f"[bold]No-think Output[{i}]:[/bold]\n{gen}")
                else:
                    console.print(f"[bold]Input:[/bold]\n{prompts[sid]}")
                    for i, gen in enumerate(generations.get(sid, [])):
                        console.print(f"[bold]Output[{i}]:[/bold]\n{gen}")

        total_time = end_time - start_time
        num_samples = len(test_data)
        avg_time_per_sample = total_time / num_samples if num_samples > 0 else 0
        console.print(f"Total time: {total_time:.2f}s, Average per sample: {avg_time_per_sample:.4f}s")

        # 4. Collect hardware info and MFU statistics (for MFU calculation)
        hardware_info = getattr(generator, 'get_hardware_info', lambda: None)()
        mfu_stats = getattr(generator, 'mfu_stats', None)
        num_params_value = getattr(generator, 'num_params', None)

        if self.mfu_debug:
            console.print(f"[MFU DEBUG] hardware_info: {hardware_info}")
            console.print(f"[MFU DEBUG] num_params: {num_params_value}")
            if mfu_stats:
                first_key = list(mfu_stats.keys())[0]
                first_stats = mfu_stats[first_key]
                console.print(f"[MFU DEBUG] mfu_stats count: {len(mfu_stats)}, first sample: {first_key}")
                console.print(f"[MFU DEBUG]   input_tokens: {first_stats.get('input_tokens', 'MISSING')}")
                console.print(f"[MFU DEBUG]   output_tokens: {first_stats.get('output_tokens', 'MISSING')}")
                console.print(f"[MFU DEBUG]   times: {first_stats.get('times', 'MISSING')}")

        cot_metrics_by_sample = getattr(generator, 'cot_metrics_by_sample', None)

        # 5. Save generation results
        self.save_generations(
            model_name=model_name,
            task_name=task_name,
            split=split,
            generations=generations,
            references=references,
            logprobs=logprobs,
            test_data=test_data,
            output_path=generation_file,
            total_time=total_time,
            avg_time_per_sample=avg_time_per_sample,
            hardware_info=hardware_info,
            mfu_stats=mfu_stats,
            num_params=getattr(generator, 'num_params', None),
            cot_metrics_by_sample=cot_metrics_by_sample,
            generation_runs=generation_runs,
            race_mode=race_mode,
            evaluation_run_times=evaluation_run_times,
        )
        
        console.print(f"Generation results saved to: {generation_file}")
        
        return None
    
    @staticmethod
    def save_generations(
        model_name: str,
        task_name: str,
        split: str,
        generations: Dict[str, List[str]],
        references: Dict[str, str],
        logprobs: Dict[str, List[float]],
        test_data: Dict[str, Dict[str, Any]],
        output_path: str,
        total_time: float,
        avg_time_per_sample: float,
        hardware_info: Optional[Dict[str, Any]] = None,
        mfu_stats: Optional[Dict[str, Dict[str, List[int]]]] = None,
        num_params: Optional[float] = None,
        cot_metrics_by_sample: Optional[Dict[str, Dict[str, Any]]] = None,
        generation_runs: Optional[List[Dict[str, Dict[str, List[Any]]]]] = None,
        race_mode: bool = False,
        evaluation_run_times: Optional[List[float]] = None,
    ):
        """
        Save generation results (excluding evaluation metrics)
        
        Result format:
        {
            "model_name": "...",
            "task_name": "...",
            "split": "...",
            "total_time": "...",
            "avg_time_per_sample": "...",
            "samples": {
                "<sample_id>": {
                    "prompt": "...",
                    "generations": ["...", "..."],
                    "ground_truth": "...",
                    "metadata": {...}  # Contains metadata from original data
                },
                ...
            }
        }
        """
        # Check if this is a classification task (label_pred)
        is_classification_task = task_name == "label_pred"
        
        samples: Dict[str, Any] = {}
        for id, gens in generations.items():
            sample_data = {
                "prompt": test_data.get(id, {}).get("prompt", ""),
                "generations": gens,
                "ground_truth": references.get(id, ""),
            }

            if id in logprobs and logprobs[id]:
                sample_data["logprobs"] = logprobs[id]

            # Add MFU statistics for this sample (for MFU calculation)
            if mfu_stats and id in mfu_stats:
                sample_data["input_tokens"] = mfu_stats[id].get("input_tokens", [])
                sample_data["output_tokens"] = mfu_stats[id].get("output_tokens", [])
                sample_data["times"] = mfu_stats[id].get("times", [])

            if is_classification_task and id in test_data:
                metadata = test_data[id].get("metadata", {})
                if "uid" in metadata:
                    sample_data["user_id"] = metadata["uid"]

            if id in test_data and "metadata" in test_data[id]:
                sample_data["metadata"] = test_data[id]["metadata"]

            if cot_metrics_by_sample and id in cot_metrics_by_sample:
                # Per-sample CoT quality metrics (delta_ll / ehr / cov_rate /
                # sid_valid_rate) — stored alongside generations so the
                # evaluator can aggregate without re-running inference.
                sample_data["cot_metrics"] = cot_metrics_by_sample[id]

            if generation_runs:
                sample_runs = []
                for run in generation_runs:
                    run_generations = run.get("generations", {}).get(id, [])
                    run_logprobs = run.get("logprobs", {}).get(id, [])
                    run_data = {"generations": run_generations}
                    if run_logprobs:
                        run_data["logprobs"] = run_logprobs
                    sample_runs.append(run_data)
                sample_data["generation_runs"] = sample_runs

            samples[id] = sample_data

        data = {
            "model_name": model_name,
            "task_name": task_name,
            "split": split,
            "total_time": total_time,
            "avg_time_per_sample": avg_time_per_sample,
            "samples": samples,
        }

        if race_mode:
            data["race_mode"] = True
        if generation_runs:
            data["evaluation_run_policy"] = "average_last_n"
            data["evaluation_run_count"] = len(generation_runs)
            data["evaluation_run_times"] = evaluation_run_times or []

        # Add hardware info and token statistics (for MFU calculation)
        if hardware_info:
            data["hardware_info"] = hardware_info
        else:
            console.print("[MFU DEBUG] ❌ Skipping hardware_info (None or empty)")

        if num_params:
            data["num_params"] = num_params
        else:
            console.print("[MFU DEBUG] ❌ Skipping num_params (None or 0)")

        # Save mfu_stats_aggregate for multi-stage MFU calculation
        # Compute aggregate statistics from per-sample mfu_stats
        mfu_debug = False
        if mfu_debug and mfu_stats:
            # Determine number of stages from first sample
            num_stages = 0
            for sample_stats in mfu_stats.values():
                num_stages = len(sample_stats.get("input_tokens", []))
                console.print(f"[MFU DEBUG] Determined num_stages: {num_stages}")
                break

            # New structure: dict with lists instead of array of dicts
            data["mfu_stats_aggregate"] = {
                "total_input_tokens": [],
                "total_output_tokens": [],
                "total_time": []
            }

            for stage_idx in range(num_stages):
                total_input_tokens = 0
                total_output_tokens = 0

                # Aggregate token stats across all samples for this stage
                for sample_stats in mfu_stats.values():
                    input_tokens_list = sample_stats.get("input_tokens", [])
                    output_tokens_list = sample_stats.get("output_tokens", [])

                    if stage_idx < len(input_tokens_list):
                        total_input_tokens += input_tokens_list[stage_idx]
                    if stage_idx < len(output_tokens_list):
                        total_output_tokens += output_tokens_list[stage_idx]

                # Calculate stage time as max across all samples
                # Ray workers run in parallel, so stage time = slowest worker time
                stage_times = []
                for sample_stats in mfu_stats.values():
                    times_list = sample_stats.get("times", [])
                    if stage_idx < len(times_list):
                        stage_times.append(times_list[stage_idx])

                # Use max time if available, otherwise 0.0
                stage_time = max(stage_times) if stage_times else 0.0

                data["mfu_stats_aggregate"]["total_input_tokens"].append(total_input_tokens)
                data["mfu_stats_aggregate"]["total_output_tokens"].append(total_output_tokens)
                data["mfu_stats_aggregate"]["total_time"].append(stage_time)
                
        else:
            console.print("[MFU DEBUG] ❌ Skipping mfu_stats processing (None or empty)")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
