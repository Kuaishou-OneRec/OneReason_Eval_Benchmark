"""
Base Evaluator for all task evaluators

Provides common interface for evaluation logic.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from benchmark.console import console, success_style


class BaseEval(ABC):
    """Base class for all task evaluators"""

    def __init__(
        self,
        samples: Dict[str, Dict[str, Any]],
        task_name: Optional[str] = None,
        predictions_dir: Optional[str] = None,
        debug: bool = False,
        task_config: Optional[Dict[str, Any]] = None,
        data_dir: Optional[str] = None,
        overwrite: bool = False,
        cached_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base evaluator

        Args:
            samples: Dictionary of samples from test_generated.json
                Format: {
                    sample_id: {
                        "prompt": "...",
                        "generations": ["..."],
                        "ground_truth": "...",
                        "metadata": {...}
                    }
                }
            task_name: Task name (e.g., "math_500")
            predictions_dir: Directory to save debug files (optional)
            debug: Whether to save debug information
            task_config: Task configuration dictionary (optional)
            data_dir: Data directory path (optional)
            overwrite: Whether to overwrite existing metrics and recompute from scratch
            cached_metrics: Existing overall metrics from eval_results (optional)
        """
        self.samples = samples
        self.task_name = task_name
        self.predictions_dir = predictions_dir
        self.debug = debug
        self.task_config = task_config or {}
        self.data_dir = data_dir
        self.overwrite = overwrite
        self.cached_metrics = cached_metrics or {}
    
    def evaluate(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Evaluate the samples and return metrics

        This method provides a simplified two-level caching-aware evaluation flow:
        1. If overwrite=True, always recompute from scratch
        2. If cached overall metrics exist in eval_results, return them with empty per_sample_metrics
        3. Otherwise, compute from scratch

        Subclasses should override:
        - required_metrics property: Return list of overall metric names
        - _compute_metrics_from_scratch(): Compute all metrics from scratch

        Returns:
            Tuple of (metrics, per_sample_metrics)
        """

        # If overwrite=True, always recompute from scratch
        if self.overwrite:
            console.print("[cyan]Overwrite=True, recomputing all metrics from scratch...[/cyan]")
            return self._compute_metrics_from_scratch()

        # If cached overall metrics exist, use them
        if self._has_all_required_metrics():
            console.print("[cyan]Using existing overall metrics from eval_results...[/cyan]")
            # Return cached metrics with empty per_sample_metrics (not needed when using cache)
            return self.cached_metrics, {}

        # Otherwise, compute from scratch
        console.print("[cyan]Computing metrics from scratch...[/cyan]")
        return self._compute_metrics_from_scratch()

    def _all_samples_have_keys(self, required_keys: List[str]) -> bool:
        """Check if all samples have required keys"""
        for sample in self.samples.values():
            for key in required_keys:
                if key not in sample:
                    return False
        return True

    @property
    def required_metrics(self) -> Optional[List[str]]:
        """Define required overall metric keys"""
        return None

    def _has_all_required_metrics(self) -> bool:
        """Check if cached_metrics contains all required keys (override for custom logic)"""
        if self.required_metrics is not None:
            return all(key in self.cached_metrics for key in self.required_metrics)
        return False

    def _compute_metrics_from_scratch(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Compute metrics from scratch (override in subclasses)"""
        raise NotImplementedError("Subclasses must implement _compute_metrics_from_scratch()")

    @staticmethod
    def extract_after_think(text: str) -> str:
        """Extract text after the last </think> tag if present."""
        if '</think>' in text:
            return text.split('</think>')[-1].strip()
        return text

    def _get_llm_client(self):
        """
        Create LLM client from evaluation config.

        llm_judge_model format: "type/model_name"
        e.g. "openai/qwen3-1.7B", "gemini/gemini-2.5-flash-lite"

        Returns:
            LLM client instance, or None if creation fails
        """
        try:
            from api import load_config, get_client
        except ImportError as e:
            console.print(f"[red]Failed to import LLM evaluation modules: {e}[/red]")
            return None

        eval_config = self.task_config.get("evaluation_config", {})
        llm_judge_model = eval_config.get("llm_judge_model")
        if not llm_judge_model:
            console.print("[red]No llm_judge_model configured in evaluation_config[/red]")
            return None
        client_type, model_name = llm_judge_model.split("/", 1)
        if model_name == "configure-judge-model":
            model_name = os.environ.get("JUDGE_MODEL")
            if not model_name:
                raise ValueError("Configure JUDGE_MODEL for this task")

        try:
            config = load_config()
            model_config = config.get(client_type, {})
            model_config["model_name"] = model_name
            llm_client = get_client(client_type, **model_config)
            console.print(f"[green]Using LLM judge: {llm_judge_model}[/green]")
            return llm_client
        except Exception as e:
            console.print(f"[red]Failed to create LLM client for evaluation: {e}[/red]")
            return None

    def _evaluate_with_llm(self, eval_fn, predictions, references, eval_config, **extra_kwargs):
        """
        Generic LLM-as-Judge evaluation wrapper.

        Args:
            eval_fn: The task-specific evaluation function to call
            predictions: Dict of {sample_id: prediction_text}
            references: Dict of {sample_id: reference_text}
            eval_config: Evaluation configuration dictionary
            **extra_kwargs: Additional arguments passed to eval_fn

        Returns:
            Tuple of (metrics, per_sample_metrics)
        """
        llm_client = self._get_llm_client()
        if llm_client is None:
            return {}, {}

        max_workers = eval_config.get("llm_max_workers", 1)
        max_samples = eval_config.get("llm_max_samples", None)

        try:
            return eval_fn(
                predictions=predictions,
                references=references,
                llm_client=llm_client,
                max_workers=max_workers,
                max_samples=max_samples,
                **extra_kwargs,
            )
        except Exception as e:
            console.print(f"[red]LLM evaluation failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return {}, {}

    def _save_debug_json(
        self,
        debug_info: Dict[str, Any],
        filename: str = "debug.json"
    ) -> Optional[str]:
        """Save debug information to JSON file"""
        if not self.predictions_dir:
            return None

        debug_filename = os.path.join(self.predictions_dir, filename)
        os.makedirs(os.path.dirname(debug_filename), exist_ok=True)

        with open(debug_filename, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=2, ensure_ascii=False)
            
        console.print(f"✓ Debug information saved to: {debug_filename}", style=success_style)
        return debug_filename
