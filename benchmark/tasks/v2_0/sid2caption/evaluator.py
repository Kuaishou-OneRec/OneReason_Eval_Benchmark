"""
SID to Caption Evaluator

Evaluates model predictions on SID to Caption task using WIP (LLM-as-Judge).
"""

import os
from typing import Dict, Any, Tuple, List

from benchmark.console import console
from benchmark.tasks.base_evaluator import BaseEval
from benchmark.tasks.v2_0.sid2caption.utils import evaluate_wip


class Sid2CaptionEvaluator(BaseEval):
    """SID to Caption task evaluator"""

    @property
    def required_metrics(self) -> List[str]:
        return ["macro_wip_double_weighted_f1"]

    def _compute_metrics_from_scratch(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        total_samples = len(self.samples)

        sample_ids = list(self.samples.keys())
        predictions = {}
        references = {}

        for sample_id in sample_ids:
            sample = self.samples[sample_id]
            references[sample_id] = sample.get("ground_truth", "")
            generations = sample.get("generations", [])
            predictions[sample_id] = generations[0] if generations else ""

        eval_config = self.task_config.get("evaluation_config", {})

        per_sample_metrics = {sample_id: {} for sample_id in sample_ids}
        metrics = {"num_samples": total_samples}

        if eval_config.get("wip_enabled", False):
            console.print("[cyan]WIP evaluation enabled, starting LLM-as-Judge evaluation...[/cyan]")
            wip_metrics, wip_per_sample = self._evaluate_with_llm(
                evaluate_wip,
                predictions=predictions,
                references=references,
                eval_config=eval_config,
                gt_cache_dir=os.path.join(self.data_dir, self.task_name),
                save_dir=self.predictions_dir,
                bertscore_model=eval_config.get("bertscore_model_type", "bert-base-chinese"),
                bertscore_num_layers=eval_config.get("bertscore_num_layers", 9),
                core_threshold=eval_config.get("wip_core_threshold", 5),
                include_failed_samples_in_macro=eval_config.get(
                    "include_failed_samples_in_macro", False
                ),
                minimum_match_success_rate=eval_config.get(
                    "minimum_match_success_rate"
                ),
            )
            metrics.update(wip_metrics)
            for sample_id in sample_ids:
                if sample_id in wip_per_sample:
                    per_sample_metrics[sample_id].update(wip_per_sample[sample_id])

        if self.debug and self.predictions_dir:
            self._save_debug_info(metrics, per_sample_metrics, predictions, references)

        return metrics, per_sample_metrics

    def _save_debug_info(
        self,
        metrics: Dict[str, Any],
        per_sample_metrics: Dict[str, Dict[str, Any]],
        predictions: Dict[str, str],
        references: Dict[str, str],
    ):
        sample_ids = list(self.samples.keys())
        debug_info = {
            "overall_metrics": metrics,
            "sample_count": len(predictions),
            "examples": [
                {
                    "sample_id": sample_ids[i],
                    "prediction": predictions.get(sample_ids[i], ""),
                    "reference": references.get(sample_ids[i], ""),
                    "wip_unweighted_f1": per_sample_metrics.get(sample_ids[i], {}).get("wip_unweighted_f1"),
                    "wip_double_weighted_f1": per_sample_metrics.get(sample_ids[i], {}).get("wip_double_weighted_f1"),
                }
                for i in range(min(10, len(sample_ids)))
            ],
        }

        self._save_debug_json(debug_info, filename="debug.json")

        console.print(f"Total samples: {metrics['num_samples']}")
        if metrics.get('macro_wip_unweighted_f1') is not None:
            console.print(f"Macro WIP Unweighted F1: {metrics['macro_wip_unweighted_f1']:.4f}")
        if metrics.get('macro_wip_double_weighted_f1') is not None:
            console.print(f"Macro WIP Double-weighted F1: {metrics['macro_wip_double_weighted_f1']:.4f}")
