"""
Evolution Select Evaluator

Evaluates whether the model can select all timeline interactions relevant to a given topic.
"""

from typing import Dict, Any, Tuple, List

from benchmark.console import console
from benchmark.tasks.base_evaluator import BaseEval
from .utils import parse_ground_truth_items, parse_prediction_items, compute_set_metrics


class EvolutionSelectEvaluator(BaseEval):
    """Set-overlap evaluator with F1 as the primary metric."""

    @property
    def required_metrics(self) -> List[str]:
        return ["precision", "recall", "f1"]

    def _compute_metrics_from_scratch(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        total_samples = len(self.samples)
        per_sample_metrics: Dict[str, Dict[str, Any]] = {}

        precision_scores = []
        recall_scores = []
        f1_scores = []

        for sample_id, sample in self.samples.items():
            ground_truth_items = parse_ground_truth_items(sample.get("ground_truth"))

            generations = sample.get("generations", [])
            response = self.extract_after_think(generations[0]) if generations else ""
            predicted_items = parse_prediction_items(response)

            sample_metrics = compute_set_metrics(predicted_items, ground_truth_items)
            precision_scores.append(sample_metrics["precision"])
            recall_scores.append(sample_metrics["recall"])
            f1_scores.append(sample_metrics["f1"])

            per_sample_metrics[sample_id] = {
                **sample_metrics,
                "prediction_items": predicted_items,
                "ground_truth_items": ground_truth_items,
            }

        n = len(precision_scores)
        metrics = {
            "num_samples": total_samples,
            "precision": sum(precision_scores) / n if n else 0.0,
            "recall": sum(recall_scores) / n if n else 0.0,
            "f1": sum(f1_scores) / n if n else 0.0,
        }

        if self.debug and self.predictions_dir:
            self._save_debug_info(metrics, per_sample_metrics)

        return metrics, per_sample_metrics

    def _save_debug_info(
        self,
        metrics: Dict[str, Any],
        per_sample_metrics: Dict[str, Dict[str, Any]],
    ) -> None:
        sample_ids = list(self.samples.keys())
        debug_info = {
            "overall_metrics": metrics,
            "sample_count": len(sample_ids),
            "examples": [
                {
                    "sample_id": sample_id,
                    "prediction_items": per_sample_metrics[sample_id]["prediction_items"],
                    "ground_truth_items": per_sample_metrics[sample_id]["ground_truth_items"],
                    "precision": per_sample_metrics[sample_id]["precision"],
                    "recall": per_sample_metrics[sample_id]["recall"],
                    "f1": per_sample_metrics[sample_id]["f1"],
                    "hit_count": per_sample_metrics[sample_id]["hit_count"],
                }
                for sample_id in sample_ids[:10]
            ],
        }

        self._save_debug_json(debug_info, filename="debug.json")

        console.print(f"Total samples: {metrics['num_samples']}")
        console.print(f"Precision: {metrics['precision']:.4f}")
        console.print(f"Recall: {metrics['recall']:.4f}")
        console.print(f"F1: {metrics['f1']:.4f}")
