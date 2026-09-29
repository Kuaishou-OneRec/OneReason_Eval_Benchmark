"""
Tongshi Choice Evaluator

Evaluates model predictions on Tongshi Choice dataset.
支持单选与多选（按 letters 集合相等比对）。
"""

from typing import Dict, Any, Tuple, List

from benchmark.console import console
from benchmark.tasks.base_evaluator import BaseEval
from benchmark.tasks.v2_0.tongshi_choice.utils import (
    parse_answer_from_response,
    parse_ground_truth,
    compute_accuracy,
    compute_pass_at_k,
    get_debug_info,
)


class TongshiChoiceEvaluator(BaseEval):
    """Tongshi Choice task evaluator"""

    @property
    def required_metrics(self) -> List[str]:
        k_values = self.task_config.get("evaluation_config", {}).get("k_values", [1])
        metrics = ["accuracy"]
        for k in k_values:
            metrics.append(f"pass@{k}")
        return metrics

    def _compute_metrics_from_scratch(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        total_samples = len(self.samples)
        k_values = self.task_config.get("evaluation_config", {}).get("k_values", [1])
        pass_at_k_counts = {k: 0 for k in k_values}
        total_accuracy = 0.0

        per_sample_metrics: Dict[str, Dict[str, Any]] = {}
        debug_info = {
            "parse_failures": [],
            "incorrect_predictions": [],
            "correct_predictions": [],
        }

        for sample_id, sample in self.samples.items():
            ground_truth_raw = sample.get("ground_truth", "")
            ground_truth_answer = parse_ground_truth(ground_truth_raw)

            generations = sample.get("generations", []) or []
            predicted_answers: List[str] = []
            response_texts: List[str] = []
            for response_text in generations:
                response_texts.append(response_text)
                cleaned = self.extract_after_think(response_text)
                predicted_answers.append(parse_answer_from_response(cleaned))

            parse_failed = (not predicted_answers) or all(p == "" for p in predicted_answers)

            if predicted_answers:
                correct_count = sum(
                    1 for ans in predicted_answers
                    if ans and compute_accuracy(ans, ground_truth_answer)
                )
                sample_accuracy = correct_count / len(predicted_answers)
            else:
                sample_accuracy = 0.0

            pass_at_k_results: Dict[str, bool] = {}
            for k in k_values:
                is_pass = compute_pass_at_k(predicted_answers, ground_truth_answer, k)
                pass_at_k_results[f"pass@{k}"] = is_pass
                if is_pass:
                    pass_at_k_counts[k] += 1

            per_sample_metrics[sample_id] = {
                "parsed_answers": predicted_answers,
                "parse_failed": parse_failed,
                "accuracy": sample_accuracy,
                **pass_at_k_results,
            }
            total_accuracy += sample_accuracy

            if self.debug:
                first_pred = predicted_answers[0] if predicted_answers else ""
                first_text = response_texts[0] if response_texts else ""
                is_first_correct = (
                    compute_accuracy(first_pred, ground_truth_answer) if first_pred else False
                )
                debug_item = get_debug_info(
                    item={
                        "uuid": sample_id,
                        "answer": ground_truth_answer,
                        "prediction": first_text,
                        "prompt": sample.get("prompt", ""),
                    },
                    predicted_answer=first_pred,
                    ground_truth_answer=ground_truth_answer,
                    is_correct=is_first_correct,
                )
                debug_item["all_predicted_answers"] = predicted_answers
                debug_item["pass_at_k_results"] = pass_at_k_results

                if parse_failed:
                    debug_info["parse_failures"].append(debug_item)
                elif not is_first_correct:
                    debug_info["incorrect_predictions"].append(debug_item)
                else:
                    debug_info["correct_predictions"].append(debug_item)

        metrics: Dict[str, Any] = {
            "total_samples": total_samples,
            "parse_failures": len(debug_info["parse_failures"]) if self.debug else 0,
            "accuracy": total_accuracy / total_samples if total_samples > 0 else 0.0,
        }
        for k in k_values:
            rate = pass_at_k_counts[k] / total_samples if total_samples > 0 else 0.0
            metrics[f"pass@{k}"] = rate
            metrics[f"pass@{k}_count"] = pass_at_k_counts[k]

        if self.debug and self.predictions_dir:
            self._save_debug_info(debug_info, metrics)

        return metrics, per_sample_metrics

    def _save_debug_info(self, debug_info: Dict[str, Any], metrics: Dict[str, Any]):
        debug_info["statistics"] = {
            "total_samples": metrics["total_samples"],
            "parse_failures_count": len(debug_info["parse_failures"]),
            "incorrect_predictions_count": len(debug_info["incorrect_predictions"]),
            "correct_predictions_count": len(debug_info["correct_predictions"]),
        }
        for key, value in metrics.items():
            if key.startswith("pass@"):
                debug_info["statistics"][key] = value

        self._save_debug_json(debug_info, filename="debug.json")

        console.print(f"Total samples: {metrics['total_samples']}")
        console.print(f"Parse failures: {len(debug_info['parse_failures'])}")
        console.print(f"Incorrect predictions: {len(debug_info['incorrect_predictions'])}")
        console.print(f"Correct predictions: {len(debug_info['correct_predictions'])}")
        for key, value in sorted(metrics.items()):
            if key.startswith("pass@") and not key.endswith("_count"):
                console.print(f"{key}: {value:.4f}")

        if debug_info["parse_failures"]:
            console.print("\nParse failure examples (first 3):")
            for i, item in enumerate(debug_info["parse_failures"][:3]):
                console.print(f"  Example {i+1}:")
                console.print(f"    UUID: {item['uuid']}")
                console.print(f"    Ground truth: {item['ground_truth_answer']}")
                console.print(f"    Response snippet: {item['response'][:200]}...")
                console.print()

        if debug_info["incorrect_predictions"]:
            console.print("Incorrect prediction examples (first 3):")
            for i, item in enumerate(debug_info["incorrect_predictions"][:3]):
                console.print(f"  Example {i+1}:")
                console.print(f"    UUID: {item['uuid']}")
                console.print(f"    Predicted: {item['predicted_answer']}")
                console.print(f"    Ground truth: {item['ground_truth_answer']}")
                console.print(f"    Response snippet: {item['response'][:200]}...")
                console.print()
