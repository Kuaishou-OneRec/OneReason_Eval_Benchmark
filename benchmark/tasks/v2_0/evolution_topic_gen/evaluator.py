"""
Evolution Topic Gen Evaluator

Evaluates model predictions on the Evolution Topic Gen dataset.
Uses DTW-based action alignment and NLI-based logic entailment evaluation.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional

from benchmark.console import console
from benchmark.tasks.base_evaluator import BaseEval
from benchmark.tasks.v2_0.evolution_topic_gen.action_logic import (
    evaluate_action_logic,
    build_nli,
    resolve_nli_model_name,
)
from benchmark.tasks.v2_0.evolution_topic_gen.utils import (
    extract_json_from_response,
    compute_combined_f1_score,
    get_debug_info,
)

logger = logging.getLogger(__name__)


class EvolutionTopicGenEvaluator(BaseEval):
    """Evolution Topic Gen task evaluator"""

    _evaluate_action_logic = staticmethod(evaluate_action_logic)

    @property
    def required_metrics(self) -> List[str]:
        return ["action_f1_score", "logic_f1_score",
                "combined_f1_score", "logic_entailment_rate"]

    def _build_nli_backend(self):
        """Build NLI backend from evaluation config.

        Returns:
            NLI backend instance, or None if NLI is disabled.
        """
        eval_config = self.task_config.get("evaluation_config", {})

        if not eval_config.get("nli_enabled", True):
            return None

        backend_name = eval_config.get("nli_backend", "huggingface")

        if backend_name == "huggingface":
            model_name = resolve_nli_model_name(
                model_name=eval_config.get("nli_model"),
                model_dir=eval_config.get("nli_model_dir"),
            )
            return build_nli("huggingface", model_name=model_name)
        elif backend_name == "llm":
            judge_model = eval_config.get("nli_judge_model", "gemini")
            return build_nli("llm", judge_model=judge_model)
        else:
            logger.warning(f"Unknown NLI backend: {backend_name}, falling back to huggingface")
            return build_nli("huggingface")

    def _compute_metrics_from_scratch(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Compute metrics from scratch by parsing predictions and evaluating
        action alignment + logic entailment.

        Returns:
            Tuple of (metrics, per_sample_metrics)
        """
        total_samples = len(self.samples)

        # Build NLI backend (lazy-loaded, shared across all samples)
        nli_backend = self._build_nli_backend()

        # Accumulators
        total_action_f1_score = 0.0
        total_logic_f1_score = 0.0
        total_combined_f1_score = 0.0
        total_logic_entailment_rate = 0.0
        parse_failure_count = 0

        # Per-sample metrics
        per_sample_metrics = {}

        # Debug information
        debug_info = {
            "parse_failures": [],
            "evaluated_samples": [],
        }

        for sample_id, sample in self.samples.items():
            # Get ground truth (logic_chain JSON from metadata["answer"])
            ground_truth = sample.get("ground_truth", "")

            # Get model generations
            generations = sample.get("generations", [])
            if not generations:
                parse_failure_count += 1
                per_sample_metrics[sample_id] = {
                    "logic_entailment_rate": None,
                    "parse_failed": True,
                }
                if self.debug:
                    debug_info["parse_failures"].append(
                        get_debug_info(sample_id, ground_truth, "", {}, parse_failed=True)
                    )
                continue

            # Evaluate each generation and average results
            gen_action_f1_scores = []
            gen_logic_f1_scores = []
            gen_combined_f1_scores = []
            gen_logic_rates = []
            best_debug_result = None

            for response_text in generations:
                cleaned_response = self.extract_after_think(response_text)

                # Extract JSON from response
                gen_json = extract_json_from_response(cleaned_response)

                if gen_json is None:
                    # JSON parsing failed for this generation
                    gen_action_f1_scores.append(0.0)
                    gen_logic_f1_scores.append(0.0)
                    gen_combined_f1_scores.append(0.0)
                    gen_logic_rates.append(None)
                    if self.debug and best_debug_result is None:
                        best_debug_result = get_debug_info(
                            sample_id, ground_truth, response_text, {}, parse_failed=True
                        )
                    continue

                try:
                    eval_result = self._evaluate_action_logic(
                        gt=ground_truth,
                        generated=gen_json,
                        nli_backend=nli_backend,
                    )

                    action_f1_score = eval_result.get("action_f1_score", 0.0)
                    logic_f1_score = eval_result.get("logic_f1_score", 0.0)
                    logic_rate = eval_result["logic_entailment_rate"]
                    combined_f1 = compute_combined_f1_score(action_f1_score, logic_f1_score)

                    gen_action_f1_scores.append(action_f1_score)
                    gen_logic_f1_scores.append(logic_f1_score)
                    gen_combined_f1_scores.append(combined_f1)
                    gen_logic_rates.append(logic_rate)

                    if self.debug:
                        if best_debug_result is None or combined_f1 > best_debug_result.get("combined_f1_score", 0):
                            best_debug_result = get_debug_info(
                                sample_id, ground_truth, response_text, eval_result
                            )
                            best_debug_result["combined_f1_score"] = combined_f1

                except Exception as e:
                    logger.warning(f"Evaluation failed for sample {sample_id}: {e}")
                    gen_action_f1_scores.append(0.0)
                    gen_logic_f1_scores.append(0.0)
                    gen_combined_f1_scores.append(0.0)
                    gen_logic_rates.append(None)
                    if self.debug and best_debug_result is None:
                        best_debug_result = get_debug_info(
                            sample_id, ground_truth, response_text, {}, parse_failed=True
                        )

            # Check if all generations failed to parse
            all_failed = all(s == 0.0 for s in gen_action_f1_scores) and all(
                r is None for r in gen_logic_rates
            )

            if all_failed:
                parse_failure_count += 1
                per_sample_metrics[sample_id] = {
                    "action_f1_score": 0.0,
                    "logic_f1_score": 0.0,
                    "combined_f1_score": 0.0,
                    "logic_entailment_rate": None,
                    "parse_failed": True,
                }
                if self.debug:
                    debug_info["parse_failures"].append(
                        best_debug_result or get_debug_info(
                            sample_id, ground_truth, generations[0], {}, parse_failed=True
                        )
                    )
                continue

            # Average across generations
            avg_action_f1_score = sum(gen_action_f1_scores) / len(gen_action_f1_scores)
            avg_logic_f1_score = sum(gen_logic_f1_scores) / len(gen_logic_f1_scores)
            avg_combined_f1 = sum(gen_combined_f1_scores) / len(gen_combined_f1_scores)
            valid_logic_rates = [r for r in gen_logic_rates if r is not None]
            avg_logic_rate = (
                sum(valid_logic_rates) / len(valid_logic_rates)
                if valid_logic_rates else None
            )

            # Accumulate
            total_action_f1_score += avg_action_f1_score
            total_logic_f1_score += avg_logic_f1_score
            total_combined_f1_score += avg_combined_f1
            if avg_logic_rate is not None:
                total_logic_entailment_rate += avg_logic_rate

            per_sample_metrics[sample_id] = {
                "action_f1_score": avg_action_f1_score,
                "logic_f1_score": avg_logic_f1_score,
                "combined_f1_score": avg_combined_f1,
                "logic_entailment_rate": avg_logic_rate,
                "parse_failed": False,
            }

            if self.debug and best_debug_result:
                debug_info["evaluated_samples"].append(best_debug_result)

        # Calculate overall metrics
        # Parse failures count as 0 score but still count in the denominator,
        # since failing to produce the required format reflects model capability.
        metrics = {
            "total_samples": total_samples,
            "evaluated_samples": total_samples - parse_failure_count,
            "parse_failures": parse_failure_count,
            "action_f1_score": (
                total_action_f1_score / total_samples if total_samples > 0 else 0.0
            ),
            "logic_f1_score": (
                total_logic_f1_score / total_samples if total_samples > 0 else 0.0
            ),
            "combined_f1_score": (
                total_combined_f1_score / total_samples if total_samples > 0 else 0.0
            ),
            "logic_entailment_rate": (
                total_logic_entailment_rate / total_samples
                if total_samples > 0 else 0.0
            ),
        }

        # Save debug information if requested
        if self.debug and self.predictions_dir:
            self._save_debug_info(debug_info, metrics)

        return metrics, per_sample_metrics

    def _save_debug_info(self, debug_info: Dict[str, Any], metrics: Dict[str, Any]):
        """Save detailed debug information to file."""

        debug_info["statistics"] = {
            "total_samples": metrics["total_samples"],
            "evaluated_samples": metrics["evaluated_samples"],
            "parse_failures": metrics["parse_failures"],
            "action_f1_score": metrics["action_f1_score"],
            "logic_f1_score": metrics["logic_f1_score"],
            "combined_f1_score": metrics["combined_f1_score"],
            "logic_entailment_rate": metrics["logic_entailment_rate"],
        }

        self._save_debug_json(debug_info, filename="debug.json")

        console.print(f"Total samples: {metrics['total_samples']}")
        console.print(f"Evaluated samples: {metrics['evaluated_samples']}")
        console.print(f"Parse failures: {metrics['parse_failures']}")
        console.print(f"Action F1 score: {metrics['action_f1_score']:.4f}")
        console.print(f"Logic F1 score: {metrics['logic_f1_score']:.4f}")
        console.print(f"Combined F1 score: {metrics['combined_f1_score']:.4f}")
        console.print(f"Logic entailment rate: {metrics['logic_entailment_rate']:.4f}")

        if debug_info["parse_failures"]:
            console.print(f"\nParse failure examples (first 3):")
            for i, item in enumerate(debug_info["parse_failures"][:3]):
                console.print(f"  Example {i+1}:")
                console.print(f"    Sample ID: {item['sample_id']}")
                gen_snippet = item.get('generation', '')[:200]
                console.print(f"    Generation snippet: {gen_snippet}...")
                console.print()
