"""
Recommendation Task Evaluator

Universal evaluator for all recommendation tasks.
Computes Pass@k and Position1_Pass@k metrics.
"""

import json
import time
from collections import defaultdict
from typing import Dict, Any, Tuple, List

from tqdm import tqdm
from benchmark.console import console, warning_style
from benchmark.tasks.base_evaluator import BaseEval
from benchmark.tasks.v2_0.recommendation import utils as utils_sid
from benchmark.tasks.v2_0.recommendation import utils_by_pid as utils_pid_from_client
from benchmark.tasks.v2_0.recommendation import utils_8sid2emb
from benchmark.tasks.v1_0.recommendation import utils_by_pid as utils_pid_from_json


WEIGHT_FIELDS = ("incremental_value", "play_count", "duration")


class RecommendationEvaluator(BaseEval):
    """
    Universal evaluator for recommendation tasks

    Supports:
    - label_cond: Predict next video given specified consumption behavior
    - video: Next video prediction
    - product: Predict next clicked product
    - ad: Predict next clicked advertisement

    Metrics:
    - Pass@k: Check if any of top-k predictions match any ground truth SID
    - Position1_Pass@k: Check if any of top-k predictions match the first ground truth SID
    """

    @property
    def required_metrics(self) -> List[str]:
        """Define required overall metrics for Recommendation evaluation"""
        evaluation_config = self.task_config.get("evaluation_config", {})
        generation_config = self.task_config.get("generation_config", {})
        k_values = evaluation_config.get("k_values", [128])
        evaluation_mode = evaluation_config.get("evaluation_mode", "sid")
        num_sid_tokens = generation_config.get("max_new_tokens", 8)

        metrics = []

        # SID合法率指标（所有模式都计算）
        for k in k_values:
            metrics.append(f"sid_validity@{k}")

        if evaluation_mode in ("sid", "both"):
            for k in k_values:
                metrics.extend([f"pass@{k}", f"position1_pass@{k}", f"recall@{k}"])
            metrics.extend([f"accuracy_at_pos{i}" for i in range(1, num_sid_tokens + 1)])

        if evaluation_mode in ("pid", "both"):
            for k in k_values:
                metrics.extend([f"pid_pass@{k}", f"pid_position1_pass@{k}", f"pid_recall@{k}"])
            # random baseline: always compute random strategy metrics alongside configured strategy
            if evaluation_config.get("sid_to_pid_strategy"):
                for k in k_values:
                    metrics.append(f"pid_pass_random@{k}")
                    metrics.append(f"pid_recall_random@{k}")
            for weight_field in WEIGHT_FIELDS:
                for k in k_values:
                    metrics.append(f"{weight_field}_weighted_pid_recall@{k}")

        # Embedding similarity metrics (only when centroid_model_dir is configured)
        if evaluation_config.get("centroid_model_dir"):
            for k in k_values:
                metrics.extend([f"mean_emb_sim@{k}", f"max_emb_sim@{k}"])

        if evaluation_mode in ("sid", "both"):
            for k in k_values:
                metrics.append(f"sid_1th_recall@{k}")

        return metrics

    @staticmethod
    def _parse_metadata_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return []
        if isinstance(value, (int, float)):
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return []

    @classmethod
    def _parse_pid_list(cls, value: Any) -> List[Any]:
        return cls._parse_metadata_list(value)

    @classmethod
    def _parse_weight_list(cls, value: Any) -> List[Any]:
        return cls._parse_metadata_list(value)

    @staticmethod
    def _is_weighted_pid_metric_name(name: str) -> bool:
        return any(name.startswith(f"{field}_weighted_pid_") for field in WEIGHT_FIELDS)

    @staticmethod
    def _deduped_pid_weight_pairs(answer_pid: List[Any], weights: List[Any]) -> List[Tuple[Any, Any]]:
        seen = set()
        pairs = []
        for pid, weight in zip(answer_pid, weights):
            if pid is None or pid == 0 or pid in seen:
                continue
            seen.add(pid)
            pairs.append((pid, weight))
        return pairs

    def _select_generations_by_strategy(
        self,
        generations: List[str],
        logprobs: List[float],
        strategy: str
    ) -> List[str]:
        """
        Select and reorder generations based on the specified strategy

        Args:
            generations: List of generation strings
            logprobs: List of cumulative logprobs for each generation
            strategy: Selection strategy ('first_k' or 'top_k_by_logprobs')

        Returns:
            Reordered list of generations

        Raises:
            ValueError: If strategy is 'top_k_by_logprobs' but logprobs data is invalid
        """
        if strategy == "first_k":
            # Keep original order
            return generations
        elif strategy == "top_k_by_logprobs":
            # Validate logprobs data
            if not logprobs:
                raise ValueError(
                    f"Strategy 'top_k_by_logprobs' requires logprobs data, but logprobs is empty. "
                    f"Please ensure the generation was run with logprobs enabled."
                )
            if len(logprobs) != len(generations):
                raise ValueError(
                    f"Strategy 'top_k_by_logprobs' requires logprobs length to match generations length. "
                    f"Got logprobs length {len(logprobs)}, generations length {len(generations)}."
                )

            # Sort generations by logprobs in descending order (higher logprob = better)
            paired = list(zip(generations, logprobs))
            paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)

            # Deduplicate while preserving order (keep first occurrence with highest logprob)
            seen = set()
            unique_generations = []
            for gen, _ in paired_sorted:
                if gen not in seen:
                    seen.add(gen)
                    unique_generations.append(gen)

            return unique_generations
        else:
            raise ValueError(
                f"Unknown selection strategy: '{strategy}'. "
                f"Supported strategies: 'first_k', 'top_k_by_logprobs'"
            )

    def _evaluate_single_mode(
        self,
        k_values: List[int],
        evaluation_mode: str,
        select_k_strategy: str,
        pid_client=None,
        code_to_pid=None,
        sid_to_pid_strategy=None,
        sid2emb: Any = None,
        num_sid_tokens: int = 8,
        emb_score_fn=None,
        valid_sid_set=None,
        sid_encoding: str = "multiply",
        sid_pattern: str = None,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Evaluate samples using a single mode (SID or PID)

        Returns:
            Tuple of (counters, per_sample_metrics, debug_info)
            - counters: metric_name -> cumulative sum across all samples
            - per_sample_metrics: sample_id -> {metric_name: value}
            - debug_info: debug information dict with passed/failed/no_generation lists
        """
        # Select utils module based on mode
        if evaluation_mode == "sid":
            utils = utils_sid
        elif evaluation_mode == "pid":
            utils = utils_pid_from_client
        else:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}")

        has_validity_check = pid_client is not None or code_to_pid is not None
        has_emb_sim = sid2emb is not None and evaluation_mode == "sid"

        # Build metric templates for dynamic failed metrics generation
        # Format: (template_with_{k}, default_value)
        k_metric_templates = [
            ("pass@{k}", False),
            ("position1_pass@{k}", False),
            ("recall@{k}", 0.0),
        ]
        has_random_baseline = code_to_pid is not None and evaluation_mode == "pid"
        if has_random_baseline:
            k_metric_templates.append(("pass_random@{k}", False))
            k_metric_templates.append(("recall_random@{k}", 0.0))
        if evaluation_mode == "sid":
            k_metric_templates.append(("sid_1th_recall@{k}", 0.0))
        if has_validity_check and evaluation_mode == "sid":
            k_metric_templates.append(("sid_validity@{k}", 0.0))
        if has_emb_sim:
            k_metric_templates.extend([("mean_emb_sim@{k}", 0.0), ("max_emb_sim@{k}", 0.0)])

        def create_failed_metrics():
            """Create metrics dict for failed samples from templates"""
            metrics = {}
            for template, default in k_metric_templates:
                for k in k_values:
                    metrics[template.format(k=k)] = default
            if evaluation_mode == "sid":
                for i in range(1, num_sid_tokens + 1):
                    metrics[f"accuracy_at_pos{i}"] = False
            return metrics

        # Unified counters: metric_name -> cumulative sum
        counters = defaultdict(float)
        per_sample_metrics = {}
        debug_info = {
            "passed_samples": [],
            "failed_samples": [],
            "no_generation_samples": [],
        }

        timers = {"sid2pid": 0.0, "pass_recall": 0.0, "token_acc": 0.0, "sid_valid": 0.0, "emb_sim": 0.0}
        pbar = tqdm(self.samples.items(), desc=f"  {evaluation_mode.upper()} eval", total=len(self.samples))
        for sample_id, sample in pbar:
            generations = sample.get("generations", [])
            logprobs = sample.get("logprobs", [])

            if not generations:
                per_sample_metrics[sample_id] = create_failed_metrics()
                if self.debug:
                    debug_info["no_generation_samples"].append({
                        "sample_id": sample_id,
                        "ground_truth": sample.get("ground_truth", "") if evaluation_mode == "sid" else sample.get("metadata", {}).get("answer_pid", []),
                    })
                continue

            # Get ground truth based on mode
            if evaluation_mode == "sid":
                ground_truth = sample.get("ground_truth", "")
                ground_truth_ids = utils.extract_ids_from_answer(ground_truth)
                first_ground_truth_id = utils.extract_first_id_from_answer(ground_truth)
            else:  # pid mode
                ground_truth_pids = sample.get("metadata", {}).get("answer_pid")
                if ground_truth_pids is None:
                    ground_truth_pids = sample.get("metadata", {}).get("answer_iid", [])
                if isinstance(ground_truth_pids, str):
                    ground_truth_pids = json.loads(ground_truth_pids)
                if isinstance(ground_truth_pids, (int, float)):
                    ground_truth_pids = [ground_truth_pids]
                ground_truth_ids = utils.extract_ids_from_answer(ground_truth_pids)
                first_ground_truth_id = utils.extract_first_id_from_answer(ground_truth_pids)

            if not ground_truth_ids:
                console.print(f"Sample {sample_id}: no valid ID found in ground truth ({evaluation_mode} mode)", style=warning_style)
                per_sample_metrics[sample_id] = create_failed_metrics()
                continue

            # Apply selection strategy to reorder generations
            selected_generations = self._select_generations_by_strategy(
                generations=generations, logprobs=logprobs, strategy=select_k_strategy
            )

            # Batch extract PIDs from generations (used for PID mode and validity check)
            # Reuse cached PID results if available, unless overwrite is requested
            cached_pids = sample.get("pid_generations")
            t0 = time.time()
            if not self.overwrite and cached_pids is not None and len(cached_pids) == len(selected_generations):
                pid_predictions = cached_pids
            elif pid_client:
                pid_predictions = utils_pid_from_client.extract_ids_from_generations(selected_generations, pid_client)
            elif code_to_pid is not None:
                pid_predictions = [
                    utils_pid_from_json.extract_id_from_generation(
                        gen,
                        code_to_pid,
                        sid_to_pid_strategy,
                        encoding=sid_encoding,
                        sid_pattern=sid_pattern,
                    )
                    for gen in selected_generations
                ]
            else:
                pid_predictions = [0] * len(selected_generations)
            timers["sid2pid"] += time.time() - t0

            # Extract predicted IDs from selected generations
            if evaluation_mode == "sid":
                predicted_ids = [utils.extract_id_from_generation(gen) for gen in selected_generations]
            else:
                predicted_ids = pid_predictions

            # Compute metrics for each k — accumulate into single dict
            sample_metrics = {}

            t0 = time.time()
            for k in k_values:
                pass_result = utils.compute_pass_at_k(predicted_ids, ground_truth_ids, k)
                sample_metrics[f"pass@{k}"] = pass_result
                counters[f"pass@{k}"] += int(pass_result)

                pos1_result = utils.compute_position1_pass_at_k(predicted_ids, first_ground_truth_id, k)
                sample_metrics[f"position1_pass@{k}"] = pos1_result
                counters[f"position1_pass@{k}"] += int(pos1_result)

                recall_result = utils.compute_recall_at_k(predicted_ids, ground_truth_ids, k)
                sample_metrics[f"recall@{k}"] = recall_result
                counters[f"recall@{k}"] += recall_result

            timers["pass_recall"] += time.time() - t0

            # Recall on the first SID code only (coarse-grained cluster hit).
            # Reuse utils.extract_sid_codes and utils.compute_recall_at_k.
            if evaluation_mode == "sid":
                pred_first = [str(codes[0]) if (codes := utils.extract_sid_codes(sid)) else "" for sid in predicted_ids]
                gt_first = {str(codes[0]) for sid in ground_truth_ids if (codes := utils.extract_sid_codes(sid))}
                for k in k_values:
                    recall = utils.compute_recall_at_k(pred_first, gt_first, k)
                    sample_metrics[f"sid_1th_recall@{k}"] = recall
                    counters[f"sid_1th_recall@{k}"] += recall

            # Compute random baseline metrics (PID mode only, when JSON mapping available)
            if has_random_baseline:
                random_pid_predictions = [
                    utils_pid_from_json.extract_id_from_generation(
                        gen,
                        code_to_pid,
                        "random",
                        encoding=sid_encoding,
                        sid_pattern=sid_pattern,
                    )
                    for gen in selected_generations
                ]
                for k in k_values:
                    random_pass = utils_pid_from_json.compute_pass_at_k(random_pid_predictions, ground_truth_ids, k)
                    sample_metrics[f"pass_random@{k}"] = random_pass
                    counters[f"pass_random@{k}"] += int(random_pass)

                    random_recall = utils_pid_from_json.compute_recall_at_k(random_pid_predictions, ground_truth_ids, k)
                    sample_metrics[f"recall_random@{k}"] = random_recall
                    counters[f"recall_random@{k}"] += random_recall

            # Compute weighted_pid_recall@k (PID mode only)
            if evaluation_mode == "pid":
                metadata = sample.get("metadata", {})
                answer_pid_raw = metadata.get("answer_pid")
                if answer_pid_raw is None:
                    answer_pid_raw = metadata.get("answer_iid", [])
                answer_pid = self._parse_pid_list(answer_pid_raw)
                weight_lists = {
                    weight_field: self._parse_weight_list(metadata.get(weight_field))
                    for weight_field in WEIGHT_FIELDS
                }
                has_all_weight_fields = bool(answer_pid) and all(
                    bool(weight_lists[weight_field]) for weight_field in WEIGHT_FIELDS
                )
                for weight_field in WEIGHT_FIELDS:
                    for k in k_values:
                        metric_name = f"{weight_field}_weighted_pid_recall@{k}"
                        if has_all_weight_fields:
                            w_recall = utils_pid_from_client.compute_weighted_pid_recall_at_k(
                                pid_generations=predicted_ids,
                                answer_pid=answer_pid,
                                incremental_value=weight_lists[weight_field],
                                k=k,
                            )
                        else:
                            w_recall = 0.0
                        sample_metrics[metric_name] = w_recall
                        counters[metric_name] += w_recall

            # Compute token-level accuracy (SID mode only)
            t0 = time.time()
            if evaluation_mode == "sid":
                first_gen = selected_generations[0] if selected_generations else ""
                first_gt = first_ground_truth_id if first_ground_truth_id else ""
                token_hits = utils.compute_token_accuracy(first_gen, first_gt, num_sid_tokens)
                for i, hit in enumerate(token_hits):
                    pos = i + 1
                    sample_metrics[f"accuracy_at_pos{pos}"] = hit
                    counters[f"accuracy_at_pos{pos}"] += int(hit)

            timers["token_acc"] += time.time() - t0

            # Compute SID validity for each k (SID mode only)
            t0 = time.time()
            if evaluation_mode == "sid" and pid_client is not None:
                for k in k_values:
                    validity_rate = utils_pid_from_client.compute_sid_validity(pid_predictions, k)
                    sample_metrics[f"sid_validity@{k}"] = validity_rate
                    counters[f"sid_validity@{k}"] += validity_rate
            elif evaluation_mode == "sid" and code_to_pid is not None:
                for k in k_values:
                    validity_rate = utils_pid_from_json.compute_sid_validity(
                        selected_generations,
                        valid_sid_set,
                        k,
                        encoding=sid_encoding,
                        sid_pattern=sid_pattern,
                    )
                    sample_metrics[f"sid_validity@{k}"] = validity_rate
                    counters[f"sid_validity@{k}"] += validity_rate

            # Compute embedding similarity (SID mode only, when sid2emb available)
            timers["sid_valid"] += time.time() - t0
            t0 = time.time()
            if has_emb_sim:
                max_k = max(k_values)
                score_fn = emb_score_fn or utils_8sid2emb.compute_all_pred_scores
                all_pred_scores = score_fn(
                    predicted_ids, ground_truth_ids, sid2emb, max_k
                )
                for k in k_values:
                    scores_k = all_pred_scores[:k]
                    if scores_k:
                        mean_sim = float(sum(scores_k) / len(scores_k))
                        max_sim = float(max(scores_k))
                    else:
                        mean_sim, max_sim = 0.0, 0.0
                    sample_metrics[f"mean_emb_sim@{k}"] = mean_sim
                    sample_metrics[f"max_emb_sim@{k}"] = max_sim
                    counters[f"mean_emb_sim@{k}"] += mean_sim
                    counters[f"max_emb_sim@{k}"] += max_sim

            # For PID mode, save pid_generations
            timers["emb_sim"] += time.time() - t0
            pbar.set_postfix({k: f"{v:.1f}s" for k, v in timers.items()}, refresh=False)
            if evaluation_mode == "pid":
                sample_metrics["generations"] = [pid if pid is not None else -1 for pid in predicted_ids]

            per_sample_metrics[sample_id] = sample_metrics

            # Debug information collection
            if self.debug:
                metadata = sample.get("metadata", {})
                raw_prompt = metadata.get("raw_prompt", "")
                pass_results = {mk: mv for mk, mv in sample_metrics.items() if mk.startswith("pass@")}
                pos1_results = {mk: mv for mk, mv in sample_metrics.items() if mk.startswith("position1_pass@")}

                if evaluation_mode == "sid":
                    debug_item = utils.get_debug_info(
                        sample_id=sample_id,
                        generations=generations,
                        ground_truth=sample.get("ground_truth", ""),
                        pass_results=pass_results,
                        position1_pass_results=pos1_results,
                        raw_prompt=raw_prompt,
                    )
                else:  # pid mode
                    answer_pid = metadata.get("answer_pid", metadata.get("answer_iid", []))
                    if isinstance(answer_pid, str):
                        answer_pid = json.loads(answer_pid)
                    if isinstance(answer_pid, (int, float)):
                        answer_pid = [answer_pid]
                    if code_to_pid is not None:
                        debug_item = utils_pid_from_json.get_debug_info(
                            sample_id=sample_id,
                            generations=generations,
                            ground_truth=answer_pid,
                            pass_results=pass_results,
                            position1_pass_results=pos1_results,
                            code_to_pid=code_to_pid,
                            strategy=sid_to_pid_strategy,
                            raw_prompt=raw_prompt,
                            encoding=sid_encoding,
                            sid_pattern=sid_pattern,
                        )
                    else:
                        debug_item = utils_pid_from_client.get_debug_info(
                            sample_id=sample_id,
                            generations=generations,
                            ground_truth=answer_pid,
                            pass_results=pass_results,
                            position1_pass_results=pos1_results,
                            predicted_pids=predicted_ids,
                            raw_prompt=raw_prompt,
                        )

                if any(pass_results.values()):
                    debug_info["passed_samples"].append(debug_item)
                else:
                    debug_info["failed_samples"].append(debug_item)

        return dict(counters), per_sample_metrics, debug_info

    def _calculate_metrics_from_counts(
        self,
        counters: Dict[str, float],
        total_samples: int,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calculate metrics by dividing cumulative counters by total_samples

        Args:
            counters: metric_name -> cumulative sum
            total_samples: Total number of samples
            prefix: Prefix to add to metric names (e.g., "pid_")

        Returns:
            Dictionary of calculated metrics (averages)
        """
        metrics = {}
        for name, total in counters.items():
            if prefix and self._is_weighted_pid_metric_name(name):
                key = name
            else:
                key = f"{prefix}{name}" if prefix else name
            metrics[key] = total / total_samples if total_samples > 0 else 0.0
        return metrics

    def _compute_metrics_from_scratch(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Compute all evaluation metrics from scratch

        Returns:
            Tuple of (metrics, per_sample_metrics)
        """
        total_samples = len(self.samples)

        # Get configuration
        evaluation_config = self.task_config.get('evaluation_config', {})
        generation_config = self.task_config.get('generation_config', {})
        k_values = evaluation_config.get("k_values", [128])
        select_k_strategy = evaluation_config.get('select_k', 'first_k')
        evaluation_mode = evaluation_config.get('evaluation_mode', 'both')
        num_sid_tokens = generation_config.get("max_new_tokens", 8)

        # Initialize PID conversion: pid_lookup_client (video/ad/live) or JSON file (product)
        from pathlib import Path
        sid_to_pid_strategy = evaluation_config.get("sid_to_pid_strategy")
        pid_client = None
        code_to_pid = None
        sid_encoding = "multiply"

        if sid_to_pid_strategy:
            # JSON file mode (product) — use v1.0 functions
            task_name = self.task_config.get("name", "")
            mapping_filename = evaluation_config.get("mapping_filename",
                "sid2iid.json" if "product" in task_name else "sid2pid.json")
            pid_mapping_path = str(Path(self.data_dir) / mapping_filename)
            if Path(pid_mapping_path).exists():
                console.print(f"[cyan]Loading SID mapping from {pid_mapping_path}...[/cyan]")
                code_to_pid = utils_pid_from_json.load_pid_mapping(pid_mapping_path)
                sid_encoding = utils_pid_from_json.detect_sid_encoding(code_to_pid.keys())
                console.print(f"[cyan]Valid SID count: {len(code_to_pid)}[/cyan]")
            else:
                raise FileNotFoundError(f"Required SID-to-PID mapping not found: {pid_mapping_path}")
        elif evaluation_mode in ("pid", "both"):
            raise ValueError("PID scoring requires a local SID-to-PID mapping; configure sid_to_pid_strategy and mapping_filename")
        # Override sid_encoding with custom function if provided
        sid_encode_fn = evaluation_config.get("sid_encode_fn")
        if sid_encode_fn:
            sid_encoding = eval(sid_encode_fn)
            console.print(f"[cyan]Using custom sid_encode_fn: {sid_encode_fn}[/cyan]")
        sid_pattern = evaluation_config.get("sid_pattern")
        if sid_pattern:
            console.print(f"[cyan]Using custom sid_pattern: {sid_pattern}[/cyan]")

        centroid_model_dir = evaluation_config.get("centroid_model_dir")
        valid_sid_set = set(code_to_pid.keys()) if code_to_pid is not None else None
        sid2emb_model = None
        emb_score_fn = utils_8sid2emb.compute_all_pred_scores
        if centroid_model_dir:
            if Path(centroid_model_dir).exists():
                centroid_num_tokens = evaluation_config.get("centroid_num_tokens", 8)
                centroid_num_layers = evaluation_config.get("centroid_num_layers", 1)
                if centroid_num_tokens != 8 or centroid_num_layers != 1:
                    from benchmark.tasks.v3_1 import utils_sid2emb
                    console.print(f"[cyan]Loading Sid2Emb centroids from {centroid_model_dir} (tokens={centroid_num_tokens}, layers={centroid_num_layers})...[/cyan]")
                    sid2emb_model = utils_sid2emb.Sid2Emb(centroid_model_dir, centroid_num_tokens, centroid_num_layers)
                    emb_score_fn = utils_sid2emb.compute_all_pred_scores
                else:
                    console.print(f"[cyan]Loading Sid2Emb centroids from {centroid_model_dir}...[/cyan]")
                    sid2emb_model = utils_8sid2emb.Sid2Emb(centroid_model_dir)
                console.print(f"[cyan]Sid2Emb model loaded successfully[/cyan]")
            else:
                console.print(f"[yellow]Warning: centroid_model_dir not found: {centroid_model_dir}, skipping embedding similarity[/yellow]")

        # Define evaluation modes to run
        # Format: (mode_name, metric_prefix, debug_filename, log_message)
        modes_config = {
            "sid": [("sid", "", "debug.json", "Evaluating using SID mode...")],
            "pid": [("pid", "pid_", "debug_pid.json", "Evaluating using PID mode...")],
            "both": [
                ("sid", "", "debug_sid.json", "  Running SID evaluation..."),
                ("pid", "pid_", "debug_pid.json", "  Running PID evaluation...")
            ]
        }

        if evaluation_mode not in modes_config:
            raise ValueError(f"Invalid evaluation_mode: '{evaluation_mode}'. Must be 'sid', 'pid', or 'both'")

        if evaluation_mode == "both":
            console.print("[cyan]Evaluating using both SID and PID modes...[/cyan]")

        # Initialize metrics
        metrics = {"total_samples": total_samples}
        per_sample_metrics = {}
        all_debug_info = {}

        # Run evaluation for each configured mode
        for mode_name, metric_prefix, debug_filename, log_message in modes_config[evaluation_mode]:
            console.print(f"[cyan]{log_message}[/cyan]")

            # Run evaluation
            counters, mode_per_sample_metrics, debug_info = self._evaluate_single_mode(
                k_values=k_values,
                evaluation_mode=mode_name,
                select_k_strategy=select_k_strategy,
                pid_client=pid_client,
                code_to_pid=code_to_pid,
                sid_to_pid_strategy=sid_to_pid_strategy,
                sid2emb=sid2emb_model if mode_name == "sid" else None,
                num_sid_tokens=num_sid_tokens,
                emb_score_fn=emb_score_fn,
                valid_sid_set=valid_sid_set,
                sid_encoding=sid_encoding,
                sid_pattern=sid_pattern,
            )

            # Calculate and add metrics
            mode_metrics = self._calculate_metrics_from_counts(
                counters, total_samples, metric_prefix
            )
            metrics.update(mode_metrics)

            # Merge per-sample metrics with appropriate prefix
            for sample_id, sample_metric in mode_per_sample_metrics.items():
                if sample_id not in per_sample_metrics:
                    per_sample_metrics[sample_id] = {}
                # Add metrics with prefix (for PID mode) or without (for SID mode)
                if metric_prefix:
                    # PID mode: add prefix to metric names, except weighted_pid_* (already named correctly)
                    for metric_name, metric_value in sample_metric.items():
                        if self._is_weighted_pid_metric_name(metric_name):
                            per_sample_metrics[sample_id][metric_name] = metric_value
                        else:
                            per_sample_metrics[sample_id][f"{metric_prefix}{metric_name}"] = metric_value
                else:
                    # SID mode: no prefix
                    per_sample_metrics[sample_id].update(sample_metric)

            # Store debug info for later saving
            if self.debug and self.predictions_dir:
                all_debug_info[mode_name] = (debug_info, debug_filename, mode_metrics)

        # Save debug info
        if self.debug and self.predictions_dir:
            for mode_name, (debug_info, debug_filename, mode_metrics) in all_debug_info.items():
                # For single mode, include all metrics; for both mode, filter by prefix
                if evaluation_mode == "both":
                    prefix = "pid_" if mode_name == "pid" else ""
                    filtered_metrics = {
                        k: v for k, v in mode_metrics.items()
                        if k == "total_samples" or k.startswith(prefix)
                    }
                    filtered_metrics["total_samples"] = total_samples
                else:
                    filtered_metrics = dict(metrics)

                self._save_debug_info(debug_info, filtered_metrics, debug_filename)

        # Record configuration
        metrics["select_k_strategy"] = select_k_strategy
        metrics["evaluation_mode"] = evaluation_mode

        # Aggregate per-sample CoT quality metrics (if present). Each sample
        # may carry a "cot_metrics" dict written by generation_runner. We mean
        # over the samples that have a numeric value for each metric — samples
        # without metrics (metrics disabled / failure) are skipped so the mean
        # reflects only the samples we actually scored.
        cot_metric_sums: Dict[str, float] = defaultdict(float)
        cot_metric_counts: Dict[str, int] = defaultdict(int)
        for sample in self.samples.values():
            sample_cot = sample.get("cot_metrics")
            if not isinstance(sample_cot, dict):
                continue
            for metric_name, metric_value in sample_cot.items():
                if isinstance(metric_value, (int, float)):
                    cot_metric_sums[metric_name] += float(metric_value)
                    cot_metric_counts[metric_name] += 1
        for metric_name, count in cot_metric_counts.items():
            if count > 0:
                metrics[f"{metric_name}_mean"] = cot_metric_sums[metric_name] / count
                metrics[f"{metric_name}_count"] = count

        return metrics, per_sample_metrics

    def _save_debug_info(self, debug_info: Dict[str, Any], metrics: Dict[str, Any], debug_filename: str = None):
        """
        Save detailed debug information to file

        Args:
            debug_info: Debug information dictionary
            metrics: Overall metrics
            debug_filename: Optional custom filename (absolute path or relative to predictions_dir)
        """
        # Add statistics to debug_info
        debug_info["statistics"] = {
            "total_samples": metrics.get("total_samples", 0),
            "passed_samples_count": len(debug_info.get("passed_samples", [])),
            "failed_samples_count": len(debug_info.get("failed_samples", [])),
            "no_generation_samples_count": len(debug_info.get("no_generation_samples", [])),
        }

        # Add metrics
        debug_info["metrics"] = metrics

        # Use default filename if not specified
        if debug_filename is None:
            debug_filename = "debug.json"

        # Save debug info to file using base class method
        self._save_debug_json(debug_info, filename=debug_filename)

        console.print(f"Total samples: {metrics['total_samples']}")
        console.print(f"Passed samples: {len(debug_info['passed_samples'])}")
        console.print(f"Failed samples: {len(debug_info['failed_samples'])}")
        console.print(f"No generation samples: {len(debug_info['no_generation_samples'])}")

        # Print metrics
        console.print("\n[bold]Metrics:[/bold]")
        for metric_name, metric_value in metrics.items():
            if metric_name != "total_samples":
                console.print(f"  {metric_name}: {metric_value:.4f}")

        # Show some failed examples
        if debug_info["failed_samples"]:
            console.print(f"\n[yellow]Failed sample examples (first 3):[/yellow]")
            for i, item in enumerate(debug_info["failed_samples"][:3]):
                console.print(f"  Example {i+1}:")
                console.print(f"    Sample ID: {item['sample_id']}")
                # Handle both SID and PID modes
                if 'ground_truth_sids' in item:
                    console.print(f"    Ground truth SIDs: {item['ground_truth_sids']}")
                elif 'ground_truth_pids' in item:
                    console.print(f"    Ground truth PIDs: {item['ground_truth_pids']}")
                console.print(f"    Top 5 generations: {item['top_10_generations'][:5]}")
                console.print()
