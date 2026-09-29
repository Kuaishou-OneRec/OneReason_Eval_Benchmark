import os
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
import math
import json

from benchmark.base_generator import Generator, VllmMixin, DISABLE_OPTIMIZATIONS_FOR_TASKS
from benchmark.console import *
from benchmark.checkpoint_utils import export_pt_to_safetensor




class VllmGenerator(VllmMixin, Generator):
    
    def __init__(
        self,
        model_name_or_path: str,
        checkpoint_path: Optional[str] = None,
        num_return_sequences: int = 2,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        do_sample: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        task_types: Optional[List[str]] = None,
        force_enable_optimizations: bool = False,
        force_disable_optimizations: bool = False,
        worker_batch_size: int = 8,
        seed: Optional[int] = 42,
        max_logprobs: int = 384,
        **kwargs
    ):
        """
        Args:
            model_name_or_path: Model name or path (e.g., "Qwen/Qwen2-7B")
            checkpoint_path: PT checkpoint path (optional). If provided, will be converted to HuggingFace format first
            num_return_sequences: Number of candidate sequences per prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            presence_penalty: Presence penalty (penalizes tokens that appeared in the text)
            frequency_penalty: Frequency penalty (penalizes tokens based on frequency)
            do_sample: Whether to sample (if False, use greedy decoding)
            tensor_parallel_size: Tensor parallel size (multi-GPU inference)
            gpu_memory_utilization: GPU memory utilization (0-1)
            trust_remote_code: Whether to trust remote code
            dtype: Model data type ("auto", "half", "float16", "bfloat16", "float", "float32")
            max_model_len: Maximum model length (optional, for limiting context length)
            task_types: List of task types to evaluate (for auto optimization control)
            force_enable_optimizations: Force enable optimizations for all tasks
            force_disable_optimizations: Force disable optimizations for all tasks
            worker_batch_size: Batch size for generation (to avoid vLLM scheduler issues)
            **kwargs: Other vLLM parameters
        """
        super().__init__(
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            do_sample=do_sample,
            **kwargs
        )
        
        self.model_name = model_name_or_path
        self.checkpoint_path = checkpoint_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.task_types = task_types or []
        self.force_enable_optimizations = force_enable_optimizations
        self.force_disable_optimizations = force_disable_optimizations
        self.worker_batch_size = worker_batch_size
        self.max_logprobs = max_logprobs

        console.print(
            "\nLoading Model\n",
            style=head_style_2,
            justify="center",
        )
        console.print(
            f"  Using vLLM to load model: [cyan]{model_name_or_path}[/cyan]",
            style=subhead_style_2,
        )
        
        # Handle PT checkpoint
        if checkpoint_path:
            console.print(
                f"  checkpoint: [yellow]{checkpoint_path}[/yellow]",
                style=subhead_style_2,
            )
            console.print(
                "  [yellow]need to convert PT checkpoint to HuggingFace format...[/yellow]",
                style=subhead_style_2,
            )
            model_path = export_pt_to_safetensor(
                config_path=model_name_or_path,
                checkpoint_path=checkpoint_path,
                trust_remote_code=trust_remote_code
            )
        else:
            model_path = model_name_or_path
        
        console.print(
            f"  tensor parallel size: [green]{tensor_parallel_size}[/green]",
            style=subhead_style_2,
        )
        console.print(
            f"  gpu memory utilization: [green]{gpu_memory_utilization}[/green]",
            style=subhead_style_2,
        )
        console.print(
            f"  dtype: [green]{dtype}[/green]",
            style=subhead_style_2,
        )

        # Check if any task in task_types requires disabling optimizations
        enable_optimizations = self._should_enable_optimizations()
        opt_status = "optimized" if enable_optimizations else "standard"

        console.print(
            f"  initializing vLLM engine ({opt_status})...",
            style=subhead_style_2,
        )
        
        vllm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "enable_chunked_prefill": enable_optimizations,
            "enable_prefix_caching": enable_optimizations,
            "max_logprobs": max_logprobs,  # Support beam search, need large enough logprobs
            "seed": seed if seed is not None else 0,
        }
        if os.environ.get("ONEREC_VLLM_SID_INTERNAL_BEAM") == "1":
            vllm_kwargs["async_scheduling"] = False
        
        if max_model_len is not None:
            vllm_kwargs["max_model_len"] = max_model_len

        vllm_kwargs.update(kwargs)
        
        try:
            import time
            start_time = time.time()
            self.llm = LLM(**vllm_kwargs)
            self.tokenizer = self.llm.get_tokenizer()
            elapsed_time = time.time() - start_time
            console.print(
                f"✓ vLLM engine initialized successfully ({elapsed_time:.1f}s)\n",
                style=success_style,
            )
        except Exception as e:
            console.print(
                f"✗ vLLM initialization failed: {e}",
                style=err_style,
            )
            raise

        self.num_params = self._count_model_parameters()

        if force_enable_optimizations:
            console.print(
                "⚙️ Optimization mode: FORCED ENABLED (chunked_prefill & prefix_caching enabled for all tasks)",
                style=warning_style,
            )
        elif force_disable_optimizations:
            console.print(
                "⚙️ Optimization mode: FORCED DISABLED (chunked_prefill & prefix_caching disabled for all tasks)",
                style=warning_style,
            )
        else:
            console.print(
                f"⚙️ Optimization mode: AUTO (will disable for tasks: {DISABLE_OPTIMIZATIONS_FOR_TASKS})",
                style=warning_style,
            )

    def _count_model_parameters(self) -> Optional[float]:
        tensor_parallel_size = getattr(self, 'tensor_parallel_size', 1)
        if tensor_parallel_size > 1:
            console.print(
                f"Warning: Tensor parallel (size={tensor_parallel_size}) detected. "
                f"Skipping parameter count (would only count local shard).",
                style=warning_style,
            )
            return None
        
        try:
            model = None
            engine = self.llm.llm_engine

            # Try multiple access paths for different vLLM versions
            if hasattr(engine, 'model_executor'):
                model_executor = engine.model_executor
                if hasattr(model_executor, 'driver_worker'):
                    model = model_executor.driver_worker.model_runner.model
                elif hasattr(model_executor, 'model'):
                    model = model_executor.model

            if model is None:
                console.print(
                    f"Warning: Cannot access model for parameter counting in this vLLM version",
                    style=warning_style,
                )
                return None

            total_params = sum(p.numel() for p in model.parameters())
            console.print(
                f"✓ Model parameters: {total_params / 1e9:.2f}B\n",
                style=success_style,
            )
            return float(total_params)
        except Exception as e:
            console.print(
                f"Warning: Failed to count parameters: {e}",
                style=warning_style,
            )
            return None



    def _generate_standard(
        self,
        prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Standard single-stage generation

        Args:
            prompts: {sample_id: prompt_text}
            **kwargs: Optional generation parameters

        Returns:
            Tuple of three values:
            - First dict: {sample_id: [generated_text_1, generated_text_2, ...]}
            - Second dict: {sample_id: [cum_logprob_1, cum_logprob_2, ...]} (only for beam search)
            - Third dict: {sample_id: {"input_tokens": [int], "output_tokens": [int], "times": [float]}} (lists for multi-stage support)
        """
        import time
        stage_start_time = time.time()

        sampling_params = self._build_sampling_params(**kwargs)
        
        console.print(
            f"Starting generation...",
            style=subhead_style_2,
        )

        if isinstance(sampling_params, BeamSearchParams):
            console.print(
                f"Sampling parameters (beam search): beam_width={sampling_params.beam_width}, "
                f"max_tokens={sampling_params.max_tokens}",
                style=subhead_style_2,
            )
        else:
            console.print(
                f"Sampling parameters: n={sampling_params.n}, max_tokens={sampling_params.max_tokens}, "
                f"temperature={sampling_params.temperature}, top_p={sampling_params.top_p}, top_k={sampling_params.top_k}, "
                f"repetition_penalty={sampling_params.repetition_penalty}, "
                f"presence_penalty={sampling_params.presence_penalty}, "
                f"frequency_penalty={sampling_params.frequency_penalty}",
                style=subhead_style_2,
            )
        console.print(
            f"Worker batch size: {self.worker_batch_size}",
            style=subhead_style_2,
        )

        sample_ids = list(prompts.keys())
        prompt_texts = list(prompts.values())

        # Batch processing to avoid vLLM scheduler issues
        results = {}
        logprobs = {}
        mfu_stats = {}
        num_batches = (len(sample_ids) + self.worker_batch_size - 1) // self.worker_batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.worker_batch_size
            end_idx = min(start_idx + self.worker_batch_size, len(sample_ids))

            batch_sample_ids = sample_ids[start_idx:end_idx]
            batch_prompt_texts = prompt_texts[start_idx:end_idx]

            try:
                # Calculate prompt lengths for this batch
                batch_prompt_lengths = []
                for text in batch_prompt_texts:
                    prompt_tokens = self.tokenizer.encode(text, add_special_tokens=True)
                    batch_prompt_lengths.append(len(prompt_tokens))

                # Generate based on parameter type
                if isinstance(sampling_params, BeamSearchParams):
                    # Beam search needs to use beam_search method
                    # beam_search input format is [{"prompt": "text"}]
                    batch_prompt_dicts = [{"prompt": text} for text in batch_prompt_texts]
                    batch_outputs = self.llm.beam_search(batch_prompt_dicts, sampling_params)
                else:
                    batch_outputs = self.llm.generate(batch_prompt_texts, sampling_params)

                # Process results for this batch
                for idx, (sample_id, output) in enumerate(zip(batch_sample_ids, batch_outputs)):
                    input_tokens = batch_prompt_lengths[idx]

                    if isinstance(sampling_params, BeamSearchParams):
                        # Beam search returns complete token IDs (including prompt), need to remove prompt part before decoding
                        prompt_length = batch_prompt_lengths[idx]
                        generated_texts = [
                            self.tokenizer.decode(seq.tokens[prompt_length:], skip_special_tokens=True)
                            for seq in output.sequences
                        ]
                        cum_logprobs = [seq.cum_logprob for seq in output.sequences]
                        results[sample_id] = generated_texts
                        logprobs[sample_id] = cum_logprobs

                        output_tokens_list = [len(seq.tokens) - prompt_length for seq in output.sequences]
                    else:
                        generated_texts = [out.text for out in output.outputs]
                        results[sample_id] = generated_texts

                        output_tokens_list = [
                            len(self.tokenizer.encode(text, add_special_tokens=False))
                            for text in generated_texts
                        ]

                    # Collect MFU stats for this sample (already in list format for multi-stage support)
                    mfu_stats[sample_id] = {
                        "input_tokens": [input_tokens],
                        "output_tokens": [sum(output_tokens_list)]
                    }

            except Exception as e:
                # When a single batch fails, return empty string and print detailed error
                import traceback
                print(f"\nBatch {batch_idx}/{num_batches} generation failed:")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                print(f"  Batch size: {len(batch_sample_ids)}")
                if batch_prompt_texts:
                    prompt_lens = [len(self.tokenizer.encode(t, add_special_tokens=True)) for t in batch_prompt_texts]
                    print(f"  Prompt token length range: min={min(prompt_lens)}, max={max(prompt_lens)}, avg={sum(prompt_lens)/len(prompt_lens):.1f}")
                print(f"  Full stack trace:\n{traceback.format_exc()}")

                num_return = sampling_params.n if not isinstance(sampling_params, BeamSearchParams) else sampling_params.beam_width
                for sample_id in batch_sample_ids:
                    results[sample_id] = [""] * num_return
                    if isinstance(sampling_params, BeamSearchParams):
                        logprobs[sample_id] = [0.0] * num_return
                    # Don't include failed samples in MFU stats

        # Calculate stage time
        stage_elapsed_time = time.time() - stage_start_time

        # Add time to all samples (same time for all samples in a batch)
        for sample_id in mfu_stats:
            mfu_stats[sample_id]["times"] = [stage_elapsed_time]

        console.print(
            f"✓ Generation completed (stage time: {stage_elapsed_time:.2f}s)",
            style=success_style,
        )

        return results, logprobs, mfu_stats

    
    def extract_token_logprobs(
        self,
        prompts: Dict[str, str],
        target_tokens: List[str],
        **kwargs
    ) -> tuple:
        """
        Extract logprobs for specific target tokens

        Args:
            prompts: {sample_id: prompt_text}
            target_tokens: List of target tokens (e.g., ["是", "否"])
            **kwargs: Optional parameters

        Returns:
            Tuple of three dicts:
            - First dict: {sample_id: [json_string]} where json_string is formatted probabilities
            - Second dict: {} (empty, no beam search logprobs for classification)
            - Third dict: {sample_id: {"input_tokens": [int], "output_tokens": [int], "times": [float]}}
        """
        import time
        stage_start_time = time.time()

        console.print(
            f"Extracting logprobs for tokens: {target_tokens}",
            style=subhead_style_2,
        )
        console.print(
            f"Worker batch size: {self.worker_batch_size}",
            style=subhead_style_2,
        )

        target_token_ids = {}
        for token in target_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1:
                target_token_ids[token] = token_ids[0]
            else:
                console.print(
                    f"⚠ Warning: Token '{token}' is encoded as multiple tokens: {token_ids}, using first token only",
                    style=warning_style
                )
                # For multi-token case, we only use the first token for now
                target_token_ids[token] = token_ids[0]
        
        console.print(
            f"Token IDs: {target_token_ids}",
            style=subhead_style_2,
        )

        sampling_params = SamplingParams(
            n=kwargs.get("num_return_sequences", 1),
            max_tokens=kwargs.get("max_new_tokens", 1),
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            logprobs=kwargs.get("logprobs", 10),
        )

        sample_ids = list(prompts.keys())
        prompt_texts = list(prompts.values())

        outputs = self.llm.generate(prompt_texts, sampling_params)

        # Extract logprobs for target tokens
        results = {}
        mfu_stats = {}
        for idx, (sample_id, output) in enumerate(zip(sample_ids, outputs)):
            token_probs = {}

            # Get logprobs from the first generated token
            if output.outputs and len(output.outputs) > 0:
                first_output = output.outputs[0]
                if first_output.logprobs and len(first_output.logprobs) > 0:
                    # Get logprobs dict for the first token
                    first_token_logprobs = first_output.logprobs[0]

                    # Extract probabilities for target tokens
                    for token, token_id in target_token_ids.items():
                        if token_id in first_token_logprobs:
                            logprob = first_token_logprobs[token_id].logprob
                            prob = math.exp(logprob)
                            token_probs[token] = prob
                        else:
                            # Token not in top-k, assign very small probability
                            token_probs[token] = 1e-10

            results[sample_id] = [json.dumps(token_probs, ensure_ascii=False)]

            prompt_text = prompt_texts[idx]
            input_tokens = len(self.tokenizer.encode(prompt_text, add_special_tokens=True))
            # Classification only generates 1 token
            output_tokens = 1
            mfu_stats[sample_id] = {
                "input_tokens": [input_tokens],
                "output_tokens": [output_tokens]
            }

        stage_elapsed_time = time.time() - stage_start_time
        for sample_id in mfu_stats:
            mfu_stats[sample_id]["times"] = [stage_elapsed_time]

        console.print(
            f"✓ Logprobs extraction completed",
            style=success_style,
        )

        return (results, {}, mfu_stats)
