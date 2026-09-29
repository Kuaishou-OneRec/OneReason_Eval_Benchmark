import torch.nn.functional as F
import torch
from typing import Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import math
import json

from benchmark.base_generator import Generator, HfTransformersMixin
from benchmark.checkpoint_utils import load_weights_from_pt
from benchmark.console import *


def _register_custom_models():
    """
    Register custom Qwen3 models from custom_models/qwen3_baggle.
    
    This allows the HfTransformersGenerator to use custom Qwen3 implementations
    that support additional parameters like item_token_start/end while keeping
    model_type = "qwen3" in config.json.
    """
    try:
        from custom_models.qwen3_baggle import Qwen3BaggleConfig, Qwen3ForCausalLM
        
        # Register the custom config and model
        # This overrides the default Qwen3 implementation from transformers
        AutoConfig.register("qwen3", Qwen3BaggleConfig, exist_ok=True)
        AutoModelForCausalLM.register(Qwen3BaggleConfig, Qwen3ForCausalLM, exist_ok=True)
        
        console.print(
            "✓ Custom Qwen3 models registered (with item_token support)",
            style="green"
        )
        return True
    except ImportError as e:
        # If custom models are not available, use default transformers implementation
        console.print(
            f"⚠ Custom Qwen3 models not found, using default transformers implementation: {e}",
            style="yellow"
        )
        return False
    except Exception as e:
        console.print(
            f"⚠ Failed to register custom Qwen3 models: {e}",
            style="yellow"
        )
        return False


class HfTransformersGenerator(HfTransformersMixin, Generator):
    """
    HuggingFace Transformers General Generator
    
    Supports all HuggingFace Transformers causal language models, including but not limited to:
    - Qwen, Llama, GPT, GLM, Mistral, Phi, etc.
    """
    
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
        device: Optional[str] = None,
        batch_size: int = 1,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        task_types: Optional[List[str]] = None,
        worker_batch_size: int = 8,
        **kwargs
    ):
        """
        Args:
            model_name_or_path: Model name or path (e.g., "Qwen/Qwen2-7B")
            checkpoint_path: PT checkpoint path (optional). If provided, will load model from config + checkpoint
            num_return_sequences: Number of candidate sequences per prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            presence_penalty: Presence penalty (penalizes tokens that appeared in the text)
            frequency_penalty: Frequency penalty (penalizes tokens based on frequency)
            do_sample: Whether to sample
            device: Device ("cuda" or "cpu"), default auto-detect
            batch_size: Batch size
            dtype: Model data type (default float16 for cuda, float32 for cpu)
            trust_remote_code: Whether to trust remote code
            task_types: List of task types to evaluate (for potential future optimizations)
            worker_batch_size: Batch size for generation
            **kwargs: Other generation parameters
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.task_types = task_types or []
        self.worker_batch_size = worker_batch_size

        # Set data type
        if dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = dtype
        
        # Register custom Qwen3 models if available
        _register_custom_models()
        
        console.print(
            "\nLoading Model\n",
            style=head_style_2,
            justify="center",
        )
        console.print(
            f"  Loading model: [cyan]{model_name_or_path}[/cyan]",
            style=subhead_style_2,
        )
        if checkpoint_path:
            console.print(
                f"  checkpoint: [yellow]{checkpoint_path}[/yellow]",
                style=subhead_style_2,
            )
        console.print(
            f"  device: [green]{self.device}[/green]",
            style=subhead_style_2,
        )
        console.print(
            f"  dtype: [green]{self.torch_dtype}[/green]",
            style=subhead_style_2,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )
        
        # Choose loading method based on whether checkpoint_path is provided
        if checkpoint_path is not None:
            console.print(
                "  Loading from PT checkpoint...",
                style=subhead_style_2,
            )
            from benchmark.checkpoint_utils import build_model_from_pt
            
            self.model = build_model_from_pt(
                config_path=model_name_or_path,
                checkpoint_path=checkpoint_path,
                device=self.device,
                torch_dtype=self.torch_dtype,
                trust_remote_code=trust_remote_code
            )
        else:
            console.print(
                "  Loading from pretrained model...",
                style=subhead_style_2,
            )
            from benchmark.checkpoint_utils import build_model_from_hf
            
            self.model = build_model_from_hf(
                model_name_or_path=model_name_or_path,
                device=self.device,
                torch_dtype=self.torch_dtype,
                trust_remote_code=trust_remote_code,
                use_device_map=True
            )
        
        self.model.eval()
        console.print(
            f"✓ Model loaded successfully, batch size: {self.batch_size}",
            style=success_style,
        )

        self.num_params = self._count_model_parameters()

    
    def _count_model_parameters(self) -> Optional[float]:
        tensor_parallel_size = getattr(self, 'tensor_parallel_size', 1)
        if tensor_parallel_size > 1:
            console.print(
                f"Warning: Tensor parallel (size={tensor_parallel_size}) detected. "
                f"Skipping parameter count (would only count local shard).",
                style=warning_style,
            )
            return None
        
        if hasattr(self, 'model') and self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            console.print(
                f"✓ Model parameters: {total_params / 1e9:.2f}B\n",
                style=success_style,
            )
            return float(total_params)
        else:
            return None

    def _generate_standard(
        self,
        prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Standard single-stage batch text generation

        Args:
            prompts: {sample_id: prompt_text}
            **kwargs: Optional parameters

        Returns:
            Tuple of three dicts:
            - First dict: {sample_id: [generated_text_1, generated_text_2, ...]}
            - Second dict: {sample_id: [score_1, score_2, ...]} (only for beam search)
            - Third dict: {sample_id: {"input_tokens": [int], "output_tokens": [int], "times": [float]}} (lists for multi-stage support)
        """
        import time
        stage_start_time = time.time()

        # Build sampling parameters using helper method
        gen_kwargs, stop_sequences = self._build_sampling_params(**kwargs)
        num_beams = kwargs.get("num_beams", 1)

        console.print(
            f"Starting generation...",
            style=subhead_style_2,
        )
        console.print(
            f"Worker batch size: {self.worker_batch_size}",
            style=subhead_style_2,
        )
        if num_beams > 1:
            console.print(
                f"Sampling parameters (beam search): num_beams={num_beams}, num_return_sequences={gen_kwargs['num_return_sequences']}",
                f"max_new_tokens={gen_kwargs['max_new_tokens']}",
                style=subhead_style_2,
            )
        else:
            console.print(
                f"Sampling parameters: num_return_sequences={gen_kwargs['num_return_sequences']}, max_new_tokens={gen_kwargs['max_new_tokens']}, "
                f"temperature={gen_kwargs['temperature']}, top_p={gen_kwargs['top_p']}, top_k={gen_kwargs['top_k']}, "
                f"repetition_penalty={gen_kwargs['repetition_penalty']}, "
                f"presence_penalty={gen_kwargs['presence_penalty']}, "
                f"frequency_penalty={gen_kwargs['frequency_penalty']}",
                style=subhead_style_2,
            )
        
        results = {}
        logprobs = {}
        mfu_stats = {}
        sample_ids = list(prompts.keys())

        # Batch processing
        with torch.no_grad():
            for i in tqdm(range(0, len(sample_ids), self.worker_batch_size), desc="Generating"):
                batch_ids = sample_ids[i:i + self.worker_batch_size]
                batch_prompts = [prompts[id] for id in batch_ids]

                # Tokenize
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)

                # Generate
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                # Decode
                for j, sample_id in enumerate(batch_ids):
                    # Each sample generates num_return_sequences results
                    start_idx = j * gen_kwargs["num_return_sequences"]
                    end_idx = start_idx + gen_kwargs["num_return_sequences"]

                    generated_texts = []
                    output_tokens_list = []

                    if num_beams > 1:
                        # Beam search: outputs is a dict with 'sequences' and 'sequences_scores'
                        sequences = outputs.sequences[start_idx:end_idx]
                        scores = outputs.sequences_scores[start_idx:end_idx].cpu().tolist()

                        for seq in sequences:
                            # Keep only the generated part (remove prompt)
                            prompt_length = inputs.input_ids[j].shape[0]
                            generated = self.tokenizer.decode(
                                seq[prompt_length:],
                                skip_special_tokens=True
                            )
                            # Truncate at stop sequences if specified
                            if stop_sequences:
                                for stop_seq in stop_sequences:
                                    if stop_seq in generated:
                                        generated = generated[:generated.index(stop_seq)]
                                        break
                            generated_texts.append(generated)
                            output_tokens_list.append(len(seq) - prompt_length)

                        results[sample_id] = generated_texts
                        logprobs[sample_id] = scores
                    else:
                        # Sampling: outputs is just sequences
                        prompt_length = inputs.input_ids[j].shape[0]
                        for output in outputs[start_idx:end_idx]:
                            # Keep only the generated part (remove prompt)
                            generated = self.tokenizer.decode(
                                output[prompt_length:],
                                skip_special_tokens=True
                            )
                            # Truncate at stop sequences if specified
                            if stop_sequences:
                                for stop_seq in stop_sequences:
                                    if stop_seq in generated:
                                        generated = generated[:generated.index(stop_seq)]
                                        break
                            generated_texts.append(generated)
                            output_tokens_list.append(len(output) - prompt_length)

                        results[sample_id] = generated_texts

                    # Collect MFU stats
                    input_tokens = inputs.input_ids[j].shape[0]
                    mfu_stats[sample_id] = {
                        "input_tokens": [input_tokens],
                        "output_tokens": [sum(output_tokens_list)]
                    }

        # Calculate stage time
        stage_elapsed_time = time.time() - stage_start_time

        # Add time to all samples (same time for all samples in a batch)
        for sample_id in mfu_stats:
            mfu_stats[sample_id]["times"] = [stage_elapsed_time]

        console.print(
            f"✓ Generation completed (stage time: {stage_elapsed_time:.2f}s)",
            style=success_style,
        )

        return (results, logprobs, mfu_stats)

    
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

        # Get token IDs for target tokens
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

        results = {}
        mfu_stats = {}
        sample_ids = list(prompts.keys())
        
        # Batch processing
        with torch.no_grad():
            for i in tqdm(range(0, len(sample_ids), self.batch_size), desc="Extracting logprobs"):
                batch_ids = sample_ids[i:i + self.batch_size]
                batch_prompts = [prompts[id] for id in batch_ids]

                # Tokenize
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)

                # Forward pass to get logits
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                )

                # Get logits for the last token position (next token prediction)
                # Shape: [batch_size, vocab_size]
                last_token_logits = outputs.logits[:, -1, :]

                # Convert logits to log probabilities
                log_probs = F.log_softmax(last_token_logits, dim=-1)

                # Extract probabilities for target tokens
                for j, sample_id in enumerate(batch_ids):
                    token_probs = {}
                    for token, token_id in target_token_ids.items():
                        log_prob = log_probs[j, token_id].item()
                        prob = math.exp(log_prob)
                        token_probs[token] = prob
                    results[sample_id] = [json.dumps(token_probs, ensure_ascii=False)]
                    prompt_text = prompts[sample_id]
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


