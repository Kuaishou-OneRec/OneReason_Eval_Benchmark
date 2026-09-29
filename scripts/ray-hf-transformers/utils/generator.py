import os
import torch
import ray
import math
import json
import hashlib
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

from benchmark.base_generator import Generator, RayMixin, HfTransformersMixin, DISABLE_OPTIMIZATIONS_FOR_TASKS
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
        
        print("✓ Custom Qwen3 models registered (with item_token support)")
        return True
    except ImportError as e:
        # If custom models are not available, use default transformers implementation
        print(f"⚠ Custom Qwen3 models not found, using default transformers implementation: {e}")
        return False
    except Exception as e:
        print(f"⚠ Failed to register custom Qwen3 models: {e}")
        return False


class HfTransformersWorker:
    """
    HuggingFace Transformers Worker that uses one or more GPUs
    
    Each Worker is responsible for:
    - Loading one model instance (potentially across multiple GPUs with tensor parallelism)
    - Processing inference tasks assigned to it
    - Returning generation results
    """
    
    def __init__(
        self,
        worker_id: int,
        model_path: str,
        gpu_ids: List[int],
        trust_remote_code: bool = False,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            worker_id: Worker ID
            model_path: Model path (HuggingFace format or local path)
            gpu_ids: List of GPU IDs assigned to this worker
            trust_remote_code: Whether to trust remote code
            dtype: Data type
            tensor_parallel_size: Tensor parallel size (must match len(gpu_ids))
            checkpoint_path: PT checkpoint path (optional)
            **kwargs: Other parameters
        """
        self.worker_id = worker_id
        self.gpu_ids = gpu_ids
        self.checkpoint_path = checkpoint_path
        self.tensor_parallel_size = tensor_parallel_size
        
        # Print debug info BEFORE setting CUDA_VISIBLE_DEVICES
        import socket
        hostname = socket.gethostname()
        print(f"  [Worker {worker_id}] Starting on host: {hostname}")
        print(f"  [Worker {worker_id}] Assigned GPU IDs: {gpu_ids}")
        print(f"  [Worker {worker_id}] Current CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        
        # Map dtype string to torch dtype
        dtype_map = {
            'auto': None,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        self.torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        # Determine device and set CUDA_VISIBLE_DEVICES
        if tensor_parallel_size == 1:
            # Single GPU mode: Let Ray handle GPU assignment
            # Ray already sets CUDA_VISIBLE_DEVICES for us based on num_gpus resource
            # We just use cuda:0 which maps to the GPU Ray assigned
            self.device = "cuda:0"
            print(f"  [Worker {worker_id}] Using Ray-managed GPU assignment (TP=1)")
        else:
            # Multiple GPUs for tensor parallelism
            # In this case, we need to explicitly set CUDA_VISIBLE_DEVICES
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            self.device = "cuda"
            print(f"  [Worker {worker_id}] Manually set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (TP={tensor_parallel_size})")
        
        gpu_str = ",".join(map(str, gpu_ids))
        print(f"  [Worker {worker_id}] Initializing... (GPU {gpu_str}, TP={tensor_parallel_size})")
        
        # Print available GPU memory
        if torch.cuda.is_available():
            print(f"  [Worker {worker_id}] CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  [Worker {worker_id}] GPU {i}: {torch.cuda.get_device_name(i)}, Total memory: {mem_total:.2f} GB")
        
        # Register custom models if available
        _register_custom_models()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        
        # Load model
        print(f"  [Worker {worker_id}] Loading model from: {model_path}")
        if checkpoint_path:
            print(f"  [Worker {worker_id}] Using checkpoint: {checkpoint_path}")
            from benchmark.checkpoint_utils import build_model_from_pt
            
            # Determine target device based on tensor parallel size
            device = "cuda" if tensor_parallel_size > 1 else self.device
            
            self.model = build_model_from_pt(
                config_path=model_path,
                checkpoint_path=checkpoint_path,
                device=device,
                torch_dtype=self.torch_dtype,
                trust_remote_code=trust_remote_code
            )
        else:
            from benchmark.checkpoint_utils import build_model_from_hf
            
            # Determine if we should use device_map based on tensor parallel size
            use_device_map = tensor_parallel_size > 1
            device = "cuda" if tensor_parallel_size > 1 else self.device
            
            self.model = build_model_from_hf(
                model_name_or_path=model_path,
                device=device,
                torch_dtype=self.torch_dtype,
                trust_remote_code=trust_remote_code,
                use_device_map=use_device_map
            )
        
        self.model.eval()
        
        # Print GPU memory usage after model loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  [Worker {worker_id}] GPU {i} memory: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB, Total={mem_total:.2f}GB")
        
        print(f"  [Worker {worker_id}] ✓ Initialized successfully (GPU {gpu_str}, TP={tensor_parallel_size})")
    
    def get_model_parameters(self) -> Optional[float]:
        """
        Get model parameter count from the worker's model instance
        
        Returns:
            float: Total number of parameters, or None if unable to count
        """
        try:
            if hasattr(self, 'model') and self.model is not None:
                total_params = sum(p.numel() for p in self.model.parameters())
                return float(total_params)
            else:
                return None
        except Exception as e:
            print(f"  [Worker {self.worker_id}] Warning: Failed to count parameters: {e}")
            return None
    


    def generate_batch(
        self,
        prompts: Dict[str, str],
        generation_params: Dict[str, Any],
        worker_batch_size: int = 4
    ) -> tuple:
        """
        Batch text generation

        Args:
            prompts: {sample_id: prompt_text}
            generation_params: Generation parameter dictionary
            worker_batch_size: Worker internal batch size

        Returns:
            Tuple of three dicts:
            - First dict: {sample_id: [generated_text_1, generated_text_2, ...]}
            - Second dict: {sample_id: [score_1, score_2, ...]} (only for beam search)
            - Third dict: {sample_id: {"input_tokens": [int], "output_tokens": [int], "times": [float]}} (lists for multi-stage support)
        """
        import time
        stage_start_time = time.time()

        if not prompts:
            return ({}, {}, {})

        # Extract stop sequences (not supported by HF, will handle manually)
        stop_sequences = generation_params.get("stop", [])

        # Create a clean copy of generation_params without unsupported parameters
        gen_params_clean = generation_params.copy()
        gen_params_clean.pop("stop", None)

        # Prepare input
        sample_ids = list(prompts.keys())
        prompt_texts = list(prompts.values())

        # Batch processing
        all_results = {}
        all_logprobs = {}
        all_mfu_stats = {}
        num_batches = (len(sample_ids) + worker_batch_size - 1) // worker_batch_size
        num_beams = gen_params_clean.get("num_beams", 1)
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * worker_batch_size
                end_idx = min(start_idx + worker_batch_size, len(sample_ids))
                
                batch_sample_ids = sample_ids[start_idx:end_idx]
                batch_prompt_texts = prompt_texts[start_idx:end_idx]
                
                try:
                    # Tokenize
                    inputs = self.tokenizer(
                        batch_prompt_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.device)
                    
                    # Generate
                    outputs = self.model.generate(
                        **inputs,
                        **gen_params_clean,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                    # Decode
                    num_return_sequences = gen_params_clean.get("num_return_sequences", 1)
                    for j, sample_id in enumerate(batch_sample_ids):
                        # Each sample generates num_return_sequences results
                        start_idx_out = j * num_return_sequences
                        end_idx_out = start_idx_out + num_return_sequences

                        generated_texts = []
                        output_tokens_list = []
                        prompt_length = inputs.input_ids[j].shape[0]

                        if num_beams > 1:
                            # Beam search: outputs is a dict with 'sequences' and 'sequences_scores'
                            sequences = outputs.sequences[start_idx_out:end_idx_out]
                            scores = outputs.sequences_scores[start_idx_out:end_idx_out].cpu().tolist()

                            for seq in sequences:
                                # Keep only the generated part (remove prompt)
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

                            all_results[sample_id] = generated_texts
                            all_logprobs[sample_id] = scores
                        else:
                            # Sampling: outputs is just sequences
                            for output in outputs[start_idx_out:end_idx_out]:
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

                            all_results[sample_id] = generated_texts

                        # Collect MFU stats
                        all_mfu_stats[sample_id] = {
                            "input_tokens": [prompt_length],
                            "output_tokens": [sum(output_tokens_list)]
                        }

                except Exception as e:
                    # When a single batch fails, return empty string
                    import traceback
                    print(f"\n[Worker {self.worker_id}] Batch {batch_idx}/{num_batches} generation failed:")
                    print(f"  Error: {str(e)}")
                    print(f"  Full stack trace:\n{traceback.format_exc()}")

                    num_return = generation_params.get("num_return_sequences", 1)
                    for sample_id in batch_sample_ids:
                        all_results[sample_id] = [""] * num_return
                        if num_beams > 1:
                            all_logprobs[sample_id] = [0.0] * num_return
                        # Don't include failed samples in MFU stats (they would have times=[0.0] which breaks MFU calculation)

        # Calculate stage time
        stage_elapsed_time = time.time() - stage_start_time

        # Add time to all samples (same time for all samples in this worker)
        for sample_id in all_mfu_stats:
            all_mfu_stats[sample_id]["times"] = [stage_elapsed_time]

        return (all_results, all_logprobs, all_mfu_stats)
    
    def extract_token_logprobs_batch(
        self,
        prompts: Dict[str, str],
        target_tokens: List[str],
        worker_batch_size: int = 8
    ) -> tuple:
        """
        Extract logprobs for specific target tokens

        Args:
            prompts: {sample_id: prompt_text}
            target_tokens: List of target tokens
            worker_batch_size: Worker internal batch size

        Returns:
            Tuple of two dicts:
            - First dict: {sample_id: [json_string]} where json_string is formatted probabilities
            - Second dict: {sample_id: {"input_tokens": [int], "output_tokens": [int], "times": [float]}}
        """
        import time
        stage_start_time = time.time()

        if not prompts:
            return ({}, {})

        import torch.nn.functional as F
        
        # Get token IDs for target tokens
        target_token_ids = {}
        for token in target_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1:
                target_token_ids[token] = token_ids[0]
            else:
                print(f"  [Worker {self.worker_id}] Warning: Token '{token}' is encoded as multiple tokens: {token_ids}")
                target_token_ids[token] = token_ids[0]
        
        # Prepare input
        sample_ids = list(prompts.keys())
        prompt_texts = list(prompts.values())

        # Batch processing
        all_results = {}
        all_mfu_stats = {}
        num_batches = (len(sample_ids) + worker_batch_size - 1) // worker_batch_size

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * worker_batch_size
                end_idx = min(start_idx + worker_batch_size, len(sample_ids))

                batch_sample_ids = sample_ids[start_idx:end_idx]
                batch_prompt_texts = prompt_texts[start_idx:end_idx]

                try:
                    # Tokenize
                    inputs = self.tokenizer(
                        batch_prompt_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.device)

                    # Forward pass
                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                    )

                    # Get logits for the last token position
                    last_token_logits = outputs.logits[:, -1, :]

                    # Convert logits to log probabilities
                    log_probs = F.log_softmax(last_token_logits, dim=-1)

                    # Extract probabilities for target tokens
                    for j, sample_id in enumerate(batch_sample_ids):
                        token_probs = {}
                        for token, token_id in target_token_ids.items():
                            log_prob = log_probs[j, token_id].item()
                            prob = math.exp(log_prob)
                            token_probs[token] = prob
                        all_results[sample_id] = [json.dumps(token_probs, ensure_ascii=False)]

                        prompt_text = batch_prompt_texts[j]
                        input_tokens = len(self.tokenizer.encode(prompt_text, add_special_tokens=True))
                        # Classification only generates 1 token
                        output_tokens = 1
                        all_mfu_stats[sample_id] = {
                            "input_tokens": [input_tokens],
                            "output_tokens": [output_tokens]
                        }

                except Exception as e:
                    import traceback
                    print(f"\n[Worker {self.worker_id}] Batch {batch_idx}/{num_batches} logprobs extraction failed:")
                    print(f"  Error: {str(e)}")
                    print(f"  Full stack trace:\n{traceback.format_exc()}")

                    for sample_id in batch_sample_ids:
                        token_probs = {token: 0.0 for token in target_tokens}
                        all_results[sample_id] = [json.dumps(token_probs, ensure_ascii=False)]


        stage_elapsed_time = time.time() - stage_start_time

        for sample_id in all_mfu_stats:
            all_mfu_stats[sample_id]["times"] = [stage_elapsed_time]

        return (all_results, all_mfu_stats)


class RayHfTransformersGenerator(RayMixin, HfTransformersMixin, Generator):
    """
    Ray-based Multi-Node Multi-GPU HuggingFace Transformers Generator
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        checkpoint_path: Optional[str] = None,
        num_return_sequences: int = 2,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        do_sample: bool = True,
        trust_remote_code: bool = False,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        num_gpus: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
        task_types: Optional[List[str]] = None,
        worker_batch_size: int = 4,
        ray_address: Optional[str] = "auto",
        allow_cross_node_tensor_parallel: bool = False,
        **kwargs
    ):
        """
        Args:
            model_name_or_path: Model name or path
            checkpoint_path: PT checkpoint path (optional)
            num_return_sequences: Number of candidate sequences per prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            presence_penalty: Presence penalty (penalizes tokens that appeared in the text)
            frequency_penalty: Frequency penalty (penalizes tokens based on frequency)
            do_sample: Whether to sample
            trust_remote_code: Whether to trust remote code
            dtype: Model data type
            tensor_parallel_size: Tensor parallel size (default 1, single GPU per worker)
            num_gpus: Number of GPUs to use (default uses all cluster GPUs)
            gpu_ids: List of GPU IDs to use (only for single-node mode)
            task_types: List of task types to evaluate (for potential future optimizations)
            worker_batch_size: Batch size for each worker
            ray_address: Ray cluster address ('auto', 'local', or specific address)
            allow_cross_node_tensor_parallel: Allow tensor parallel across nodes (not recommended)
            **kwargs: Other parameters
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
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.worker_batch_size = worker_batch_size
        self.task_types = task_types or []
        self.ray_address = ray_address
        self.allow_cross_node_tensor_parallel = allow_cross_node_tensor_parallel
        self.num_gpus = num_gpus
        self.gpu_ids = gpu_ids
        self._temp_dir = None

        console.print(
            "\nLoading Model\n",
            style=head_style_2,
            justify="center",
        )
        console.print(
            f"  Using Ray + HuggingFace Transformers (Multi-Node) to load model: [cyan]{model_name_or_path}[/cyan]",
            style=subhead_style_2,
        )
        
        # 1. Initialize Ray cluster connection
        self._initialize_ray_cluster()
        
        # 2. Determine GPUs to use (from cluster)
        all_gpu_ids = self._determine_gpu_ids_from_cluster()
        
        # 3. Group GPUs for workers (ensuring same-node constraint if needed)
        self.worker_gpu_groups, self.worker_node_assignments = self._group_gpus_for_workers(
            all_gpu_ids, tensor_parallel_size
        )
        num_workers = len(self.worker_gpu_groups)
        
        # Display cluster and GPU information
        self._display_cluster_info(all_gpu_ids, num_workers)
        
        # 4. Handle PT checkpoint (convert if needed)
        model_path = model_name_or_path
        if checkpoint_path:
            console.print(
                f"  checkpoint: [yellow]{checkpoint_path}[/yellow]",
                style=subhead_style_2,
            )
            console.print(
                "  [yellow]Note: PT checkpoint will be loaded by each worker[/yellow]",
                style=subhead_style_2,
            )
        
        # 5. Create Workers
        console.print(
            f"  Creating {num_workers} HuggingFace Transformers Workers...",
            style=subhead_style_2,
        )
        self.workers = []
        
        worker_kwargs = {
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "checkpoint_path": checkpoint_path,
        }
        worker_kwargs.update(kwargs)
        
        # Create Ray remote class with dynamic GPU count and scheduling strategy
        for i, (gpu_group, node_id) in enumerate(zip(self.worker_gpu_groups, self.worker_node_assignments)):
            # Create worker with node placement constraint
            HfTransformersWorkerRemote = ray.remote(num_gpus=tensor_parallel_size)(HfTransformersWorker)
            
            # If we have node assignment, use scheduling strategy
            if node_id is not None:
                from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
                scheduling_strategy = NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False  # Hard constraint: must be on this node
                )
                worker = HfTransformersWorkerRemote.options(
                    scheduling_strategy=scheduling_strategy
                ).remote(
                    worker_id=i,
                    model_path=model_path,
                    gpu_ids=gpu_group,
                    **worker_kwargs
                )
            else:
                # No node constraint, let Ray decide
                worker = HfTransformersWorkerRemote.remote(
                    worker_id=i,
                    model_path=model_path,
                    gpu_ids=gpu_group,
                    **worker_kwargs
                )
            self.workers.append(worker)
        
        # Wait for all Workers to initialize
        console.print(
            "  Waiting for all Workers to initialize...\n\n",
            style=subhead_style_2,
        )
        ray.get([worker.generate_batch.remote({}, {}, 1) for worker in self.workers])
        
        console.print(
            f"✓ All Workers initialized successfully\n",
            style=success_style,
        )

        self.num_params = self._count_model_parameters()

    def _count_model_parameters(self) -> Optional[float]:
        """
        Override HfTransformersMixin._count_model_parameters() for Ray-based generators.
        
        In Ray-based architecture, model instances are in worker processes.
        Query the first worker to get model parameter count.
        
        Returns:
            float or None: Total number of parameters
        """
        tensor_parallel_size = getattr(self, 'tensor_parallel_size', 1)
        if tensor_parallel_size > 1:
            console.print(
                f"Warning: Tensor parallel (size={tensor_parallel_size}) detected. "
                f"Skipping parameter count (would only count local shard).",
                style=warning_style,
            )
            return None
        
        # Query the first worker to get parameter count
        try:
            import ray
            num_params = ray.get(self.workers[0].get_model_parameters.remote())
            if num_params:
                console.print(
                    f"✓ Model parameters: {num_params / 1e9:.2f}B\n",
                    style=success_style,
                )
            return num_params
        except Exception as e:
            console.print(
                f"Warning: Failed to get parameter count from worker: {e}",
                style=warning_style,
            )
            return None

    def _generate_standard(
        self,
        prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Standard single-stage generation (round-robin assignment to multiple Workers)

        Args:
            prompts: {sample_id: prompt_text}
            **kwargs: Optional generation parameters

        Returns:
            Tuple of three dicts:
            - First dict: {sample_id: [generated_text_1, generated_text_2, ...]}
            - Second dict: {sample_id: [score_1, score_2, ...]} (only for beam search)
            - Third dict: {sample_id: {"input_tokens": [int], "output_tokens": [int], "times": [float]}} (lists for multi-stage support)
        """
        # Build sampling parameters using helper method
        gen_kwargs, stop_sequences = self._build_sampling_params(**kwargs)
        num_beams = kwargs.get("num_beams", 1)

        console.print(
            f"Starting generation...",
            style=subhead_style_2,
        )
        if num_beams > 1:
            console.print(
                f"Gen arguments (beam search): num_beams={num_beams}, num_return_sequences={gen_kwargs['num_return_sequences']}, "
                f"max_new_tokens={gen_kwargs['max_new_tokens']}",
                style=subhead_style_2,
            )
        else:
            console.print(
                f"Gen arguments: num_return_sequences={gen_kwargs['num_return_sequences']}, max_new_tokens={gen_kwargs['max_new_tokens']}, "
                f"temperature={gen_kwargs['temperature']}, top_p={gen_kwargs['top_p']}, top_k={gen_kwargs['top_k']}, "
                f"repetition_penalty={gen_kwargs['repetition_penalty']}, "
                f"presence_penalty={gen_kwargs['presence_penalty']}, "
                f"frequency_penalty={gen_kwargs['frequency_penalty']}",
                style=subhead_style_2,
            )
        
        # Add stop sequences to gen_kwargs for workers to extract
        # Note: Cannot add in _build_sampling_params because HfTransformersGenerator
        # passes gen_kwargs directly to HuggingFace's generate() which doesn't support 'stop' parameter
        if stop_sequences:
            gen_kwargs["stop"] = stop_sequences
        
        # Round-robin assign tasks to Workers
        sample_ids = list(prompts.keys())
        num_workers = len(self.workers)
        worker_tasks = [dict() for _ in range(num_workers)]
        
        for i, sample_id in enumerate(sample_ids):
            worker_idx = i % num_workers
            worker_tasks[worker_idx][sample_id] = prompts[sample_id]
        
        console.print(
            f"Task distribution: {[len(task) for task in worker_tasks]}",
            style=subhead_style_2,
        )
        console.print(
            f"Worker batch size: {self.worker_batch_size}",
            style=subhead_style_2,
        )
        
        # Execute in parallel
        futures = []
        for worker, task in zip(self.workers, worker_tasks):
            if task:  # Only submit non-empty tasks
                future = worker.generate_batch.remote(task, gen_kwargs, self.worker_batch_size)
                futures.append(future)
        
        # Collect results
        worker_results = ray.get(futures)

        # Merge results (each worker_result is a tuple of (texts_dict, logprobs_dict, mfu_stats_dict))
        results = {}
        logprobs = {}
        mfu_stats = {}
        for worker_result in worker_results:
            texts_dict, logprobs_dict, mfu_stats_dict = worker_result
            results.update(texts_dict)
            logprobs.update(logprobs_dict)
            mfu_stats.update(mfu_stats_dict)

        console.print(
            f"✓ Generation completed",
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
        Extract logprobs for specific target tokens (round-robin assignment to multiple Workers)

        Args:
            prompts: {sample_id: prompt_text}
            target_tokens: List of target tokens
            **kwargs: Optional parameters

        Returns:
            Tuple of three dicts:
            - First dict: {sample_id: [json_string]} where json_string is formatted probabilities
            - Second dict: {} (empty, no beam search logprobs for classification)
            - Third dict: {sample_id: {"input_tokens": [int], "output_tokens": [int], "times": [float]}}
        """
        console.print(
            f"Extracting logprobs for tokens: {target_tokens}",
            style=subhead_style_2,
        )
        console.print(
            f"Worker batch size: {self.worker_batch_size}",
            style=subhead_style_2,
        )

        if not prompts:
            return ({}, {}, {})
        
        # Round-robin assign tasks to Workers
        sample_ids = list(prompts.keys())
        num_workers = len(self.workers)
        worker_tasks = [dict() for _ in range(num_workers)]
        
        for i, sample_id in enumerate(sample_ids):
            worker_idx = i % num_workers
            worker_tasks[worker_idx][sample_id] = prompts[sample_id]
        
        console.print(
            f"Task distribution: {[len(task) for task in worker_tasks]}",
            style=subhead_style_2,
        )

        # Get Worker internal batch size
        worker_batch_size = self.worker_batch_size

        # Execute in parallel
        futures = []
        for worker, task in zip(self.workers, worker_tasks):
            if task:  # Only submit non-empty tasks
                future = worker.extract_token_logprobs_batch.remote(
                    task, target_tokens, worker_batch_size
                )
                futures.append(future)
        
        # Collect results
        worker_results = ray.get(futures)

        # Merge results (each worker_result is a tuple of (probs_dict, mfu_stats_dict))
        results = {}
        mfu_stats = {}
        for worker_result in worker_results:
            probs_dict, mfu_stats_dict = worker_result
            results.update(probs_dict)
            mfu_stats.update(mfu_stats_dict)

        console.print(
            f"✓ Logprobs extraction completed",
            style=success_style,
        )

        return (results, {}, mfu_stats)
    