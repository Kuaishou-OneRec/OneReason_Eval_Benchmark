"""End2End visual model generator for OneReason Eval Benchmark."""

import os
import time
from typing import Dict, Optional

import torch
from tqdm import tqdm

from benchmark.base_generator import Generator
from benchmark.console import *


class End2EndGenerator(Generator):
    """End2End visual model generator (Encoder + Projection + Decoder).

    Wraps End2EndInferencer from custom_models/end2end/inferencer.py and
    adapts it to the OneReason Eval Benchmark Generator interface.
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        projection_path: str,
        model_name: str = "end2end",
        num_query_tokens: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: Optional[str] = None,
        use_flash_attn: bool = True,
        nframe: int = 64,
        fps: Optional[float] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name

        from custom_models.end2end.inferencer import End2EndInferencer

        console.print(f"[bold]Initializing End2EndGenerator: {model_name}[/bold]")
        self.inferencer = End2EndInferencer(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            projection_path=projection_path,
            num_query_tokens=num_query_tokens,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            do_sample=do_sample,
            system_prompt=system_prompt,
            use_flash_attn=use_flash_attn,
            fps=fps,
            nframe=nframe,
            device=device,
        )
        self.num_params = self.inferencer.count_parameters()
        console.print(
            f"Model parameters: {self.num_params / 1e9:.2f}B",
            style=success_style,
        )

    def _generate_standard(self, prompts: Dict[str, str], **kwargs) -> tuple:
        """Generate captions from video inputs.

        Prompts are formatted strings from apply_chat_template (containing <|encoder_pad|>).
        Video paths are in metadata[sample_id]["video_path"] (passed via kwargs).
        """
        metadata = kwargs.get("metadata", getattr(self, "_current_metadata", {}))
        results = {}
        logprobs = {}
        mfu_stats = {}

        sample_ids = list(prompts.keys())

        for sample_id in tqdm(sample_ids, desc="End2End Generating"):
            sample_meta = metadata.get(sample_id, {})
            video_path = sample_meta.get("video_path", "")
            formatted_prompt = prompts[sample_id]

            if not video_path or not os.path.exists(video_path):
                console.print(
                    f"[yellow]Sample {sample_id}: video not found: {video_path}, skipping[/yellow]"
                )
                results[sample_id] = [""]
                mfu_stats[sample_id] = {
                    "input_tokens": [0],
                    "output_tokens": [0],
                    "times": [0.0],
                }
                continue

            t0 = time.time()
            try:
                generated_text, n_input, n_output = self.inferencer.generate(
                    video_path=video_path,
                    formatted_prompt=formatted_prompt,
                )
            except Exception as e:
                console.print(
                    f"[yellow]Sample {sample_id}: inference failed ({type(e).__name__}: {e}), skipping[/yellow]"
                )
                results[sample_id] = [""]
                mfu_stats[sample_id] = {
                    "input_tokens": [0],
                    "output_tokens": [0],
                    "times": [0.0],
                }
                continue
            elapsed = time.time() - t0

            results[sample_id] = [generated_text]
            mfu_stats[sample_id] = {
                "input_tokens": [n_input],
                "output_tokens": [n_output],
                "times": [elapsed],
            }

        return results, logprobs, mfu_stats
