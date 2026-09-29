"""Qwen3.5-VL Encoder Processor for end-to-end models.

Wraps the standard Qwen3VLProcessor and appends learnable query placeholder
tokens (<|query_pad|>) after each sample.  At model forward time, these
placeholders are replaced with the encoder's learned query embeddings via
masked_scatter, which preserves gradient flow to the nn.Parameter.
"""

from __future__ import annotations

import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

from .processing_qwen3_vl import (
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)

QUERY_PAD_TOKEN = "<|query_pad|>"


class Qwen3_5EncoderProcessor(Qwen3VLProcessor):
    """Qwen3VLProcessor extended with query-pad placeholders for the encoder.

    After the base processor tokenizes text/image/video inputs, this
    processor appends ``num_query_tokens`` ``<|query_pad|>`` tokens to
    every sample.  In the encoder model, these positions are replaced
    by learnable query embeddings via ``masked_scatter``.
    """

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        num_query_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, video_processor, chat_template, **kwargs)
        self.num_query_tokens = num_query_tokens
        self._register_query_pad_token()

    def _register_query_pad_token(self):
        if QUERY_PAD_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [QUERY_PAD_TOKEN]}
            )
        self.query_pad_token_id = self.tokenizer.convert_tokens_to_ids(QUERY_PAD_TOKEN)

    @classmethod
    def from_base_processor(
        cls,
        base_processor: Qwen3VLProcessor,
        num_query_tokens: int = 256,
    ) -> "Qwen3_5EncoderProcessor":
        """Construct from an already-loaded Qwen3VLProcessor instance."""
        instance = cls.__new__(cls)
        instance.__dict__.update(base_processor.__dict__)
        instance.num_query_tokens = num_query_tokens
        instance._register_query_pad_token()
        return instance

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
        outputs = super().__call__(images=images, text=text, videos=videos, **kwargs)
        return self._append_query_tokens(outputs)

    def _append_query_tokens(self, outputs: BatchFeature) -> BatchFeature:
        """Append query_pad tokens to the end of each sample."""
        input_ids = outputs["input_ids"]

        if isinstance(input_ids, torch.Tensor):
            B = input_ids.shape[0]
            device, dtype = input_ids.device, input_ids.dtype
            query_ids = torch.full(
                (B, self.num_query_tokens), self.query_pad_token_id,
                device=device, dtype=dtype,
            )
            outputs["input_ids"] = torch.cat([input_ids, query_ids], dim=1)

            if "attention_mask" in outputs and outputs["attention_mask"] is not None:
                ones = torch.ones(
                    B, self.num_query_tokens,
                    device=device, dtype=outputs["attention_mask"].dtype,
                )
                outputs["attention_mask"] = torch.cat(
                    [outputs["attention_mask"], ones], dim=1,
                )

            if "mm_token_type_ids" in outputs and outputs["mm_token_type_ids"] is not None:
                zeros = torch.zeros(
                    B, self.num_query_tokens,
                    device=device, dtype=outputs["mm_token_type_ids"].dtype,
                )
                outputs["mm_token_type_ids"] = torch.cat(
                    [outputs["mm_token_type_ids"], zeros], dim=1,
                )
        else:
            for i in range(len(input_ids)):
                outputs["input_ids"][i] = (
                    list(input_ids[i])
                    + [self.query_pad_token_id] * self.num_query_tokens
                )
                if "attention_mask" in outputs and outputs["attention_mask"] is not None:
                    outputs["attention_mask"][i] = (
                        list(outputs["attention_mask"][i])
                        + [1] * self.num_query_tokens
                    )
                if "mm_token_type_ids" in outputs and outputs["mm_token_type_ids"] is not None:
                    outputs["mm_token_type_ids"][i] = (
                        list(outputs["mm_token_type_ids"][i])
                        + [0] * self.num_query_tokens
                    )

        return outputs
