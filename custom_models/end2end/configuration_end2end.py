from dataclasses import dataclass, field
from typing import Optional


@dataclass
class End2EndConfig:
    """Configuration for the End-to-End encoder-decoder model.

    The encoder is a Qwen3.5-VL model with appended learnable query tokens.
    The decoder is a Qwen3-8B causal LM conditioned on the encoder output.
    """

    encoder_model_dir: str = ""
    decoder_model_dir: str = ""
    num_query_tokens: int = 256
    projection_hidden_size: Optional[int] = None
    encoder_pad_token: str = "<|encoder_pad|>"
    query_pad_token: str = "<|query_pad|>"
