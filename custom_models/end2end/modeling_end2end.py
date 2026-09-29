"""
End-to-end encoder-decoder model.

Qwen3_5Encoder: Qwen3.5-VL backbone with learnable query embeddings appended.
    Processes multimodal input and returns N query hidden states.

End2EndModel: Chains encoder -> projection -> Qwen3-8B decoder for
    end-to-end training with NTP loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .qwen_3_5_vl.modeling_qwen3_5 import (
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
    Qwen3_5ModelOutputWithPast,
)
from .qwen_3_5_vl.configuration_qwen3_5 import Qwen3_5Config
from .qwen3.modeling_qwen3 import Qwen3ForCausalLM
from .qwen3.configuration_qwen3 import Qwen3Config


# ---------------------------------------------------------------------------
# Encoder output
# ---------------------------------------------------------------------------

@dataclass
class EncoderOutput:
    query_hidden_states: torch.FloatTensor
    last_hidden_state: torch.FloatTensor | None = None


# ---------------------------------------------------------------------------
# Qwen3_5Encoder
# ---------------------------------------------------------------------------

class Qwen3_5Encoder(Qwen3_5PreTrainedModel):
    """Qwen3.5-VL used as a multimodal encoder.

    Uses *num_query_tokens* learnable query embeddings that are scattered
    into ``<|query_pad|>`` placeholder positions in the input sequence.
    Because these tokens are inserted per-sample *before* packing, every
    sample in a packed batch has its own query positions.  Causal attention
    ensures queries attend to the full preceding context.

    The forward method returns the hidden states at the query positions.
    """

    config_class = Qwen3_5Config

    def __init__(
        self,
        config: Qwen3_5Config,
        num_query_tokens: int = 256,
        query_pad_token_id: int | None = None,
    ):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.num_query_tokens = num_query_tokens
        self.query_pad_token_id = query_pad_token_id

        hidden_size = config.text_config.hidden_size
        self.query_embeddings = nn.Parameter(
            torch.empty(1, num_query_tokens, hidden_size)
        )
        nn.init.normal_(self.query_embeddings, std=0.02)

        self.post_init()

    # -- Delegated accessors (needed for weight loading) --------------------

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    # -- Forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs,
    ) -> EncoderOutput:
        # ---- 1. Text embeddings + vision injection (mirrors Qwen3_5Model) ----
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = self.model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.model.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # ---- 2. Scatter query embeddings into <|query_pad|> positions ------
        #   The processor has already inserted query_pad tokens per-sample,
        #   so position_ids and cu_seqlens already account for them.
        #   masked_scatter is differentiable — gradients flow back to
        #   self.query_embeddings.
        if self.query_pad_token_id is None:
            raise ValueError("query_pad_token_id is None. Please set it when building Qwen3_5Encoder.")

        query_mask = (input_ids == self.query_pad_token_id)
        if not query_mask.any():
            raise ValueError(
                "No <|query_pad|> tokens found in encoder input_ids. "
                "Please check processor/tokenizer setup."
            )

        num_total_queries = int(query_mask.sum().item())
        if num_total_queries % self.num_query_tokens != 0:
            raise ValueError(
                "Invalid query placeholder count. "
                f"total_query_pad={num_total_queries}, "
                f"num_query_tokens={self.num_query_tokens}, "
                f"input_shape={tuple(input_ids.shape)}"
            )
        num_query_groups = num_total_queries // self.num_query_tokens

        query_embeds = self.query_embeddings.squeeze(0).repeat(num_query_groups, 1)
        query_embeds = query_embeds.to(
            dtype=inputs_embeds.dtype, device=inputs_embeds.device,
        )
        query_mask_3d = query_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            query_mask_3d, query_embeds,
        )

        # ---- 3. Run language model -----------------------------------------
        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # ---- 4. Extract query hidden states at <|query_pad|> positions -----
        hidden_states = outputs.last_hidden_state
        gathered_queries = hidden_states[query_mask]
        query_hidden_states = gathered_queries.reshape(
            -1, self.num_query_tokens, hidden_states.shape[-1],
        )
        if query_hidden_states.shape[0] != num_query_groups:
            raise ValueError(
                "Query hidden-state reshape mismatch. "
                f"expected_groups={num_query_groups}, got={query_hidden_states.shape[0]}, "
                f"hidden_shape={tuple(hidden_states.shape)}"
            )

        return EncoderOutput(
            query_hidden_states=query_hidden_states,
            last_hidden_state=hidden_states,
        )


# ---------------------------------------------------------------------------
# End2EndModel
# ---------------------------------------------------------------------------

class End2EndModel(nn.Module):
    """End-to-end encoder-decoder for multimodal-to-text generation.

    The encoder (Qwen3.5-VL) compresses a multimodal input into
    *num_query_tokens* embeddings.  A learned projection maps them to
    the decoder (Qwen3-8B) hidden dimension, where they replace
    ``<|encoder_pad|>`` placeholder tokens at the beginning of the
    decoder sequence.  The decoder then generates a textual description
    via next-token prediction.
    """

    def __init__(
        self,
        encoder: Qwen3_5Encoder,
        decoder: Qwen3ForCausalLM,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        num_query_tokens: int = 256,
        encoder_pad_token_id: int = 0,
        projection_hidden_size: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_query_tokens = num_query_tokens
        self.encoder_pad_token_id = encoder_pad_token_id

        if projection_hidden_size is not None:
            self.projection = nn.Sequential(
                nn.Linear(encoder_hidden_size, projection_hidden_size),
                nn.GELU(),
                nn.Linear(projection_hidden_size, decoder_hidden_size),
            )
        elif encoder_hidden_size != decoder_hidden_size:
            self.projection = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        else:
            self.projection = nn.Linear(encoder_hidden_size, decoder_hidden_size)

    # -- Forward (returns decoder logits; loss computed externally) ----------

    def forward(
        self,
        # ---- encoder inputs ----
        encoder_input_ids: torch.LongTensor,
        encoder_position_ids: torch.LongTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_cu_seqlens: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        # ---- decoder inputs ----
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_labels: torch.LongTensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        decoder_cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ):
        if decoder_input_ids is None:
            raise ValueError("decoder_input_ids must be provided.")

        # 1. Encode --------------------------------------------------------
        encoder_kwargs = dict(kwargs)
        # Backward-compatible path: still accept legacy cu_seqlens via kwargs.
        if encoder_cu_seqlens is None and "cu_seqlens" in encoder_kwargs:
            encoder_cu_seqlens = encoder_kwargs.pop("cu_seqlens")
        if encoder_cu_seqlens is not None:
            encoder_kwargs["cu_seqlens"] = encoder_cu_seqlens

        enc_out: EncoderOutput = self.encoder(
            input_ids=encoder_input_ids,
            position_ids=encoder_position_ids,
            attention_mask=encoder_attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            **encoder_kwargs,
        )
        query_hidden_states = enc_out.query_hidden_states  # (B, N, H_enc)

        # 2. Project -------------------------------------------------------
        projected = self.projection(query_hidden_states)  # (B, N, H_dec)
        if projected.shape[1] != self.num_query_tokens:
            raise ValueError(
                "Projected query count mismatch. "
                f"expected_N={self.num_query_tokens}, got_N={projected.shape[1]}, "
                f"projected_shape={tuple(projected.shape)}"
            )
        projected_for_scatter = projected
        if decoder_input_ids.shape[0] == projected.shape[0]:
            pass
        elif (
            decoder_input_ids.shape[0] == 1
            and projected.shape[0] > 1
            and decoder_cu_seqlens is not None
        ):
            # Packed decoder mode: decoder is flattened to [1, total_len],
            # while encoder queries are grouped as [num_samples, N, H].
            num_decoder_groups = int(decoder_cu_seqlens.numel()) - 1
            if num_decoder_groups != projected.shape[0]:
                raise ValueError(
                    "Packed encoder/decoder group count mismatch. "
                    f"decoder_groups={num_decoder_groups} from decoder_cu_seqlens, "
                    f"encoder_groups={projected.shape[0]}, "
                    f"decoder_cu_seqlens={decoder_cu_seqlens.tolist()}"
                )
            projected_for_scatter = projected.reshape(1, -1, projected.shape[-1])

        else:
            raise ValueError(
                "Encoder/decoder batch mismatch. "
                f"encoder_groups={projected.shape[0]}, "
                f"decoder_batch={decoder_input_ids.shape[0]}, "
                f"decoder_input_shape={tuple(decoder_input_ids.shape)}"
            )

        # 3. Build decoder inputs_embeds -----------------------------------
        decoder_embeds = self.decoder.model.embed_tokens(decoder_input_ids)

        enc_mask = (decoder_input_ids == self.encoder_pad_token_id)
        expected_slots = projected_for_scatter.shape[0] * projected_for_scatter.shape[1]
        actual_slots = int(enc_mask.sum().item())
        if actual_slots != expected_slots:
            raise ValueError(
                "Decoder <|encoder_pad|> count mismatch before scatter. "
                f"expected_slots={expected_slots}, "
                f"actual_slots={actual_slots}, decoder_input_shape={tuple(decoder_input_ids.shape)}"
            )
        enc_mask_3d = enc_mask.unsqueeze(-1).expand_as(decoder_embeds)
        decoder_embeds = decoder_embeds.masked_scatter(
            enc_mask_3d,
            projected_for_scatter.to(decoder_embeds.dtype),
        )

        # 4. Decoder forward -----------------------------------------------
        outputs = self.decoder(
            inputs_embeds=decoder_embeds,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            labels=decoder_labels,
            cu_seqlens=decoder_cu_seqlens,
        )

        return outputs

    # -- Convenience: encoder-only forward ----------------------------------

    def encode(self, **encoder_kwargs) -> torch.FloatTensor:
        """Run encoder and return projected query embeddings."""
        enc_out = self.encoder(**encoder_kwargs)
        return self.projection(enc_out.query_hidden_states)
