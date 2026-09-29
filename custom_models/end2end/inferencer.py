"""
Standalone End2End model inferencer.

Extracted from VLMEvalKit's End2EndChat class. Handles model loading
(encoder + projection + decoder) and single-sample video-to-text inference,
with no dependency on VLMEvalKit base classes.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_processor(model_dir):
    from transformers import AutoProcessor
    from transformers.processing_utils import ProcessorMixin

    _orig = ProcessorMixin.check_argument_for_proper_class
    ProcessorMixin.check_argument_for_proper_class = lambda self, *a, **kw: None
    try:
        return AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    finally:
        ProcessorMixin.check_argument_for_proper_class = _orig


def load_safetensors_state_dict(model_dir: str) -> dict:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        sd = {}
        for shard in sorted(shard_files):
            sd.update(load_file(os.path.join(model_dir, shard)))
        return sd
    for fname in os.listdir(model_dir):
        if fname.endswith(".safetensors"):
            return load_file(os.path.join(model_dir, fname))
    raise FileNotFoundError(f"No safetensors files found in {model_dir}")


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


# ---------------------------------------------------------------------------
# End2EndInferencer
# ---------------------------------------------------------------------------

class End2EndInferencer:
    """Standalone End2End visual model inferencer.

    Loads encoder (Qwen3.5-VL) + projection + decoder (Qwen3-8B) from
    safetensors directories and provides a simple generate() method.
    """

    FRAME_FACTOR = 2

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        projection_path: str,
        num_query_tokens: int = 4,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        temperature: float = 0.7,
        do_sample: bool = True,
        system_prompt: Optional[str] = None,
        use_flash_attn: bool = True,
        fps: Optional[float] = None,
        nframe: int = 64,
        device: Optional[str] = None,
    ):
        self.num_query_tokens = num_query_tokens
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            do_sample=do_sample,
        )
        self.system_prompt = system_prompt
        self.fps = fps
        self.nframe = nframe

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model, self.encoder_processor, self.decoder_tokenizer, self.encoder_pad_token_id = \
            self._build_model(encoder_path, decoder_path, projection_path,
                              num_query_tokens, use_flash_attn, self.device)

    def _build_model(self, encoder_path, decoder_path, projection_path,
                     num_query_tokens, use_flash_attn, device):
        """Build encoder + projection + decoder.

        Adapted from End2EndChat._build_model() (end2end_chat.py:113-199).
        """
        from transformers import AutoTokenizer
        from custom_models.end2end.qwen_3_5_vl.configuration_qwen3_5 import Qwen3_5Config
        from custom_models.end2end.qwen3.configuration_qwen3 import Qwen3Config
        from custom_models.end2end.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from custom_models.end2end.modeling_end2end import Qwen3_5Encoder, End2EndModel
        from custom_models.end2end.processing_end2end import Qwen3_5EncoderProcessor

        attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"

        # --- Encoder ---
        print(f"[End2EndInferencer] Loading encoder config from {encoder_path} ...")
        encoder_config = Qwen3_5Config.from_pretrained(encoder_path)
        encoder_config._attn_implementation = attn_impl
        encoder_config.use_cache = False

        encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path, trust_remote_code=True)
        query_pad_token = "<|query_pad|>"
        if query_pad_token not in encoder_tokenizer.get_vocab():
            encoder_tokenizer.add_special_tokens({"additional_special_tokens": [query_pad_token]})
        query_pad_token_id = encoder_tokenizer.convert_tokens_to_ids(query_pad_token)
        encoder_config.text_config.vocab_size = len(encoder_tokenizer)
        if hasattr(encoder_config, "vocab_size"):
            encoder_config.vocab_size = len(encoder_tokenizer)

        # --- Decoder ---
        print(f"[End2EndInferencer] Loading decoder config from {decoder_path} ...")
        decoder_config = Qwen3Config.from_pretrained(decoder_path)
        decoder_config._attn_implementation = attn_impl
        decoder_config.use_cache = True

        decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_path, trust_remote_code=True)
        encoder_pad_token = "<|encoder_pad|>"
        if encoder_pad_token not in decoder_tokenizer.get_vocab():
            decoder_tokenizer.add_special_tokens({"additional_special_tokens": [encoder_pad_token]})
        encoder_pad_token_id = decoder_tokenizer.convert_tokens_to_ids(encoder_pad_token)
        decoder_config.vocab_size = len(decoder_tokenizer)

        # --- Build models ---
        print("[End2EndInferencer] Building encoder ...")
        encoder = Qwen3_5Encoder(encoder_config, num_query_tokens=num_query_tokens,
                                 query_pad_token_id=query_pad_token_id)

        print("[End2EndInferencer] Building decoder ...")
        decoder = Qwen3ForCausalLM(decoder_config)

        encoder_hidden_size = encoder_config.text_config.hidden_size
        decoder_hidden_size = decoder_config.hidden_size

        print("[End2EndInferencer] Building End2EndModel ...")
        model = End2EndModel(
            encoder=encoder,
            decoder=decoder,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            num_query_tokens=num_query_tokens,
            encoder_pad_token_id=encoder_pad_token_id,
        )

        # --- Load weights ---
        print("[End2EndInferencer] Loading encoder weights ...")
        enc_sd = load_safetensors_state_dict(encoder_path)
        missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
        if missing:
            print(f"  Encoder missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Encoder unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        print("[End2EndInferencer] Loading decoder weights ...")
        dec_sd = load_safetensors_state_dict(decoder_path)
        missing, unexpected = model.decoder.load_state_dict(dec_sd, strict=False)
        if missing:
            print(f"  Decoder missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Decoder unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        print("[End2EndInferencer] Loading projection weights ...")
        proj_sd = load_safetensors_state_dict(projection_path)
        model.projection.load_state_dict(proj_sd, strict=True)

        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()

        # --- Processor ---
        base_processor = _load_processor(encoder_path)
        encoder_processor = Qwen3_5EncoderProcessor.from_base_processor(
            base_processor, num_query_tokens=num_query_tokens,
        )

        print(f"[End2EndInferencer] Model loaded. encoder_hidden={encoder_hidden_size}, "
              f"decoder_hidden={decoder_hidden_size}, num_query_tokens={num_query_tokens}")

        return model, encoder_processor, decoder_tokenizer, encoder_pad_token_id

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def _encode_video(self, video_path: str) -> torch.Tensor:
        """Encode a video through the encoder and projection.

        Returns projected query embeddings of shape (1, num_query_tokens, decoder_hidden).
        """
        from custom_models.end2end.qwen_vl_utils_onerec import process_vision_info
        import cv2

        # 1. Build encoder text with video placeholder
        encoder_text = "<|vision_start|><|video_pad|><|vision_end|>"

        # 2. Determine nframes
        video_url = ensure_video_url(video_path)
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        video_item = {'type': 'video', 'video': video_url}
        if self.fps is not None:
            video_item['fps'] = self.fps
        else:
            if frame_count < self.nframe:
                nframes_val = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
            else:
                nframes_val = self.nframe
            video_item['nframes'] = nframes_val

        # 3. Process vision info
        messages_for_vision = [[{"role": "user", "content": [video_item]}]]
        image_inputs, video_inputs = process_vision_info(messages_for_vision)

        # 4. Encoder processing
        enc_kwargs = dict(text=[encoder_text], return_tensors="pt", padding=True)
        if video_inputs:
            enc_kwargs["videos"] = video_inputs
        enc_inputs = self.encoder_processor(**enc_kwargs)

        device = self.device
        enc_input_ids = enc_inputs["input_ids"].to(device)
        enc_attention_mask = enc_inputs.get("attention_mask")
        if enc_attention_mask is not None:
            enc_attention_mask = enc_attention_mask.to(device)

        pixel_values = enc_inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)

        pixel_values_videos = enc_inputs.get("pixel_values_videos")
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(device=device, dtype=torch.bfloat16)

        image_grid_thw = enc_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        video_grid_thw = enc_inputs.get("video_grid_thw")
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(device)

        # 5. Run encoder -> projection
        projected = self.model.encode(
            input_ids=enc_input_ids,
            attention_mask=enc_attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        return projected

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff'}

    @torch.no_grad()
    def _encode_image(self, image_path: str) -> torch.Tensor:
        """Encode a single image through the encoder and projection.

        Returns projected query embeddings of shape (1, num_query_tokens, decoder_hidden).
        """
        from custom_models.end2end.qwen_vl_utils_onerec import process_vision_info

        encoder_text = "<|vision_start|><|image_pad|><|vision_end|>"
        image_url = ensure_image_url(image_path)
        image_item = {'type': 'image', 'image': image_url}

        messages_for_vision = [[{"role": "user", "content": [image_item]}]]
        image_inputs, video_inputs = process_vision_info(messages_for_vision)

        enc_kwargs = dict(text=[encoder_text], return_tensors="pt", padding=True)
        if image_inputs:
            enc_kwargs["images"] = image_inputs
        enc_inputs = self.encoder_processor(**enc_kwargs)

        device = self.device
        enc_input_ids = enc_inputs["input_ids"].to(device)
        enc_attention_mask = enc_inputs.get("attention_mask")
        if enc_attention_mask is not None:
            enc_attention_mask = enc_attention_mask.to(device)

        pixel_values = enc_inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)

        image_grid_thw = enc_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        projected = self.model.encode(
            input_ids=enc_input_ids,
            attention_mask=enc_attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=None,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
        )
        return projected

    @torch.no_grad()
    def generate(self, video_path: str, formatted_prompt: str) -> tuple:
        """Run End2End inference for a single video or image + formatted prompt.

        Args:
            video_path: Path to the video or image file.
            formatted_prompt: Already formatted prompt string from apply_chat_template,
                containing <|encoder_pad|> tokens as visual placeholders.

        Returns:
            (generated_text, n_input_tokens, n_output_tokens)
        """
        # 1. Encode visual input -> projected embeddings (route by file extension)
        ext = os.path.splitext(video_path)[1].lower()
        if ext in self.IMAGE_EXTENSIONS:
            projected = self._encode_image(video_path)
        else:
            projected = self._encode_video(video_path)

        # 2. Tokenize the formatted prompt (decoder tokenizer knows <|encoder_pad|>)
        decoder_ids = self.decoder_tokenizer.encode(formatted_prompt, add_special_tokens=False)
        decoder_input_ids = torch.tensor([decoder_ids], dtype=torch.long, device=self.device)

        n_input = len(decoder_ids)

        n_enc_pad = int((decoder_input_ids == self.encoder_pad_token_id).sum().item())
        if n_enc_pad != self.num_query_tokens:
            raise ValueError(
                f"<|encoder_pad|> count mismatch: got {n_enc_pad}, expected {self.num_query_tokens}"
            )

        # 3. Scatter projected embeddings into <|encoder_pad|> positions
        decoder_embeds = self.model.decoder.model.embed_tokens(decoder_input_ids)
        enc_mask = (decoder_input_ids == self.encoder_pad_token_id)
        enc_mask_3d = enc_mask.unsqueeze(-1).expand_as(decoder_embeds)
        decoder_embeds = decoder_embeds.masked_scatter(
            enc_mask_3d, projected.to(decoder_embeds.dtype),
        )

        # 4. Generate
        outputs = self.model.decoder.generate(
            inputs_embeds=decoder_embeds,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            pad_token_id=self.decoder_tokenizer.pad_token_id or self.decoder_tokenizer.eos_token_id,
            **self.generate_kwargs,
        )

        generated_ids = outputs[0]
        n_output = len(generated_ids)
        response = self.decoder_tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response, n_input, n_output
