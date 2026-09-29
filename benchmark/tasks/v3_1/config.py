"""Configuration dependencies for the competition tasks (data v3.1)."""

_SID_TO_PID_VIDEO_AD = {
    "sid_to_pid_strategy": "max_pid",
    "mapping_filename": "sid2pid_video_ad.json",
}

_SID_TO_PID_PRODUCT = {
    "sid_to_pid_strategy": "max_pid",
    "mapping_filename": "sid2pid_product.json",
}

_SID_TO_PID_LIVE = {
    "sid_to_pid_strategy": "max_pid",
    "mapping_filename": "sid2pid_live.json",
}

from benchmark.tasks.v2_0.recommendation.config import (
    _BASE_EVAL,
    _SID_PATTERN_3,
    _PROMPT_CONFIG as _RECOMMENDATION_PROMPT_CONFIG,
    _make_generation_config as _recommendation_make_generation_config,
)

_RECOMMENDATION_EVALUATION_CONFIG = {
    **_BASE_EVAL,
    "centroid_model_dir": None,
    # "centroid_model_dir": "/path/to/resource",
    "centroid_num_tokens": 1,
    "centroid_num_layers": 3,
    "sid_pattern": _SID_PATTERN_3,
}

VIDEO_CONFIG = {
    "name": "video",
    "source": "Benchmark dataset",
    "splits": ["test"],
    "description": "Next video prediction",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {**_RECOMMENDATION_PROMPT_CONFIG},
    "generation_config": _recommendation_make_generation_config("<|video_begin|>", 3),
    "evaluation_config": {**_RECOMMENDATION_EVALUATION_CONFIG, **_SID_TO_PID_VIDEO_AD},
}

PRODUCT_CONFIG = {
    "name": "product",
    "source": "Benchmark dataset",
    "splits": ["test"],
    "description": "Predict next clicked product",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {**_RECOMMENDATION_PROMPT_CONFIG},
    "generation_config": _recommendation_make_generation_config("<|prod_begin|>", 3),
    "evaluation_config": {**_RECOMMENDATION_EVALUATION_CONFIG, **_SID_TO_PID_PRODUCT},
}

AD_CONFIG = {
    "name": "ad",
    "source": "Benchmark dataset",
    "splits": ["test"],
    "description": "Predict next clicked advertisement",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {**_RECOMMENDATION_PROMPT_CONFIG},
    "generation_config": _recommendation_make_generation_config("<|ad_begin|>", 3),
    "evaluation_config": {**_RECOMMENDATION_EVALUATION_CONFIG, **_SID_TO_PID_VIDEO_AD},
}

LIVE_CONFIG = {
    "name": "live",
    "source": "Benchmark dataset",
    "splits": ["test"],
    "description": "Predict next live stream",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {**_RECOMMENDATION_PROMPT_CONFIG},
    "generation_config": _recommendation_make_generation_config("<|living_begin|>", 3),
    "evaluation_config": {**_RECOMMENDATION_EVALUATION_CONFIG, **_SID_TO_PID_LIVE},
}

from benchmark.tasks.v2_0.sid2caption.config import (
    _make_sid2caption_config as _v2_make_sid2caption_config,
)

def _make_challenge_sid2caption_config(domain: str) -> dict:
    config = _v2_make_sid2caption_config(domain)
    config["name"] = f"challenge_itemic_pattern_caption_{domain}"
    config["evaluation_config"] = {
        **config["evaluation_config"],
        "llm_judge_model": "openai/configure-judge-model",
        "llm_max_workers": 128,
        "include_failed_samples_in_macro": True,
        "minimum_match_success_rate": 0.8,
    }
    return config

CHALLENGE_SID2CAPTION_VIDEO_CONFIG = _make_challenge_sid2caption_config("video")

CHALLENGE_SID2CAPTION_PRODUCT_CONFIG = _make_challenge_sid2caption_config("product")

CHALLENGE_SID2CAPTION_AD_CONFIG = _make_challenge_sid2caption_config("ad")

