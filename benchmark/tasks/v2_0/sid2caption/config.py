"""Shared configuration builders used by data v3.1."""

_PROMPT_CONFIG = {
    "enable_thinking": False,
    "custom_chat_template": "qwen3_soft_switch.jinja2",
}

_GENERATION_CONFIG = {
    "num_return_sequences": 1,
    "max_new_tokens": 512,
    "temperature": 0.01,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "do_sample": False,
    "num_return_thinking_sequences": 1,
    "max_new_thinking_tokens": 1000,
}

_EVALUATION_CONFIG = {
    "metrics": ["macro_wip_double_weighted_f1"],
    "bertscore_model_type": "bert-base-chinese",
    "bertscore_num_layers": 9,
    "bertscore_lang": "zh",
    "wip_enabled": True,
    "llm_judge_model": "gemini/gemini-2.5-flash-lite",
    "llm_max_workers": 10,
    "wip_core_threshold": 5,
    "llm_max_samples": None,
}

def _make_sid2caption_config(domain: str) -> dict:
    return {
        "name": f"sid2caption_{domain}",
        "source": "sid2caption",
        "splits": ["test"],
        "description": f"SID to Caption generation task ({domain})",
        "data_fields": {
            "messages_field": "messages",
            "metadata_field": "metadata",
        },
        "prompt_config": {**_PROMPT_CONFIG},
        "generation_config": {**_GENERATION_CONFIG},
        "evaluation_config": {**_EVALUATION_CONFIG},
    }
