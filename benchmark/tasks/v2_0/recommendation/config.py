"""Shared configuration builders used by data v3.1."""

import os

_PROMPT_CONFIG = {
    "enable_thinking": False,
    "custom_chat_template": "qwen3_soft_switch.jinja2",
}

_SID_PATTERN_3 = r'<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>'

_BASE_EVAL = {
    "metrics": ["pass@k", "position1_pass@k", "recall@k", "mean_emb_sim@k", "max_emb_sim@k", "incremental_value_weighted_pid_recall@k", "play_count_weighted_pid_recall@k", "duration_weighted_pid_recall@k", "sid_1th_recall@k"],
    "k_values": [1, 4, 8, 16, 32, 64],
    "select_k": "first_k",
    "evaluation_mode": "both",
    "centroid_model_dir": os.environ.get("SID_CENTROID_MODEL_DIR"),
}

def _make_generation_config(prompt_token: str, max_new_tokens: int) -> dict:
    return {
        "num_return_sequences": 64,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 50,
        "presence_penalty": 1.0,
        "frequency_penalty": 1.0,
        "prompt_token": prompt_token,
        "max_new_thinking_tokens": 4096,
        "num_return_thinking_sequences": 1,
        "num_beams": 64,
    }
