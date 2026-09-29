"""
Evolution Select Task Configuration
"""


EVOLUTION_SELECT_CONFIG = {
    "name": "evolution_select",
    "source": "OneReason_Eval_Benchmark",
    "splits": ["test"],
    "description": "Evaluate model's ability to select all timeline interactions relevant to a given evolution theme",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {
        "enable_thinking": False,
        "force_thinking": False,
        "custom_chat_template": "qwen3_soft_switch.jinja2",
    },
    "generation_config": {
        "num_return_sequences": 1,
        "max_new_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.0,
        "do_sample": True,
    },
    "evaluation_config": {
        "primary_metric": "f1",
        "metrics": ["precision", "recall", "f1"],
    },
}
