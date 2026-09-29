"""
Tongshi Choice Task Configuration

通识选择题数据集：中文 4 选 1 单选 + 多选混合，答案为 letters 字符串
（单选 "A"，多选按字母升序如 "ABC"）。
"""

TONGSHI_CHOICE_CONFIG = {
    "name": "tongshi_choice",
    "source": "Tongshi Choice (通识选择题)",
    "splits": ["test"],
    "description": "Chinese general-knowledge multiple-choice (single + multi-select) benchmark",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {
        "enable_thinking": False,
        "force_thinking": False,
        "custom_chat_template": "qwen3_choice.jinja2",
    },
    "generation_config": {
        "num_return_sequences": 1,
        "max_new_tokens": 60000,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "do_sample": True,
    },
    "evaluation_config": {
        "metrics": ["pass@k"],
        "k_values": [1],
    },
}
