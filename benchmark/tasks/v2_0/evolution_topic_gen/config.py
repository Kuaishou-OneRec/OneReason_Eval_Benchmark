"""
Evolution Topic Gen Task Configuration
"""
import os

EVOLUTION_TOPIC_GEN_CONFIG = {
    "name": "evolution_topic_gen",
    "source": "OneReason_Eval_Benchmark",
    "splits": ["test"],
    "description": "Evaluate model's ability to generate interest evolution logic chains for specific topics",
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
        "metrics": ["action_f1_score", "logic_f1_score", "combined_f1_score", "logic_entailment_rate"],
        "nli_enabled": True,
        "nli_backend": "huggingface",
        "nli_model": "cross-encoder/nli-deberta-v3-base",
        "nli_model_dir": os.environ.get("NLI_MODEL_DIR"),
        "nli_judge_model": "gemini",
    }
}
