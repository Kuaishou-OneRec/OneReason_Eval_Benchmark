#!/usr/bin/env python3
"""
交互式 caption → SID 推理脚本
超参数默认与 caption2sid_video 配置一致
"""
import os
import re
import sys
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# caption2sid video 默认配置（v7.0: 3-token SID）
DOMAIN_CFG = {"prompt_token": "<|video_begin|>", "max_new_tokens": 3, "sid_pattern": r'<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>'}

GENERATION_DEFAULTS = {
    "num_return_sequences": 64,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.0,
    "frequency_penalty": 1.0,
}

TEMPLATE_DIR = Path(__file__).parents[3] / "benchmark/tasks/templates"

SYSTEM_PROMPT = (
    "你是一个视频推荐系统的核心模型，擅长理解用户兴趣画像。"
    "给定用户的历史观看行为与偏好描述，你需要预测用户最可能感兴趣的下一个视频，"
    "并以视频 token 序列的形式输出。"
)


def build_prompt(caption: str, tokenizer, enable_thinking: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": caption},
    ]
    chat_template = (TEMPLATE_DIR / "qwen3_soft_switch.jinja2").read_text()
    return tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def extract_sids(texts: list[str], pattern: str) -> list[tuple]:
    """返回 (数字元组 或 None, 原始文本)"""
    results = []
    for text in texts:
        m = re.search(pattern, text)
        results.append((m.groups() if m else None, text.strip()))
    return results


def main():
    print("=" * 50)
    print("Caption → SID 交互式推理")
    print("=" * 50)

    default_model = os.environ.get("BENCHMARK_MODEL_PATH", "models/model")
    model_path = input(f"\n请输入模型路径 [默认: {default_model}]: ").strip() or default_model

    cfg = DOMAIN_CFG

    n_str = input(f"num_return_sequences [默认: {GENERATION_DEFAULTS['num_return_sequences']}]: ").strip()
    n = int(n_str) if n_str else GENERATION_DEFAULTS["num_return_sequences"]

    print(f"\n正在加载模型: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    llm = LLM(model=model_path, trust_remote_code=False)

    sampling_params = SamplingParams(
        n=n,
        temperature=GENERATION_DEFAULTS["temperature"],
        top_p=GENERATION_DEFAULTS["top_p"],
        top_k=GENERATION_DEFAULTS["top_k"],
        max_tokens=cfg["max_new_tokens"],
        presence_penalty=GENERATION_DEFAULTS["presence_penalty"],
        frequency_penalty=GENERATION_DEFAULTS["frequency_penalty"],
    )

    max_model_len = llm.llm_engine.model_config.max_model_len
    print(f"\n模型加载完成！最大输入 token 数: {max_model_len}")
    print("输入 caption（quit/exit 退出）\n" + "=" * 50)

    while True:
        sys.stdout.write("\n[Caption] ")
        sys.stdout.flush()
        caption = sys.stdin.buffer.readline().decode('utf-8', errors='replace').strip()
        if not caption:
            continue
        if caption.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        full_caption = caption + "\n请你推荐这个用户最可能感兴趣的下一个视频。"
        base_prompt = build_prompt(full_caption, tokenizer)
        prompt_with_token = base_prompt + cfg["prompt_token"]
        input_tokens = len(tokenizer.encode(prompt_with_token))
        print(f"[输入 token 数: {input_tokens} / {max_model_len}]")

        outputs = llm.generate([prompt_with_token], sampling_params)
        texts = [o.text for o in outputs[0].outputs]
        sids = extract_sids(texts, cfg["sid_pattern"])

        print(f"\n[生成的 SID（共 {len(sids)} 个）]")
        for i, (nums, raw) in enumerate(sids, 1):
            if nums:
                print(f"  {i}. {'-'.join(nums)}")
            else:
                print(f"  {i}. [未匹配] {raw}")


if __name__ == "__main__":
    main()
