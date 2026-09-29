#!/usr/bin/env python3
"""
简单的交互式chat脚本，使用vLLM进行推理
"""
from vllm import LLM, SamplingParams


def main():
    print("="*50)
    print("交互式配置")
    print("="*50)

    # 交互式输入模型路径
    model_path = input("\n请输入模型路径: ").strip()

    # 交互式输入推理参数（提供默认值）
    print("\n请输入推理参数（直接按回车使用默认值）:")

    temperature = input("temperature [默认: 0.01]: ").strip()
    temperature = float(temperature) if temperature else 0.01

    top_p = input("top_p [默认: 0.95]: ").strip()
    top_p = float(top_p) if top_p else 0.95

    top_k = input("top_k [默认: 1]: ").strip()
    top_k = int(top_k) if top_k else 1

    max_tokens = input("max_new_tokens [默认: 512]: ").strip()
    max_tokens = int(max_tokens) if max_tokens else 512

    n = input("num_return_sequences [默认: 1]: ").strip()
    n = int(n) if n else 1

    repetition_penalty = input("repetition_penalty [默认: 1]: ").strip()
    repetition_penalty = float(repetition_penalty) if repetition_penalty else 1.0

    # 推理参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        n=n,
        repetition_penalty=repetition_penalty,
        logprobs=20  # 获取top 20个token的概率
    )

    print(f"\n正在加载模型: {model_path}")
    print("这可能需要一些时间...")

    # 加载模型（只加载一次）
    llm = LLM(model=model_path)

    print("\n模型加载完成！")
    print("="*50)
    print("你可以开始输入prompt了（输入 'quit' 或 'exit' 退出）")
    print("="*50)

    # 交互式循环
    while True:
        prompt = input("\n[你] ")

        if prompt.strip().lower() in ['quit', 'exit', 'q']:
            print("\n再见！")
            break

        if not prompt.strip():
            continue

        # 生成回答
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text

        # 获取第一个token的logprobs
        first_token_logprobs = outputs[0].outputs[0].logprobs[0] if outputs[0].outputs[0].logprobs else None

        if first_token_logprobs:
            print("\n[第一个token的概率分布（Top 10）]")
            # 按概率从高到低排序
            import math
            sorted_tokens = sorted(first_token_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
            for i, (_, logprob_obj) in enumerate(sorted_tokens[:10]):
                prob = math.exp(logprob_obj.logprob)
                print(f"  {i+1}. Token: '{logprob_obj.decoded_token}' | 概率: {prob:.4f} ({prob*100:.2f}%)")

            # 特别检查"是"和"否"的概率
            print("\n[特别关注：'是'/'否'的概率]")
            target_tokens = ["是", "否"]
            for target in target_tokens:
                found = False
                for _, logprob_obj in first_token_logprobs.items():
                    if logprob_obj.decoded_token.strip() == target:
                        prob = math.exp(logprob_obj.logprob)
                        print(f"  '{target}': {prob:.6f} ({prob*100:.4f}%)")
                        found = True
                        break
                if not found:
                    print(f"  '{target}': 未在top 20中找到（概率很低）")

        print(f"\n[模型] {response}")


if __name__ == "__main__":
    main()
