"""
vLLM 批量生成命令行工具

支持单模型和多模型并行评估。在单机多卡上自动管理 GPU 资源分配。

使用示例:
    # 多模型评估（推荐）
    python generate.py \
        --input_json input.json \
        --output output.json

    # 指定可用的 GPU
    python generate.py \
        --input_json input.json \
        --output output.json \
        --gpus 0 1 2 3

    # 每个模型使用 2 张卡
    python generate.py \
        --input_json input.json \
        --output output.json \
        --tensor_parallel_size 2

输入 JSON 格式（多模型）:
    {
        "models": [
            "Qwen/Qwen2-7B",
            "Qwen/Qwen2-14B"
        ],
        "prompts": ["问题1", "问题2"],
        "params": {
            "temperature": 0.7,
            "max_new_tokens": 128
        }
    }

输入 JSON 格式（单模型，models 列表只有一个元素）:
    {
        "models": ["Qwen/Qwen2-7B"],
        "prompts": ["问题1", "问题2"],
        "params": {
            "temperature": 0.7,
            "max_new_tokens": 128
        }
    }

输出 JSON 格式:
    {
        "results": {
            "Qwen_Qwen2-7B": [
                {
                    "prompt": "问题1",
                    "generated_texts": ["结果1", "结果2"]
                },
                ...
            ],
            ...
        },
        "params": {...},
        "models": [...]
    }
"""

import argparse
import sys

from scripts.vllm.inference.interface import MultiModelEvaluator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="vLLM 批量生成工具（支持单模型和多模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python generate.py --input_json input.json --output output.json

  # 指定 GPU
  python generate.py --input_json input.json --output output.json --gpus 0 1 2 3

  # 每个模型使用 2 张卡
  python generate.py --input_json input.json --output output.json --tensor_parallel_size 2

说明:
  - 输入 JSON 必须包含 "models" 字段（模型路径列表）
  - 支持单模型评估：models 列表只包含一个模型即可
  - 支持多模型评估：models 列表包含多个模型，自动并行执行
        """
    )

    # 必需参数
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="输入 JSON 文件路径（包含 models, prompts 和 params）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径"
    )

    # GPU 配置
    parser.add_argument(
        "--gpus",
        type=int,
        nargs='+',
        default=None,
        help="可用的 GPU ID 列表（例如 --gpus 0 1 2 3）。如果不指定，自动检测所有 GPU"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="每个模型使用的 GPU 数量（默认 1）"
    )

    # 模型参数
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="模型数据类型（默认 bfloat16）"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU 内存利用率 (0-1，默认 0.9)"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 80)
    print("vLLM 批量生成工具")
    print("=" * 80)
    print(f"\n输入文件: {args.input_json}")
    print(f"输出文件: {args.output}")
    if args.gpus:
        print(f"可用 GPU: {args.gpus}")
    else:
        print(f"可用 GPU: 自动检测")
    print(f"每个模型使用 GPU 数: {args.tensor_parallel_size}")
    print(f"数据类型: {args.dtype}")
    print(f"GPU 内存利用率: {args.gpu_memory_utilization}")
    print()

    try:
        # 创建评估器
        print("正在初始化评估器...")
        evaluator = MultiModelEvaluator(
            available_gpus=args.gpus,
            default_tensor_parallel_size=args.tensor_parallel_size,
            default_dtype=args.dtype,
            default_gpu_memory_utilization=args.gpu_memory_utilization
        )

        # 执行评估
        print(f"\n正在从 {args.input_json} 读取配置并执行评估...")
        results = evaluator.evaluate_from_json(
            input_json_path=args.input_json,
            output_path=args.output
        )

        # 打印统计信息
        print("\n" + "=" * 80)
        print("评估完成!")
        print("=" * 80)
        num_models = len(results['results'])
        print(f"成功评估了 {num_models} 个模型")
        if num_models == 1:
            print(f"(单模型评估)")
        elif num_models > 1:
            print(f"(多模型并行评估)")

        if results['results']:
            num_prompts = len(list(results['results'].values())[0])
            print(f"每个模型处理了 {num_prompts} 个 prompts")
        print(f"结果已保存到: {args.output}")
        print()

    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
