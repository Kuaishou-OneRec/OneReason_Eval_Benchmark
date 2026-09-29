#!/usr/bin/env python
"""
Ray + vLLM 多机多卡批量推理脚本

功能：
- 输入：Parquet 文件（包含 messages 列）
- 输出：Parquet 文件（messages 列存储 prompt + 回答拼接）
- 支持断点续传（按 prompt 粒度）
- 支持跨机 Tensor Parallel

使用示例：
    python generate.py \
        --input_parquet input.parquet \
        --output_parquet output.parquet \
        --model_path /path/to/model \
        --tensor_parallel_size 2 \
        --ray_address auto
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import pyarrow.parquet as pq

from utils.generator import RayVllmGenerator
from checkpoint import CheckpointManager


def _convert_messages_format(messages: list) -> list:
    """
    转换消息格式

    {"role": "...", "content": [{"type": "text", "text": "..."}]}
    ->
    {"role": "...", "content": "..."}

    Args:
        messages: 原始消息列表

    Returns:
        转换后的消息列表
    """
    converted = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            # 从 content 列表中提取文本
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            converted.append({
                "role": msg.get("role"),
                "content": "".join(text_parts)
            })
        else:
            # 已经是旧格式
            converted.append(msg)
    return converted


def _is_thinking_empty(generated_text: str) -> bool:
    """
    检测生成文本中的 thinking 内容是否为空或无意义

    Args:
        generated_text: 生成的文本，可能包含 <think>...</think> 标签

    Returns:
        True 如果 thinking 内容为空、只包含空白字符或无意义内容，False 否则
    """
    import re
    import json

    # 提取 <think>...</think> 之间的内容
    match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)

    if match:
        thinking_content = match.group(1).strip()

        # 检查是否为空或只包含空白字符
        if len(thinking_content) == 0:
            return True

        # 检查是否只包含无意义的 JSON 结构
        # 例如: [{"imestop":0,"ismaximize":true}] 或类似的控制信息
        try:
            # 尝试解析为 JSON
            parsed = json.loads(thinking_content)

            # 如果是列表或字典，检查是否只包含控制字段
            if isinstance(parsed, (list, dict)):
                # 将内容转为字符串，检查是否只包含特定的控制关键字
                content_str = json.dumps(parsed).lower()
                control_keywords = ['imestop', 'ismaximize', 'timestop', 'maximize']

                # 如果内容很短（<100字符）且包含控制关键字，认为是无意义的
                if len(content_str) < 100 and any(kw in content_str for kw in control_keywords):
                    return True
        except (json.JSONDecodeError, ValueError):
            # 不是有效的 JSON，继续检查其他条件
            pass

        # 检查是否只包含很少的有意义字符（例如少于10个非空白字符）
        non_whitespace_chars = len(re.sub(r'\s', '', thinking_content))
        if non_whitespace_chars < 10:
            return True

        return False

    # 如果没有找到 <think> 标签，认为不是空的（可能是非两阶段生成）
    return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Ray + vLLM 多机多卡批量推理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 输入输出
    parser.add_argument(
        "--input_parquet", type=str, required=True,
        help="输入 Parquet 文件路径"
    )
    parser.add_argument(
        "--output_parquet", type=str, required=True,
        help="输出 Parquet 文件路径"
    )
    parser.add_argument(
        "--input_column", type=str, default="messages",
        help="输入列名（prompt 所在列）"
    )
    parser.add_argument(
        "--output_column", type=str, default="messages",
        help="输出列名（存储 prompt + 回答）"
    )

    # 模型配置
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型路径或 HuggingFace 模型名"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="PT checkpoint 路径（可选）"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        help="模型数据类型"
    )
    parser.add_argument(
        "--max_model_len", type=int, default=None,
        help="最大模型长度"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False,
        help="是否信任远程代码"
    )

    # Ray 和 GPU 配置
    parser.add_argument(
        "--ray_address", type=str, default="auto",
        help="Ray 集群地址: 'auto', 'local', 或具体地址"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Tensor parallel 大小"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=None,
        help="使用的 GPU 数量"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="GPU 内存利用率"
    )
    parser.add_argument(
        "--allow_cross_node_tensor_parallel", action="store_true",
        help="允许跨节点 tensor parallel（不推荐）"
    )

    # 生成参数
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="采样温度"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Top-p 采样参数"
    )
    parser.add_argument(
        "--top_k", type=int, default=-1,
        help="Top-k 采样参数"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0,
        help="重复惩罚"
    )
    parser.add_argument(
        "--num_beams", type=int, default=None,
        help="Beam search 宽度（None 表示使用 sampling）"
    )
    parser.add_argument(
        "--num_return_sequences", type=int, default=1,
        help="返回的序列数量"
    )

    # 两阶段生成参数
    parser.add_argument(
        "--enable_thinking", action="store_true",
        help="启用两阶段生成（thinking + 生成）"
    )
    parser.add_argument(
        "--max_new_thinking_tokens", type=int, default=1000,
        help="Stage 1 thinking 最大 token 数"
    )
    parser.add_argument(
        "--num_return_thinking_sequences", type=int, default=1,
        help="Stage 1 返回的 thinking 候选数"
    )
    parser.add_argument(
        "--prompt_token", type=str, default=None,
        help="Stage 2 开始的提示 token（如 '<|sid_begin|>'）"
    )
    parser.add_argument(
        "--filter_empty_thinking", action="store_true",
        help="过滤掉 thinking 内容为空的生成结果（仅在两阶段生成时有效）"
    )
    parser.add_argument(
        "--skip_mfu_stats", action="store_true",
        help="跳过 MFU 统计计算（可避免某些边界情况下的错误）"
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=None,
        help="最大 prompt 长度（字符数），超过此长度的样本将被跳过"
    )

    # Chat template 配置
    parser.add_argument(
        "--use_chat_template", action="store_true",
        help="使用 tokenizer 的 chat template 处理输入 messages"
    )

    # 批处理配置
    parser.add_argument(
        "--batch_size", type=int, default=100,
        help="每批处理的 prompt 数量"
    )
    parser.add_argument(
        "--worker_batch_size", type=int, default=8,
        help="每个 worker 内部的批大小"
    )

    # 断点续传
    parser.add_argument(
        "--resume", action="store_true",
        help="从断点恢复（默认启用）"
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="不从断点恢复，重新开始"
    )

    return parser.parse_args()


def load_parquet_data(input_path: str, input_column: str) -> pd.DataFrame:
    """
    加载 Parquet 文件

    Args:
        input_path: Parquet 文件路径
        input_column: prompt 所在列名

    Returns:
        DataFrame
    """
    print(f"[Data] 加载数据: {input_path}")
    df = pd.read_parquet(input_path)

    if input_column not in df.columns:
        raise ValueError(f"列 '{input_column}' 不存在于输入文件中。可用列: {list(df.columns)}")

    # 重置索引，确保索引从 0 开始
    df = df.reset_index(drop=True)

    print(f"[Data] 总行数: {len(df)}")
    return df


def save_results(
    df: pd.DataFrame,
    results: Dict[int, List[str]],
    output_path: str,
    output_column: str,
    filter_empty_thinking: bool = False
):
    """
    保存结果到 Parquet 文件

    将多个生成序列展开为多行，每行包含：
    - 原始行的所有列（uuid, metadata, images, videos, source, segments, image, video, text, label, line_id）
    - 修改后的 messages 列（添加 assistant role）

    Args:
        df: 原始 DataFrame
        results: {index: [generated_text_1, generated_text_2, ...]} 生成结果
        output_path: 输出文件路径
        output_column: 输出列名（通常是 "messages"）
        filter_empty_thinking: 是否过滤掉 thinking 内容为空的生成结果
    """
    import json

    # 检测输入格式（是否为字符串）
    first_idx = next(iter(results.keys()))
    sample_messages = df.loc[first_idx, output_column]
    is_string_format = isinstance(sample_messages, str)

    # 构建新的 DataFrame 行列表
    new_rows = []
    filtered_count = 0

    for idx, generated_texts in results.items():
        # 获取原始行的所有数据
        original_row = df.loc[idx].to_dict()

        # 获取原始 messages（可能是字符串或列表）
        original_messages = original_row[output_column]
        if isinstance(original_messages, str):
            try:
                original_messages = json.loads(original_messages)
            except:
                # 如果解析失败，包装为列表
                original_messages = []

        # 为每个生成序列创建一行
        for generated_text in generated_texts:
            # 如果启用了过滤，检查 thinking 是否为空
            if filter_empty_thinking and _is_thinking_empty(generated_text):
                filtered_count += 1
                continue

            # 复制原始行的所有列
            new_row = original_row.copy()

            # 构建新的 messages：原始 messages + assistant role
            new_messages = original_messages.copy() if isinstance(original_messages, list) else []
            new_messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": generated_text}]
            })

            # 根据输入格式决定输出格式
            if is_string_format:
                # 输入是字符串，输出也转为字符串
                new_row[output_column] = json.dumps(new_messages, ensure_ascii=False)
            else:
                # 输入是列表，输出保持列表
                new_row[output_column] = new_messages

            new_rows.append(new_row)

    # 创建新的 DataFrame
    output_df = pd.DataFrame(new_rows)

    # 保存
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_df.to_parquet(output_path, index=False)
    print(f"[Data] 结果已保存: {output_path}")
    print(f"[Data] 输入行数: {len(df)}, 输出行数: {len(output_df)} (展开后)")
    if filter_empty_thinking and filtered_count > 0:
        print(f"[Data] 已过滤空 thinking 的序列数: {filtered_count}")


def process_batch(
    generator: RayVllmGenerator,
    prompts: Dict[str, str],
    generation_kwargs: dict
) -> Dict[int, List[str]]:
    """
    处理一批 prompt

    Args:
        generator: RayVllmGenerator 实例
        prompts: {sample_id: prompt_text}
        generation_kwargs: 生成参数字典

    Returns:
        {index: [generated_text_1, generated_text_2, ...]} (只返回生成的文本，不包含 prompt)
    """
    # 调用生成器（支持两阶段生成）
    results, logprobs = generator.generate(
        prompts=prompts,
        **generation_kwargs
    )

    # 只返回生成的文本（不拼接 prompt）
    output = {}
    prompt_token = generation_kwargs.get("prompt_token", None)

    for sample_id, generated_texts in results.items():
        idx = int(sample_id)

        # 添加 <|sid_end|> token（如果需要）
        if prompt_token == "<|sid_begin|>":
            output[idx] = [gen_text + "<|sid_end|>" for gen_text in generated_texts]
        else:
            output[idx] = generated_texts


    return output


def main():
    """主函数"""
    args = parse_args()

    # 1. 加载数据
    df = load_parquet_data(args.input_parquet, args.input_column)
    total_rows = len(df)

    # 2. 初始化断点续传管理器
    output_dir = Path(args.output_parquet).parent
    checkpoint_mgr = CheckpointManager(str(output_dir))

    # 处理 resume 参数
    if args.no_resume:
        checkpoint_mgr.reset()

    # 获取待处理的索引
    pending_indices = checkpoint_mgr.get_pending_indices(total_rows)
    progress = checkpoint_mgr.get_progress(total_rows)

    print(f"[Progress] 已完成: {progress['completed']}/{progress['total']} "
          f"({progress['progress_percent']}%)")
    print(f"[Progress] 待处理: {len(pending_indices)} 个 prompt")

    if not pending_indices:
        print("[Done] 所有 prompt 已处理完成")
        return

    # 2.5. 加载 tokenizer（如果需要 chat template）
    tokenizer = None
    if args.use_chat_template:
        print("\n[Tokenizer] 加载 tokenizer 用于 chat template...")
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
                trust_remote_code=args.trust_remote_code
            )
            print(f"[Tokenizer] ✓ Tokenizer 加载成功")
        except Exception as e:
            print(f"[Tokenizer] ✗ Tokenizer 加载失败: {e}")
            print("[Tokenizer] 将使用原始 messages 列内容作为 prompt")
            args.use_chat_template = False

    # 3. 初始化生成器
    print("\n[Model] 初始化 RayVllmGenerator...")
    generator = RayVllmGenerator(
        model_name_or_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        ray_address=args.ray_address,
        allow_cross_node_tensor_parallel=args.allow_cross_node_tensor_parallel,
        num_gpus=args.num_gpus,
        worker_batch_size=args.worker_batch_size,
        num_return_sequences=1,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty
    )

    # 4. 批量处理
    all_results = {}
    batch_size = args.batch_size
    num_batches = (len(pending_indices) + batch_size - 1) // batch_size

    print(f"\n[Generate] 开始生成，共 {num_batches} 批")

    try:
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(pending_indices))
            batch_indices = pending_indices[start:end]

            # 构建 prompts 字典
            prompts = {}
            for idx in batch_indices:
                raw_value = df.at[idx, args.input_column]

                if args.use_chat_template and tokenizer:
                    # 使用 chat template 处理
                    try:
                        # 解析 messages（如果是字符串）
                        if isinstance(raw_value, str):
                            import json
                            messages = json.loads(raw_value)
                        else:
                            messages = raw_value

                        # 转换消息格式
                        messages = _convert_messages_format(messages)

                        # 应用 chat template
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=args.enable_thinking,
                        )
                        prompts[str(idx)] = formatted_prompt
                    except Exception as e:
                        print(f"[Warning] Sample {idx}: chat template 处理失败 ({e}), 使用原始内容")
                        prompts[str(idx)] = raw_value if isinstance(raw_value, str) else str(raw_value)
                else:
                    # 直接使用原始内容
                    prompts[str(idx)] = raw_value if isinstance(raw_value, str) else str(raw_value)

            # 过滤掉过长的 prompts
            if args.max_prompt_length is not None:
                filtered_prompts = {}
                skipped_count = 0
                for sample_id, prompt in prompts.items():
                    if len(prompt) <= args.max_prompt_length:
                        filtered_prompts[sample_id] = prompt
                    else:
                        skipped_count += 1
                if skipped_count > 0:
                    print(f"[Warning] 跳过 {skipped_count} 个超过最大长度 ({args.max_prompt_length}) 的样本")
                prompts = filtered_prompts

            print(f"\n[Batch {batch_idx + 1}/{num_batches}] 处理 {len(batch_indices)} 个 prompt...")

            # 构建生成参数
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "num_return_sequences": args.num_return_sequences,
                "return_logprobs": False,  # 不需要 logprobs
            }

            # 添加 beam search 参数（如果指定）
            if args.num_beams is not None:
                generation_kwargs["num_beams"] = args.num_beams

            # 添加两阶段生成参数（如果启用）
            if args.enable_thinking:
                generation_kwargs["enable_thinking"] = True
                generation_kwargs["max_new_thinking_tokens"] = args.max_new_thinking_tokens
                generation_kwargs["num_return_thinking_sequences"] = args.num_return_thinking_sequences
                if args.prompt_token:
                    generation_kwargs["prompt_token"] = args.prompt_token

            # 添加 skip_mfu_stats 参数
            if args.skip_mfu_stats:
                generation_kwargs["skip_mfu_stats"] = True

            # 处理批次
            batch_results = process_batch(
                generator=generator,
                prompts=prompts,
                generation_kwargs=generation_kwargs
            )

            # 更新结果
            all_results.update(batch_results)

            # 标记完成并保存进度
            checkpoint_mgr.mark_completed(batch_indices)

            # 增量保存结果
            save_results(df, all_results, args.output_parquet, args.output_column, args.filter_empty_thinking)

            # 打印进度
            progress = checkpoint_mgr.get_progress(total_rows)
            print(f"[Progress] {progress['completed']}/{progress['total']} "
                  f"({progress['progress_percent']}%)")

    except KeyboardInterrupt as e:
        print(f"[Interrupt] 用户中断，保存当前进度... {e}")
        save_results(df, all_results, args.output_parquet, args.output_column, args.filter_empty_thinking)
        print("[Interrupt] 进度已保存，可使用 --resume 继续")
        return

    # 5. 清理资源
    print("\n[Cleanup] 释放 GPU 资源...")
    generator.cleanup()

    print("\n[Done] 全部完成!")
    print(f"[Done] 输出文件: {args.output_parquet}")


if __name__ == "__main__":
    main()
