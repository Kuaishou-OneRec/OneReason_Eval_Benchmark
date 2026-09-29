#!/bin/bash

# Ray + vLLM 多机多卡推理启动脚本

# ============================================================================
# 配置参数（请根据实际情况修改）
# ============================================================================

# 输入输出
INPUT_PARQUET="./data/input.parquet"
OUTPUT_PARQUET="./data/output.parquet"

# 模型配置
MODEL_PATH="./models/model"  # 修改为实际模型路径
# CHECKPOINT_PATH="/path/to/checkpoint.pt"  # 如果有 PT checkpoint，取消注释

# Ray 配置
RAY_ADDRESS="auto"  # auto: 自动检测集群, local: 单机模式, 或具体地址如 ray://head_ip:10001
TENSOR_PARALLEL_SIZE=1  # 每个模型使用的 GPU 数量
# NUM_GPUS=8  # 限制使用的 GPU 数量（可选）
GPU_MEMORY_UTILIZATION=0.9

# 生成参数
MAX_NEW_TOKENS=3
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=-1
REPETITION_PENALTY=1.0

# 生成模式配置
GENERATION_MODE="two_stage"  # sampling, beam_search, two_stage

# Beam Search 参数（仅当 GENERATION_MODE=beam_search 或 two_stage 时使用）
NUM_BEAMS=8
NUM_RETURN_SEQUENCES=64

# 两阶段生成参数（仅当 GENERATION_MODE=two_stage 时使用）
ENABLE_THINKING=true
MAX_NEW_THINKING_TOKENS=1024
NUM_RETURN_THINKING_SEQUENCES=8
PROMPT_TOKEN="<|sid_begin|>"
FILTER_EMPTY_THINKING=true  # 是否过滤掉 thinking 内容为空的生成结果
SKIP_MFU_STATS=true  # 是否跳过 MFU 统计计算
MAX_PROMPT_LENGTH=20000  # 最大 prompt 长度（字符数），超过此长度的样本将被跳过

# Chat Template 配置
USE_CHAT_TEMPLATE=true  # 是否使用 chat template 处理输入

# 批处理参数
BATCH_SIZE=1000
WORKER_BATCH_SIZE=1000

# ============================================================================
# 运行脚本
# ============================================================================

# 构建基础命令
CMD="python3 generate.py \
    --input_parquet \"$INPUT_PARQUET\" \
    --output_parquet \"$OUTPUT_PARQUET\" \
    --model_path \"$MODEL_PATH\" \
    --ray_address \"$RAY_ADDRESS\" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --repetition_penalty $REPETITION_PENALTY \
    --batch_size $BATCH_SIZE \
    --worker_batch_size $WORKER_BATCH_SIZE"

# 根据生成模式添加参数
if [ "$GENERATION_MODE" = "beam_search" ] || [ "$GENERATION_MODE" = "two_stage" ]; then
    CMD="$CMD --num_beams $NUM_BEAMS --num_return_sequences $NUM_RETURN_SEQUENCES"
fi

if [ "$GENERATION_MODE" = "two_stage" ]; then
    CMD="$CMD --enable_thinking --max_new_thinking_tokens $MAX_NEW_THINKING_TOKENS"
    CMD="$CMD --num_return_thinking_sequences $NUM_RETURN_THINKING_SEQUENCES"
    if [ -n "$PROMPT_TOKEN" ]; then
        CMD="$CMD --prompt_token \"$PROMPT_TOKEN\""
    fi
    if [ "$FILTER_EMPTY_THINKING" = "true" ]; then
        CMD="$CMD --filter_empty_thinking"
    fi
    if [ "$SKIP_MFU_STATS" = "true" ]; then
        CMD="$CMD --skip_mfu_stats"
    fi
fi

# 添加 Chat Template 参数
if [ "$USE_CHAT_TEMPLATE" = "true" ]; then
    CMD="$CMD --use_chat_template"
fi

# 添加最大 prompt 长度限制
if [ -n "$MAX_PROMPT_LENGTH" ]; then
    CMD="$CMD --max_prompt_length $MAX_PROMPT_LENGTH"
fi

mkdir -p cot_logs/
# 执行命令并保存日志
LOG_FILE="$(date +%Y%m%d_%H%M%S).log"
echo "日志将保存到: cot_logs/$LOG_FILE"
eval $CMD 2>&1 | tee cot_logs/"$LOG_FILE"
