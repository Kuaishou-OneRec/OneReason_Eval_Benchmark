#!/bin/bash

# vLLM 批量生成快速运行脚本
# 使用方法：
#   ./run.sh                              # 使用默认配置
#   ./run.sh input.json output.json      # 指定输入输出
#   ./run.sh input.json output.json 0,1  # 指定输入输出和GPU

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 获取项目根目录（向上 3 级）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 默认配置
INPUT_JSON="${1:-example_input.json}"
OUTPUT_JSON="${2:-output.json}"
GPUS="${3:-}"

# 检查输入文件
if [[ ! -f "$SCRIPT_DIR/$INPUT_JSON" ]]; then
    echo "错误: 输入文件不存在: $INPUT_JSON"
    exit 1
fi

# 构建命令 - 直接运行脚本
CMD="python3 $SCRIPT_DIR/generate.py --input_json $SCRIPT_DIR/$INPUT_JSON --output $SCRIPT_DIR/$OUTPUT_JSON --gpus 0 1"

# 添加 GPU 参数
# if [[ -n "$GPUS" ]]; then
#     GPU_ARRAY=(${GPUS//,/ })
#     CMD="$CMD --gpus ${GPU_ARRAY[@]}"
# fi

# 显示配置
echo "输入: $INPUT_JSON"
echo "输出: $OUTPUT_JSON"
[[ -n "$GPUS" ]] && echo "GPU: $GPUS"
echo ""

echo "$CMD"
# 执行
eval "$CMD"
