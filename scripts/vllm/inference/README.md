# vLLM 批量生成接口

简单易用的 vLLM 批量文本生成接口，支持单模型和多模型并行评估，通过智能 GPU 调度充分利用硬件资源。

## 功能特性

- ✅ 支持单模型和多模型评估（统一接口）
- ✅ 智能 GPU 调度：自动分配空闲 GPU 给待评估模型
- ✅ 单机多卡并行：充分利用所有 GPU 资源
- ✅ 支持 tensor parallelism：单个模型使用多张卡
- ✅ 支持从 JSON 文件批量加载输入数据
- ✅ 支持自定义生成参数（temperature, top_p, max_tokens 等）
- ✅ 支持 PT checkpoint 加载
- ✅ 既可作为 Python 模块使用，也可作为命令行工具使用
- ✅ 所有结果汇总到单个输出文件

## 安装依赖

```bash
pip install vllm transformers
```

## 快速开始

### 命令行方式

#### 1. 准备输入 JSON 文件

创建一个 JSON 文件（例如 `input.json`），包含 `models`、`prompts` 和 `params` 字段：

**多模型评估：**
```json
{
  "models": [
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-14B"
  ],
  "prompts": [
    "请介绍一下人工智能的发展历程。",
    "什么是深度学习？请简要说明。"
  ],
  "params": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 128,
    "num_return_sequences": 2
  }
}
```

**单模型评估（models 列表只有一个元素）：**
```json
{
  "models": ["Qwen/Qwen2-7B"],
  "prompts": ["问题1", "问题2"],
  "params": {
    "temperature": 0.7,
    "max_new_tokens": 128
  }
}
```

**字段说明：**
- `models`: 模型路径列表（HuggingFace 模型名或本地路径）
- `prompts`: 输入文本列表
- `params`: 生成参数
  - `temperature`: 采样温度（0-2）
  - `top_p`: Nucleus sampling 参数（0-1）
  - `max_new_tokens`: 最大生成 token 数
  - `num_return_sequences`: 每个 prompt 生成的候选数量

#### 2. 运行生成

```bash
# 基本使用（自动检测所有 GPU）
python generate.py \
    --input_json input.json \
    --output output.json

# 指定可用的 GPU
python generate.py \
    --input_json input.json \
    --output output.json \
    --gpus 0 1 2 3

# 每个模型使用 2 张卡（tensor parallelism）
python generate.py \
    --input_json input.json \
    --output output.json \
    --tensor_parallel_size 2 \
    --gpus 0 1 2 3
```

#### 3. 查看输出

输出文件（`output.json`）格式：

```json
{
  "results": {
    "Qwen_Qwen2-7B": [
      {
        "prompt": "请介绍一下人工智能的发展历程。",
        "generated_texts": ["结果1", "结果2"]
      },
      {
        "prompt": "什么是深度学习？请简要说明。",
        "generated_texts": ["结果1", "结果2"]
      }
    ],
    "Qwen_Qwen2-14B": [...]
  },
  "params": {...},
  "models": [...]
}
```

### Python 模块方式

#### 多模型评估

```python
from scripts.vllm.interface import MultiModelEvaluator

# 创建评估器
evaluator = MultiModelEvaluator(
    available_gpus=[0, 1, 2, 3],  # 可用的 GPU
    default_tensor_parallel_size=1,  # 每个模型使用的 GPU 数
    default_dtype="bfloat16"
)

# 方式 1：直接评估
results = evaluator.evaluate(
    models=["Qwen/Qwen2-7B", "Qwen/Qwen2-14B"],
    prompts=["问题1", "问题2"],
    params={"temperature": 0.7, "max_new_tokens": 128}
)

# 方式 2：从 JSON 文件评估
results = evaluator.evaluate_from_json(
    input_json_path="input.json",
    output_path="output.json"
)
```

#### 单模型评估

对于单模型评估，可以使用 `VllmInferenceInterface`（更轻量）：

```python
from scripts.vllm.interface import VllmInferenceInterface

# 初始化接口（模型会被加载并可复用）
interface = VllmInferenceInterface(
    model_path="Qwen/Qwen2-7B",
    tensor_parallel_size=1
)

# 直接传入数据
results = interface.generate(
    prompts=["问题1", "问题2"],
    temperature=0.7,
    max_new_tokens=128,
    num_return_sequences=2
)
# 返回：{"0": ["答案1", "答案2"], "1": ["答案1", "答案2"]}
```

或者使用 `MultiModelEvaluator`（models 列表只有一个元素，自动处理）：

```python
from scripts.vllm.interface import MultiModelEvaluator

evaluator = MultiModelEvaluator()
results = evaluator.evaluate(
    models=["Qwen/Qwen2-7B"],  # 单个模型
    prompts=["问题1", "问题2"],
    params={"temperature": 0.7}
)
```

#### 便捷函数

```python
from scripts.vllm.interface import evaluate_models_from_file

# 一行代码完成多模型评估
results = evaluate_models_from_file(
    input_json_path="input.json",
    output_path="output.json",
    available_gpus=[0, 1, 2, 3]
)
```

## GPU 调度策略

多模型评估系统采用智能 GPU 调度策略：

1. **自动检测 GPU**：如果不指定 `--gpus`，自动检测所有可用 GPU
2. **动态分配**：当有空闲 GPU 时，自动分配给待评估模型
3. **并行执行**：多个模型可以同时在不同 GPU 上运行
4. **资源释放**：模型评估完成后自动释放 GPU，供下一个模型使用

### 示例场景

**场景 1：4 张 GPU，评估 3 个模型，每个模型使用 1 张卡**
- GPU 0-2 同时运行前 3 个模型
- GPU 3 空闲
- 最快完成的模型释放 GPU 后，GPU 可用于其他任务

**场景 2：4 张 GPU，评估 3 个模型，每个模型使用 2 张卡**
- GPU 0-1 运行第 1 个模型
- GPU 2-3 运行第 2 个模型
- 第 1 个模型完成后，GPU 0-1 运行第 3 个模型

**场景 3：单模型评估，使用 2 张卡（tensor parallelism）**
```bash
python generate.py \
    --input_json input.json \
    --output output.json \
    --tensor_parallel_size 2 \
    --gpus 0 1
```

## 命令行参数

- `--input_json`: 输入 JSON 文件路径（必需）
- `--output`: 输出文件路径（必需）
- `--gpus`: 可用的 GPU ID 列表（可选，例如 `--gpus 0 1 2 3`）
- `--tensor_parallel_size`: 每个模型使用的 GPU 数量（默认 1）
- `--dtype`: 模型数据类型（默认 bfloat16）
- `--gpu_memory_utilization`: GPU 内存利用率（默认 0.9）

## 示例文件

项目中提供了示例输入文件 `example_input.json`，可以直接使用：

```bash
python generate.py \
    --input_json example_input.json \
    --output example_output.json \
    --gpus 0 1 2 3
```

## 性能优势

### vLLM vs HuggingFace Transformers

相比 HuggingFace Transformers，vLLM 具有以下优势：
- 更快的推理速度（通过 PagedAttention 和 Continuous Batching）
- 更高的吞吐量
- 更好的 GPU 内存利用率

对于批量生成任务，vLLM 的速度通常是 HuggingFace Transformers 的 **2-5 倍**。

### 多模型并行 vs 顺序评估

| 场景 | 顺序评估 | 并行评估 | 加速比 |
|------|---------|---------|--------|
| 4 GPU，4 模型，每模型 1 GPU | 4x 时间 | 1x 时间 | **4x** |
| 4 GPU，4 模型，每模型 2 GPU | 4x 时间 | 2x 时间 | **2x** |
| 8 GPU，8 模型，每模型 1 GPU | 8x 时间 | 1x 时间 | **8x** |

## 注意事项

1. **GPU 内存**：确保每张 GPU 有足够内存加载模型
2. **并行数量**：最大并行模型数 = GPU 总数 / 每个模型需要的 GPU 数
3. **错误处理**：如果某个模型评估失败，其他模型继续运行
4. **输入格式**：所有输入 JSON 必须包含 `models` 字段（列表格式）
5. **单模型评估**：`models` 列表只包含一个元素即可

## 故障排查

### 问题 1：CUDA Out of Memory

**解决方案：**
- 降低 `--gpu_memory_utilization`（例如从 0.9 降到 0.8）
- 减少 `num_return_sequences`
- 增加 `tensor_parallel_size`（使用更多 GPU）

### 问题 2：生成结果质量不佳

**解决方案：**
- 调整 `temperature`（降低以获得更确定的输出）
- 调整 `top_p` 和 `top_k`
- 增加 `repetition_penalty` 以减少重复

### 问题 3：找不到 GPU

**解决方案：**
- 检查 CUDA 是否正确安装：`nvidia-smi`
- 检查 PyTorch 是否支持 CUDA：`python -c "import torch; print(torch.cuda.is_available())"`
- 使用 `--gpus` 参数明确指定 GPU

### 问题 4：模型评估失败

**解决方案：**
- 查看错误信息中的详细堆栈
- 确认模型路径正确（HuggingFace 模型名或本地路径）
- 确认有足够的磁盘空间和内存

## 文件结构

```
scripts/vllm/
├── interface.py              # 核心接口（包含 VllmInferenceInterface 和 MultiModelEvaluator）
├── generate.py               # 命令行工具
├── example_input.json        # 示例输入文件
├── utils/
│   ├── generator.py          # VllmGenerator 基础类
│   └── arguments.py          # 参数定义
└── README.md                 # 本文档
```

## 更多资源

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [HuggingFace Models](https://huggingface.co/models)
