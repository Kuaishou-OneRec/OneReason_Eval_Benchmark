# Ray + vLLM 多机多卡批量推理

基于 Ray 和 vLLM 的多机多卡批量推理工具，支持断点续传。

## 依赖安装

```bash
pip install filelock pandas pyarrow
```

## 功能特性

- 输入/输出：Parquet 文件格式
- 支持跨机 Tensor Parallel
- 按 prompt 粒度断点续传
- 自动连接已有 Ray 集群
- 容错和自动重试

## 快速开始

### 基本用法

```bash
python generate.py \
    --input_parquet input.parquet \
    --output_parquet output.parquet \
    --model_path /path/to/model
```

### 多机多卡（Tensor Parallel）

```bash
python generate.py \
    --input_parquet input.parquet \
    --output_parquet output.parquet \
    --model_path /path/to/model \
    --tensor_parallel_size 4 \
    --ray_address auto
```

## 输入输出格式

### 输入

Parquet 文件，包含 `messages` 列（可通过 `--input_column` 指定其他列名）：

| messages |
|----------|
| "请介绍人工智能" |
| "什么是深度学习" |

### 输出

Parquet 文件，`messages` 列存储 prompt + 回答拼接：

| messages |
|----------|
| "请介绍人工智能\n\n人工智能是..." |
| "什么是深度学习\n\n深度学习是..." |

## 命令行参数

### 输入输出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_parquet` | 必填 | 输入 Parquet 文件路径 |
| `--output_parquet` | 必填 | 输出 Parquet 文件路径 |
| `--input_column` | messages | 输入列名 |
| `--output_column` | messages | 输出列名 |

### 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 必填 | 模型路径或 HuggingFace 模型名 |
| `--checkpoint_path` | None | PT checkpoint 路径 |
| `--dtype` | bfloat16 | 模型数据类型 |
| `--max_model_len` | None | 最大模型长度 |

### Ray 和 GPU 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ray_address` | auto | Ray 集群地址 |
| `--tensor_parallel_size` | 1 | Tensor parallel 大小 |
| `--num_gpus` | None | 使用的 GPU 数量 |
| `--gpu_memory_utilization` | 0.9 | GPU 内存利用率 |

### 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_new_tokens` | 512 | 最大生成 token 数 |
| `--temperature` | 0.7 | 采样温度 |
| `--top_p` | 0.9 | Top-p 采样参数 |
| `--top_k` | -1 | Top-k 采样参数 |
| `--repetition_penalty` | 1.0 | 重复惩罚 |

### 批处理和断点续传

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 100 | 每批处理的 prompt 数量 |
| `--worker_batch_size` | 8 | 每个 worker 内部批大小 |
| `--no_resume` | False | 不从断点恢复，重新开始 |

## 断点续传

任务中断后会自动保存进度到输出目录的 `progress.json` 文件。

### 恢复任务

```bash
# 默认自动恢复
python generate.py --input_parquet input.parquet --output_parquet output.parquet --model_path /path/to/model

# 强制重新开始
python generate.py --input_parquet input.parquet --output_parquet output.parquet --model_path /path/to/model --no_resume
```
