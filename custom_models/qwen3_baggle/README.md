# Qwen3 Baggle 自定义模型

这是一个基于 Qwen3 的自定义模型实现，添加了额外的 item token 支持。

## 文件结构

```
custom_models/qwen3_baggle/
├── __init__.py              # 模块初始化和导出
├── configuration_qwen3.py   # 配置类（Qwen3Config 和 Qwen3BaggleConfig）
├── modeling_qwen3.py        # 模型实现
├── modular_qwen3.py         # 模块化实现
└── README.md               # 本文件
```

## 使用方法

### 方法 1: 通过 trust_remote_code 使用（推荐）

如果你的模型权重目录中包含了这些自定义文件，可以直接使用：

```bash
# 在你的模型权重目录中，应该包含：
# - config.json
# - model.safetensors (或其他权重文件)
# - configuration_qwen3.py
# - modeling_qwen3.py
# - __init__.py

python scripts/hf-transformers/evaluate.py \
    --model_path /path/to/your/qwen3_baggle_checkpoint \
    --data_dir ./data \
    --output_dir ./results
```

`HfTransformersGenerator` 默认使用 `trust_remote_code=True`，会自动加载模型目录中的自定义代码。

### 方法 2: 在项目中直接导入

如果你想在代码中直接使用这个模型，首先确保已安装项目：

```bash
# 在项目根目录安装（开发模式）
pip install -e .
```

然后可以直接导入：

```python
from custom_models.qwen3_baggle import Qwen3BaggleConfig, Qwen3ForCausalLM
from transformers import AutoTokenizer

# 加载配置
config = Qwen3BaggleConfig.from_pretrained("/path/to/checkpoint")

# 加载模型
model = Qwen3ForCausalLM.from_pretrained(
    "/path/to/checkpoint",
    config=config,
    torch_dtype="auto"
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoint")
```

### 方法 3: 注册为 AutoModel（高级用法）

如果你想让 `AutoConfig` 和 `AutoModelForCausalLM` 自动识别你的模型：

```python
from transformers import AutoConfig, AutoModelForCausalLM
from custom_models.qwen3_baggle import Qwen3BaggleConfig, Qwen3ForCausalLM

# 注册自定义配置和模型
AutoConfig.register("qwen3_baggle", Qwen3BaggleConfig)
AutoModelForCausalLM.register(Qwen3BaggleConfig, Qwen3ForCausalLM)

# 现在可以使用 Auto 类加载
config = AutoConfig.from_pretrained("/path/to/checkpoint", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/path/to/checkpoint", trust_remote_code=True)
```

## 配置说明

`Qwen3BaggleConfig` 继承自 `Qwen3Config`，并添加了以下参数：

- `item_token_start`: item token 的起始 ID
- `item_token_end`: item token 的结束 ID

示例配置文件 (`config.json`):

```json
{
  "model_type": "qwen3_baggle",
  "vocab_size": 151936,
  "hidden_size": 4096,
  "intermediate_size": 22016,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 32,
  "head_dim": 128,
  "item_token_start": 151000,
  "item_token_end": 151100,
  ...
}
```

## 在评估脚本中使用

### 使用 hf-transformers 脚本

```bash
# 评估所有任务
python scripts/hf-transformers/evaluate.py \
    --model_path /path/to/qwen3_baggle_checkpoint \
    --data_dir ./data \
    --output_dir ./results \
    --batch_size 4 \
    --dtype bfloat16

# 评估特定任务
python scripts/hf-transformers/evaluate.py \
    --model_path /path/to/qwen3_baggle_checkpoint \
    --data_dir ./data \
    --output_dir ./results \
    --task_types math_500 gsm8k
```

### 使用 PT checkpoint

如果你有 `.pt` 格式的 checkpoint：

```bash
python scripts/hf-transformers/evaluate.py \
    --model_path /path/to/model/config \
    --checkpoint_path /path/to/checkpoint.pt \
    --torch_dtype bfloat16 \
    --data_dir ./data \
    --output_dir ./results \
    --batch_size 4
```

## 注意事项

1. **trust_remote_code**: 使用自定义模型时，确保设置 `trust_remote_code=True`
2. **模型文件位置**: 如果使用方法 1，需要将 `configuration_qwen3.py`、`modeling_qwen3.py` 等文件复制到你的模型权重目录中
3. **config.json**: 确保 `config.json` 中的 `model_type` 设置为 `"qwen3_baggle"`
4. **项目安装**: 如果使用方法 2 或 3，确保已通过 `pip install -e .` 安装项目

## 故障排查

### 问题 1: 找不到模块

```
ModuleNotFoundError: No module named 'custom_models'
```

**解决方案**: 确保已在项目根目录安装项目：

```bash
pip install -e .
```

### 问题 2: 无法加载自定义模型

```
ValueError: Unrecognized configuration class
```

**解决方案**: 
1. 检查 `config.json` 中的 `model_type` 是否为 `"qwen3_baggle"`
2. 确保使用 `trust_remote_code=True`
3. 或者手动注册模型（见方法 3）

### 问题 3: 权重加载失败

**解决方案**: 
1. 确保权重文件格式正确（`.safetensors` 或 `.bin`）
2. 检查配置文件中的参数是否与权重匹配
3. 使用 `--checkpoint_path` 参数加载 PT checkpoint

## 开发说明

- `modeling_qwen3.py` 是从 `modular_qwen3.py` 自动生成的
- 如需修改，请编辑 `modular_qwen3.py` 并重新生成
- 配置类在 `configuration_qwen3.py` 中定义

