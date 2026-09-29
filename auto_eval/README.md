> 比赛默认配置：datav3.1、seed 42、Race、nothink；任务列表仅包含当前 11 个比赛任务。推荐任务的 Race 内部仍包含 think 分支，详见根目录 README。

# OneReason Eval Benchmark Auto Evaluation Web Interface

模块化的 Gradio Web 应用,用于 OneReason Eval Benchmark 的自动评估任务提交和结果分析。

## 项目结构

```
auto_eval/
├── web.py                  # 主入口文件
├── config.py               # 共享配置
├── utils.py                # 共享工具函数
├── submit_cli.py           # CLI 评估任务提交工具
├── README.md               # 项目文档
├── icon/                   # SVG 图标
│   ├── submit.svg          # 提交任务图标
│   ├── analyze.svg         # 结果分析图标
│   └── transform.svg       # PID 提取图标
├── submit/                 # 提交任务模块
│   ├── __init__.py
│   ├── core.py             # 任务构建与写队列核心逻辑（Web UI 与 CLI 共用）
│   ├── handlers.py         # 业务逻辑
│   ├── ui.py               # UI 组件
│   └── styles.py           # CSS/JS 样式
├── analyze/                # 结果分析模块
│   ├── __init__.py
│   ├── handlers.py         # 业务逻辑
│   ├── ui.py               # UI 组件
│   ├── plot.py             # 雷达图和表格生成
│   └── styles.py           # CSS/JS 样式
├── extract_pid/            # PID 提取模块
│   ├── __init__.py
│   ├── handlers.py         # SID 提取和转换逻辑
│   ├── ui.py               # UI 组件
│   ├── styles.py           # CSS/JS 样式
│   └── README.md           # 模块文档
├── executor/               # 任务执行模块
│   ├── __init__.py
│   ├── eval_script.py      # 评估脚本包装器
│   ├── eval_script.sh      # Shell 评估脚本
│   └── sentinel_worker.py  # 后台任务监控进程
├── incoming/               # 等待处理的任务
├── processing/             # 正在运行的任务
├── done/                   # 已完成的任务
├── failed/                 # 失败的任务
└── plots/                  # 生成的图表缓存
```

## 配置说明

### 路径配置

所有服务器路径都通过 `config.py` 中的 `BASE_BENCHMARK_DIR` 统一管理：

```python
# 在 config.py 中修改基础路径
BASE_BENCHMARK_DIR = Path("/path/to/resource")  # 默认值

# 其他路径自动派生：
# - RESULTS_BASE_DIR = BASE_BENCHMARK_DIR / "results" / VERSION
# - QUEUE_BASE_DIR = BASE_BENCHMARK_DIR / "auto_eval" / VERSION
# - LOG_BASE_DIR = BASE_BENCHMARK_DIR / "auto_eval_logs" / VERSION
# - DATA_DIR = BASE_BENCHMARK_DIR / "data_{DATA_VERSION}"
# - GENERATION_BASE_DIR = BASE_BENCHMARK_DIR / "generation_case"
```

**如需更换用户或目录**，只需修改 `config.py` 中的 `BASE_BENCHMARK_DIR`，所有其他路径会自动更新。

## 功能特性

### 1. 任务提交 (Submit)

- **路径验证**: 实时检查模型路径是否存在
- **输出后缀生成**: 自动从模型路径提取 3 个目录生成后缀
- **预览编辑**: 支持手动修改 output_suffix,自动更新派生路径
- **任务队列管理**:
  - 4 个队列:等待中、运行中、已完成、已失败
  - 批量删除任务(等待中/已完成/已失败)
  - 自动刷新(每 10 分钟)
- **CLI 自动化提交**: `python -m auto_eval.submit_cli` 供训练脚本在 checkpoint 产出后自动写入 `incoming/` 队列,无需打开 Web UI

### 2. 结果分析 (Analyze)

- **目录选择**: 从 `{RESULTS_BASE_DIR}` (默认: `/path/to/resource`) 选择多个结果目录
- **雷达图生成**:
  - 一个图多条线,每条线代表一个模型
  - 维度为数据集名称(如 math_500, gsm8k, ...)
  - 归一化显示,附带原始值范围
- **指标表格**: 显示所有数据集的详细指标
- **样本对比**:
  - 点击表格"查看"列查看具体数据集的样本
  - 随机选择最多 10 个共同样本
  - 并排展示不同模型的生成结果、正确性等

### 3. PID 提取 (Extract PID)

- **SID 提取**: 从文本中提取 `<|sid_begin|><s_a_X><s_b_Y><s_c_Z><|sid_end|>` 格式的 SID 代码
- **PID 转换**: 将提取的 SID 转换为对应的 PID (基于 parquet 文件映射)
- **批量处理**:
  - 初始 5 个输入框，可动态添加/删除
  - 每个输入框对应一个输出结果
- **展开/收起**: 输出结果固定高度，超出部分可展开查看
- **复制功能**: 每个输出框右上角有复制按钮，点击复制 PID 列表
- **仅服务器**: 功能仅在服务器环境可用（需要访问 parquet 文件）

## 配置说明

### config.py

- **路径配置**:
  - `RESULTS_DIR`: 结果目录路径(本地测试时使用 `test_results/`)
  - `QUEUE_DIRS`: 任务队列目录

- **任务设置**:
  - `MAX_RETRIES`: 任务最大重试次数(默认 3)
  - `EMAIL_DOMAIN`: 邮箱域名(默认 example.org)

- **UI 设置**:
  - `MAX_RECENT_TASKS`: 最大显示最近任务数(默认 20)
  - `AUTO_REFRESH_INTERVAL`: 自动刷新间隔(默认 600 秒)

- **样本对比设置**:
  - `MAX_SAMPLES_TO_COMPARE`: 最大对比样本数(默认 10)

## 运行方式

### 服务器部署

```bash
# 默认路径
cd auto_eval
python3 web.py

# 如需修改基础路径，请编辑 config.py 中的 BASE_BENCHMARK_DIR
```

应用将在 `http://0.0.0.0:8890` 启动。

### 本地开发

```bash
cd auto_eval
python3 web.py
```

本地运行时会使用 `auto_eval/test_results/` 作为结果目录。

### CLI 自动提交评估任务

训练脚本在 checkpoint 产出后，可直接调用 `python -m auto_eval.submit_cli` 将评估任务写入共享 `incoming/` 队列，无需人工干预。

**参数优先级：命令行 > 环境变量 > Python 默认值**

**dry-run（只打印 JSON，不写队列）**

```bash
python -m auto_eval.submit_cli \
  --model-path /path/to/ckpt/global_step_1000/actor \
  --dry-run
```

**真实提交（写入 incoming/ 队列）**

不传其他参数时，会使用环境变量或 Python 默认值补齐 `version`、`tasks`、`rollout_mode` 等参数。

```bash
python -m auto_eval.submit_cli \
  --model-path /path/to/ckpt/global_step_1000/actor
```

**通过环境变量提交（适合 CI / 训练脚本）**

```bash
export AUTO_EVAL_VERSION=v3.1
export AUTO_EVAL_USER=example_user
export AUTO_EVAL_TASKS=all
export AUTO_EVAL_ROLLOUT_MODE=Default

python -m auto_eval.submit_cli --model-path /path/to/ckpt/global_step_1000/actor
```

**在训练脚本中集成**

```python
import os
import subprocess
import sys

def submit_eval(ckpt_path: str):
    env = os.environ.copy()
    env.update({
        "AUTO_EVAL_VERSION": "v3.1",
        "AUTO_EVAL_USER": "example_user",
        "AUTO_EVAL_TASKS": "all",
        "AUTO_EVAL_ROLLOUT_MODE": "Default",
    })
    subprocess.run(
        [
            sys.executable, "-m", "auto_eval.submit_cli",
            "--model-path", ckpt_path,
        ],
        env=env,
        check=True,
    )
```

**参数说明**

| 参数 | 对应环境变量 | 默认值 | 说明 |
|------|------------|--------|------|
| `--model-path` | — | 必填，无默认 | checkpoint 路径，CLI **不校验**路径是否存在，请自行确认 |
| `--version` | `AUTO_EVAL_VERSION` | `v3.1` | benchmark 版本 |
| `--tasks` | `AUTO_EVAL_TASKS` | `all` | 评估任务，逗号分隔任务名 |
| `--output-suffix` | `AUTO_EVAL_OUTPUT_SUFFIX` | 自动推导 | 不填时从模型路径末尾若干目录拼接生成 |
| `--rollout-mode` | `AUTO_EVAL_ROLLOUT_MODE` | `Default` | 推理模式（如 `Default`、`Final`、`Normal 128`、`Think 64`） |
| `--enable-thinking` / `--no-enable-thinking` | `AUTO_EVAL_ENABLE_THINKING` | 不启用 | 布尔环境变量仅接受 `1`/`true`（启用）或 `0`/`false`（禁用），大小写不敏感，其他值报错 |
| `--seed` | `AUTO_EVAL_SEED` | `42` | 随机种子，非 42 时会体现在自动生成的 `output-suffix` 中 |
| `--user` | `AUTO_EVAL_USER` | 当前系统用户（`$USER` / `$USERNAME`） | 未传 `--email` 时，自动补 `@example.org` 作为邮箱 |
| `--email` | `AUTO_EVAL_EMAIL` | 由 `--user` 自动生成 | 提交者邮箱，优先级高于 `--user` |
| `--overwrite` / `--no-overwrite` | `AUTO_EVAL_OVERWRITE` | 关闭 | 覆盖已有结果；`--no-overwrite` 可显式覆盖 `AUTO_EVAL_OVERWRITE=true`；布尔环境变量同上 |
| `--queue-dir` | `AUTO_EVAL_QUEUE_DIR` | `/path/to/resource` | 与 Web UI 提交共用同一 `incoming/` 队列；可显式指定目标目录 |
| `--dry-run` | — | 关闭 | 只打印任务 JSON，不写队列 |

**布尔环境变量说明**

`AUTO_EVAL_ENABLE_THINKING` 和 `AUTO_EVAL_OVERWRITE` 只接受以下值（大小写不敏感）：

| 值 | 含义 |
|----|------|
| `1` 或 `true` | 启用 |
| `0` 或 `false` | 禁用 |
| 其他值 | 报错退出，不静默猜测 |

**注意事项**

- 默认写入普通共享 `incoming/` 队列，与 Web UI 提交共用同一队列。
- CLI 不校验 `--model-path` 是否存在，路径填错不会报错，请自行确认。
- 评估结果可在 Web UI 的结果分析页面查看。

## 依赖项

```
gradio>=4.0.0
pandas
numpy
matplotlib
```

## 页面导航

### 左侧导航栏

- **深灰色背景** (#2c2c2c)
- **浅灰色按钮** (默认 #4a4a4a)
- **白色按钮** (选中状态)
- **SVG 图标**:
  - 默认白色填充
  - 选中时红色填充 (#d81e06)

### 提交任务页面

- 新建评估任务
- 用户名自动补全 @example.org
- 模型路径输入(支持多行)
- 实时预览表格
- 任务队列状态(4 个 Tab)

### 结果分析页面

- 左侧:目录选择(checkbox)
- 右侧:
  - 雷达图(上方)
  - 指标表格(下方,带"查看"列)
  - 点击"查看"打开模态窗口显示样本对比

## 模块说明

### executor 模块

后台任务执行模块,不直接与 Web 界面交互。

- **sentinel_worker.py**:
  - 后台守护进程,监控任务队列
  - 从 `incoming/` 目录获取任务
  - 移动任务到 `processing/` 并调用 `eval_script.py`
  - 根据执行结果移动到 `done/` 或 `failed/`
  - 支持任务重试(最多 3 次)

- **eval_script.py**:
  - 读取任务 JSON 文件
  - 调用 `eval_script.sh` 执行实际评估
  - 捕获输出和错误日志
  - 返回执行状态码

- **eval_script.sh**:
  - Bash 脚本,执行实际的评估命令
  - 分 3 个任务组执行不同的评估任务
  - 使用不同的 GPU 内存配置和批次大小

**运行方式**:
```bash
# 启动后台 sentinel worker（默认路径）
cd auto_eval
python3 executor/sentinel_worker.py

# 如需修改基础路径，请编辑 config.py 中的 BASE_BENCHMARK_DIR
```

### submit 模块

- **core.py**:
  - `build_eval_task()`: 构建任务 JSON dict（Web UI 与 CLI 共用）
  - `submit_eval_task()`: 将任务写入指定队列目录
  - `generate_output_suffix()`: 从模型路径推导输出后缀
  - `normalize_tasks()`: 统一规范化任务列表

- **handlers.py**:
  - `validate_email()`: 邮箱验证
  - `generate_output_suffix()`: 生成输出后缀
  - `parse_paths_to_preview()`: 解析路径生成预览
  - `submit_tasks()`: 提交任务到队列
  - `get_queue_status()`: 获取队列状态
  - `delete_selected_tasks()`: 删除选中任务

- **ui.py**:
  - `create_submit_ui()`: 创建提交页面 UI

- **styles.py**:
  - `SUBMIT_CSS`: 提交页面 CSS
  - `SUBMIT_JS`: 提交页面 JavaScript(复选框、列宽控制等)

### submit_cli.py

CLI 入口，供训练脚本在 checkpoint 产出后自动提交评估任务，通过 `python -m auto_eval.submit_cli` 调用。复用 `submit/core.py` 的构建与写队列逻辑，不校验模型路径是否存在。详细用法见"运行方式 - CLI 自动提交评估任务"章节。

### analyze 模块

- **handlers.py**:
  - `get_result_directories()`: 获取结果目录列表
  - `load_eval_results()`: 加载 eval_results.json
  - `load_test_generated_samples()`: 加载 test_generated.json
  - `compare_samples()`: 对比样本
  - `format_comparison_for_display()`: 格式化对比结果为 HTML

- **plot.py**:
  - `generate_radar_chart_and_table()`: 生成雷达图和表格(主函数)
  - `create_radar_chart()`: 创建雷达图
  - `create_summary_table()`: 创建汇总表格

- **ui.py**:
  - `create_analyze_ui()`: 创建分析页面 UI
  - `refresh_directories()`: 刷新目录列表
  - `generate_analysis()`: 生成分析报告
  - `show_sample_comparison()`: 显示样本对比模态窗口

- **styles.py**:
  - `ANALYZE_CSS`: 分析页面 CSS(包括模态窗口样式)
  - `ANALYZE_JS`: 分析页面 JavaScript

## 扩展指南

### 添加新页面

1. 在 `auto_eval/` 下创建新模块目录(如 `new_feature/`)
2. 创建 `__init__.py`, `handlers.py`, `ui.py`, `styles.py`
3. 在 `web.py` 中导入并集成
4. 在左侧导航栏添加新按钮

### 修改样式

- **全局样式**: 修改 `web.py` 中的 `COMBINED_CSS`
- **模块样式**: 修改对应模块的 `styles.py`

### 修改配置

编辑 `config.py` 中的配置项。

## 注意事项

1. **路径配置**: 确保 `RESULTS_DIR` 路径正确并有访问权限
2. **队列目录**: 提交机必须能访问队列目录
3. **图标文件**: 确保 `icon/` 目录下有 `submit.svg` 和 `analyze.svg`
4. **权限**: 确保应用有权限读写队列目录和结果目录
