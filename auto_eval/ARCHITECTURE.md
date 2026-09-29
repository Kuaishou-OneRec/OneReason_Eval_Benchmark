# 架构设计文档

## 概述

OneReason Eval Benchmark Auto Eval 是一个模块化的 Web 应用系统,用于 OneReason Eval Benchmark 的自动评估任务提交、执行和结果分析。

## 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Web Interface (web.py)                          │
│  ┌───────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │  Submit Page  │  │  Analyze Page    │  │ Extract PID Page │     │
│  │  (submit/)    │  │  (analyze/)      │  │ (extract_pid/)   │     │
│  └───────┬───────┘  └────────┬─────────┘  └──────────────────┘     │
│          │                   │                                       │
│          │ Write Tasks       │ Read Results                          │
│          ▼                   ▼                                       │
├─────────────────────────────────────────────────────────────────────┤
│                   File System (Queue + Results)                      │
│  ┌──────────┐  ┌────────────┐  ┌──────┐  ┌────────┐                │
│  │incoming/ │→ │processing/ │→ │done/ │  │failed/ │                │
│  └──────────┘  └────────────┘  └──────┘  └────────┘                │
│                      ▲    │                                          │
│                      │    │                                          │
│                 Read │    │ Execute                                  │
│                      │    ▼                                          │
│                ┌──────────────────────────┐                         │
│                │  Executor Module         │                         │
│                │  (executor/)             │                         │
│                │  - sentinel_worker.py    │                         │
│                │  - eval_script.py        │                         │
│                │  - eval_script.sh        │                         │
│                └──────────────────────────┘                         │
│                           │                                          │
│                           │ Write Results                            │
│                           ▼                                          │
│                ┌──────────────────────────┐                         │
│                │  /results/               │                         │
│                │  - eval_results.json     │                         │
│                │  - test_generated.json   │                         │
│                └──────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

## 模块职责

### 1. Web Interface (`web.py`)

**职责**: 应用入口,页面路由和导航

**功能**:
- 集成 submit、analyze 和 extract_pid 三个页面
- 实现左侧导航栏
- 合并 CSS 和 JavaScript
- 处理页面切换逻辑

**不做什么**:
- 不直接处理业务逻辑
- 不执行任务
- 不直接操作文件系统

---

### 2. Submit Module (`submit/`)

**职责**: 任务提交和队列管理

**文件结构**:
```
submit/
├── __init__.py      # 模块导出
├── handlers.py      # 业务逻辑
├── ui.py            # Gradio UI 组件
└── styles.py        # CSS/JavaScript
```

**核心功能**:
- ✅ 验证用户输入(邮箱、模型路径)
- ✅ 生成输出后缀
- ✅ 创建任务 JSON 文件
- ✅ 写入 `incoming/` 队列
- ✅ 显示队列状态(4 个队列)
- ✅ 删除任务(等待中/已完成/已失败)

**数据流**:
```
User Input → Validation → Generate JSON → Write to incoming/
```

**不做什么**:
- 不执行评估任务
- 不移动任务文件(由 executor 负责)
- 不生成结果分析

---

### 3. Analyze Module (`analyze/`)

**职责**: 结果分析和可视化

**文件结构**:
```
analyze/
├── __init__.py      # 模块导出
├── handlers.py      # 业务逻辑
├── ui.py            # Gradio UI 组件
├── plot.py          # 雷达图生成
└── styles.py        # CSS/JavaScript
```

**核心功能**:
- ✅ 扫描结果目录
- ✅ 加载 `eval_results.json`
- ✅ 生成雷达图(多模型对比)
- ✅ 生成指标表格
- ✅ 加载 `test_generated.json`
- ✅ 样本对比展示(模态窗口)

**数据流**:
```
Select Directories → Load eval_results.json → Generate Radar Chart + Table
                                            ↓
                                 Click "View" → Load test_generated.json
                                            ↓
                                 Compare Samples → Display Modal
```

**不做什么**:
- 不修改结果文件
- 不执行评估
- 不管理任务队列

---

### 4. Extract PID Module (`extract_pid/`)

**职责**: SID 代码提取和 PID 转换

**文件结构**:
```
extract_pid/
├── __init__.py      # 模块导出
├── handlers.py      # SID 提取和转换逻辑
├── ui.py            # Gradio UI 组件
├── styles.py        # CSS/JavaScript
└── README.md        # 模块文档
```

**核心功能**:
- ✅ 正则提取 SID 代码 (`<|sid_begin|><s_a_X><s_b_Y><s_c_Z><|sid_end|>`)
- ✅ 加载 parquet 文件建立 SID 到 PID 映射
- ✅ 批量转换文本中的 SID 为 PID
- ✅ 动态添加/删除输入框
- ✅ 展开/收起长结果
- ✅ 复制 PID 到剪贴板

**数据流**:
```
User Input (Text with SIDs) → Extract SIDs → Lookup in parquet mapping
                                             ↓
                                  Convert to PIDs → Display as comma-separated
                                             ↓
                                  Copy to Clipboard (on click)
```

**数据来源**:
- Parquet 文件: `/share/example_user/user_foundation_data/qwen3_8b_4096_itemic_codes/pid2sid_merged.parquet`

**不做什么**:
- 不修改 parquet 文件
- 不执行任何后台任务
- 不保存转换历史

---

### 5. Executor Module (`executor/`)

**职责**: 后台任务执行和监控

**文件结构**:
```
executor/
├── __init__.py           # 模块导出
├── sentinel_worker.py    # 守护进程
├── eval_script.py        # Python 包装器
└── eval_script.sh        # Shell 脚本
```

**核心功能**:

#### `sentinel_worker.py`
- ✅ 监控 `incoming/` 队列(FIFO)
- ✅ 移动任务到 `processing/`
- ✅ 调用 `eval_script.py`
- ✅ 根据退出码移动到 `done/` 或 `failed/`
- ✅ 支持重试(最多 3 次)
- ✅ 记录任务历史

#### `eval_script.py`
- ✅ 读取任务 JSON
- ✅ 调用 `eval_script.sh`
- ✅ 捕获 stdout/stderr
- ✅ 记录错误日志到 JSON

#### `eval_script.sh`
- ✅ 执行实际评估命令
- ✅ 分 3 个任务组
- ✅ 不同 GPU 内存配置
- ✅ 输出到指定目录

**数据流**:
```
incoming/ → sentinel_worker.py → processing/
                ↓
         eval_script.py → eval_script.sh → Generate Results
                ↓
         Check Exit Code
                ↓
         done/ or failed/
```

**运行方式**:
```bash
# 独立后台进程,与 Web 应用分离
python3 executor/sentinel_worker.py
```

**不做什么**:
- 不提供 Web 界面
- 不直接与用户交互
- 不分析结果

---

## 共享组件

### `config.py`
- 环境检测(本地/服务器)
- 路径配置
- 队列目录
- UI 设置

### `utils.py`
- 邮箱验证
- 路径处理
- JSON 加载
- 通用工具函数

---

## 数据模型

### 任务 JSON (Task JSON)

**位置**: `incoming/`, `processing/`, `done/`, `failed/`

**文件名格式**: `{timestamp}_{username}_{suffix}_{random}.json`

**结构**:
```json
{
  "task_id": "20241118_120000_example_user_stg2_v0.1.1_8b_step8000_abc1",
  "model_path": "/path/to/resource",
  "output_suffix": "stg2_v0.1.1_8b_step8000",
  "submitter_email": "example_user@example.org",
  "submitted_at": "2024-11-18T12:00:00",
  "max_retries": 3,
  "retry_count": 0,
  "error_log": {  // 仅在失败时存在
    "exit_code": 1,
    "stdout": "...",
    "stderr": "..."
  }
}
```

### 评估结果 JSON (eval_results.json)

**位置**: `{BASE_BENCHMARK_DIR}/results/results_{suffix}/`
**默认**: `/path/to/resource`
> 可通过 `config.py` 中的 `BASE_BENCHMARK_DIR` 修改

**结构**:
```json
{
  "model": {
    "math_500": {
      "test": {
        "pass@1": 0.354,
        "total_samples": 500,
        ...
      }
    },
    "gsm8k": { ... },
    ...
  }
}
```

### 测试生成数据 (test_generated.json)

**位置**: `{BASE_BENCHMARK_DIR}/results/results_{suffix}/`
**默认**: `/path/to/resource`
> 可通过 `config.py` 中的 `BASE_BENCHMARK_DIR` 修改

**文件名**: `{dataset_name}_test_generated.json`

**结构**:
```json
{
  "model_name": "Qwen3-1.7B",
  "task_name": "gsm8k",
  "split": "test",
  "samples": {
    "0": {
      "prompt": "...",
      "generations": ["..."],
      "ground_truth": "...",
      "is_correct": false,
      "pass@1": false,
      "metadata": { ... }
    },
    ...
  }
}
```

---

## 部署架构

### 单机部署

```
Server:
  - Web App (web.py)      → Port 8890
  - Sentinel Worker       → Background Process
  - File System           → Shared Storage
```

**启动命令**:
```bash
# Terminal 1: 启动 Web 应用
python3 web.py

# Terminal 2: 启动 Sentinel Worker
python3 executor/sentinel_worker.py
```

### 分布式部署(未来)

```
Web Server:
  - web.py (Port 8890)

Worker Server(s):
  - executor/sentinel_worker.py
  - Multiple workers can run in parallel

Shared Storage:
  - NFS/Distributed FS
  - incoming/, processing/, done/, failed/
  - /results/
```

---

## 扩展指南

### 添加新页面

1. 创建新模块目录:
   ```bash
   mkdir auto_eval/new_feature
   ```

2. 创建模块文件:
   ```
   new_feature/
   ├── __init__.py
   ├── handlers.py
   ├── ui.py
   └── styles.py
   ```

3. 在 `web.py` 中导入和集成:
   ```python
   from new_feature.ui import create_new_feature_ui
   from new_feature.styles import NEW_FEATURE_CSS
   ```

4. 添加导航按钮

### 修改任务执行逻辑

编辑 `executor/eval_script.sh` 或 `executor/eval_script.py`

### 添加新的分析功能

在 `analyze/` 模块中:
- 添加新的处理函数到 `handlers.py`
- 添加新的 UI 组件到 `ui.py`
- 更新 `plot.py` 如果需要新的图表类型

---

## 安全考虑

1. **路径验证**: 所有用户输入的路径都经过验证
2. **JSON 注入**: 使用标准 `json` 库,防止注入
3. **命令注入**: 不直接执行用户输入的命令
4. **权限控制**: 文件系统权限隔离
5. **日志记录**: 所有操作都有日志

---

## 性能优化

1. **异步加载**: 页面加载时异步获取队列状态
2. **图表缓存**: 生成的雷达图缓存在 `plots/` 目录
3. **批量操作**: 支持批量删除任务
4. **轮询间隔**: Sentinel worker 5 秒轮询间隔
5. **结果限制**: 只显示最近 20 个已完成任务

---

## 故障排查

### Web 应用无法启动
- 检查端口 8890 是否被占用
- 检查 Python 依赖是否安装
- 检查 `icon/` 目录和 SVG 文件

### 任务卡在 incoming/
- 检查 sentinel_worker.py 是否运行
- 检查队列目录权限
- 查看 sentinel worker 日志

### 结果分析页面显示空
- 检查 `{RESULTS_BASE_DIR}` 路径 (默认: `/path/to/resource`)
- 检查目录权限
- 确认 `eval_results.json` 文件存在

### 样本对比不显示
- 检查 `test_generated.json` 文件存在
- 检查文件格式和 `task_name` 字段
- 查看浏览器控制台错误

---

## 总结

这个架构设计遵循以下原则:

✅ **关注点分离**: 每个模块职责单一明确
✅ **松耦合**: 模块间通过文件系统解耦
✅ **可扩展**: 易于添加新模块和功能
✅ **可维护**: 清晰的目录结构和代码组织
✅ **可测试**: 每个模块可以独立测试
✅ **可部署**: 支持单机和分布式部署
