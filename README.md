# OneReason_Eval_Benchmark

## 目录结构

```text
benchmark/                 # Benchmark 调度、生成流程、任务配置与评分器
api/                       # 多 provider 客户端与外部配置入口
scripts/                   # 运行、评估、重评及 vLLM / Transformers 后端脚本
auto_eval/                 # 可选本地队列、worker 和 Web UI
custom_models/             # 自定义模型实现
```

## 安装

需要 Python 3.10+，并选择与 GPU/CUDA 匹配的 PyTorch 与推理后端。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .

# Ray + vLLM 生成
python3 -m pip install -e '.[vllm]'

# Caption 裁判、BERTScore、NLI
python3 -m pip install -e '.[judge]'
```

从仓库根目录运行，并设置模块路径：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## 生成并评估

正式配置为 **datav3.1、seed=42、Race、nothink**。`datav3.1` 表示数据版本；程序中的任务版本参数为 `--version v3.1`。

运行全部 10 个任务：

```bash
bash scripts/run_competition.sh /path/to/model --ray_address local --num_gpus 1
```

脚本默认读取 `../data`，输出至 `results/datav3.1_seed42_race_nothink`。可通过 `BENCHMARK_DATA_DIR` 和 `BENCHMARK_RESULTS_DIR` 指定目录。核心参数为：

```bash
python3 scripts/ray-vllm/evaluate.py \
  --model_path /path/to/model \
  --version v3.1 --seed 42 \
  --race_mode true --enable_thinking false \
  --data_dir ../data \
  --output_dir ./results/datav3.1_seed42_race_nothink \
  --sample_size full --ray_address local --num_gpus 1
```

不传 `--task_types` 时运行全部 10 个任务。

Race 规则：

- 常识、事件选择、主题生成和三个 caption 任务仅运行 nothink。
- 四个推荐任务将 64 个候选分成 32 个 think 和 32 个 nothink，按 think 在前、nothink 在后合并后评分。
- 界面选择 nothink 对应 `--enable_thinking false`，不会关闭推荐任务的 Race 内部 think 分支。
- 推荐输出沿用 3 个 SID token，模型 tokenizer 必须支持对应特殊 token。

## 重新评分

```bash
python3 scripts/eval_dev_results.py \
  --version v3.1 \
  --data_dir ../data \
  --output_dir ./results/datav3.1_seed42_race_nothink \
  --overwrite
```
