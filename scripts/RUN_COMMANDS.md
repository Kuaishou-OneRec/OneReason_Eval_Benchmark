# 在其他机器运行推荐、通识和演进评估

## 1. 安装与公共参数

在已配置适配 GPU/CUDA 的 Python 3.10+ 环境中：

```bash
cd /path/to/work/OneReason_Eval_Benchmark
python3 -m pip install -e '.[vllm,judge]'
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

export BENCHMARK_DATA_DIR="$(cd ../data_anonymized && pwd)"
export MODEL_PATH=/path/to/model
export RUN_ROOT="$PWD/results/my_model_datav3.1_seed42_race_nothink"

# 按目标机器和模型显存需求修改；以下是单 GPU 参数。
export NUM_GPUS=1
export TP_SIZE=1

# 演进 topic_gen 的 NLI 评分需要本地模型目录。
# 配置中的模型为 cross-encoder/nli-deberta-v3-base；搬运对应模型及 tokenizer。
export NLI_MODEL_DIR="$(cd ../../model/nli-deberta-v3-base && pwd)"
```

`NUM_GPUS` 是总 GPU 数；`TP_SIZE` 是每个模型副本使用的 GPU 数。例如 8 卡、每个模型副本占 2 卡时设置 8 和 2。模型须能装入所选 TP 规模。`judge` 安装组提供演进评分的依赖；这 7 个任务不需要配置 caption 的 JUDGE_API_KEY。

上述 NLI_MODEL_DIR 为本机存放路径；搬到其他机器后，将它改为该机器上模型目录的实际路径。整个模型目录需一起搬运。

## 2. 一次运行全部 7 个任务

```bash
export BENCHMARK_RESULTS_DIR="$RUN_ROOT/all7"
bash scripts/run_competition.sh "$MODEL_PATH" \
  --ray_address local \
  --num_gpus "$NUM_GPUS" \
  --tensor_parallel_size "$TP_SIZE" \
  --gpu_memory_utilization 0.8 \
  --worker_batch_size 4 \
  --task_types \
    challenge_recommendation_video_full \
    challenge_recommendation_product \
    challenge_recommendation_ad \
    challenge_recommendation_live \
    challenge_common_sense \
    challenge_evolution_action_select \
    challenge_evolution_topic_gen
```

脚本固定 `--version v3.1 --seed 42 --race_mode true --enable_thinking false --sample_size full`。这里 data v3.1 的程序参数是 `--version v3.1`。

保留原 Race 行为：推荐任务分成 32 个 think 和 32 个 nothink 候选，think 在前合并评分；通识和演进仅 nothink。不要额外覆盖候选数、beam 或采样参数，以免改变评测口径。

该入口先完成模型推理，再释放 vLLM 显存，随后计算指标。

## 3. 按任务族分别运行

以下三组可按顺序运行；每组使用独立结果目录。

```bash
# 推荐：四个领域
export BENCHMARK_RESULTS_DIR="$RUN_ROOT/recommendation"
bash scripts/run_competition.sh "$MODEL_PATH" \
  --ray_address local --num_gpus "$NUM_GPUS" \
  --tensor_parallel_size "$TP_SIZE" \
  --gpu_memory_utilization 0.8 --worker_batch_size 4 \
  --task_types challenge_recommendation_video_full \
    challenge_recommendation_product challenge_recommendation_ad \
    challenge_recommendation_live

# 通识
export BENCHMARK_RESULTS_DIR="$RUN_ROOT/common_sense"
bash scripts/run_competition.sh "$MODEL_PATH" \
  --ray_address local --num_gpus "$NUM_GPUS" \
  --tensor_parallel_size "$TP_SIZE" \
  --gpu_memory_utilization 0.8 --worker_batch_size 4 \
  --task_types challenge_common_sense

# 演进：事件选择 + 主题逻辑链生成
export BENCHMARK_RESULTS_DIR="$RUN_ROOT/evolution"
bash scripts/run_competition.sh "$MODEL_PATH" \
  --ray_address local --num_gpus "$NUM_GPUS" \
  --tensor_parallel_size "$TP_SIZE" \
  --gpu_memory_utilization 0.8 --worker_batch_size 4 \
  --task_types challenge_evolution_action_select challenge_evolution_topic_gen
```

## 4. 少量样本试跑

试跑直接使用 Python 入口；完整比赛脚本会固定 sample_size=full。输出放入单独目录，避免全量运行复用抽样结果。

```bash
python3 scripts/ray-vllm/evaluate.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$BENCHMARK_DATA_DIR" \
  --output_dir "$RUN_ROOT/smoke" \
  --version v3.1 --seed 42 --race_mode true --enable_thinking false \
  --sample_size 8 --ray_address local \
  --num_gpus "$NUM_GPUS" --tensor_parallel_size "$TP_SIZE" \
  --gpu_memory_utilization 0.8 --worker_batch_size 1 \
  --task_types challenge_common_sense challenge_recommendation_live
```

## 5. 已完成推理，只重新评分

若生成已完成但评分中断，可以跳过推理，对已有 test_generated.json 重新评分。以下对应第 3 节的 all7 结果目录：

```bash
# 在同一进程内设置随机种子，再调用原重新评分入口，固定随机 PID 基线。
python3 - \
  --output_dir "$RUN_ROOT/all7" \
  --data_dir "$BENCHMARK_DATA_DIR" \
  --version v3.1 --overwrite \
  --task_types challenge_recommendation_video_full \
    challenge_recommendation_product challenge_recommendation_ad \
    challenge_recommendation_live challenge_common_sense \
    challenge_evolution_action_select challenge_evolution_topic_gen <<'PY'
import random
import runpy
import numpy as np
random.seed(42)
np.random.seed(42)
runpy.run_path('scripts/eval_dev_results.py', run_name='__main__')
PY
```

分组运行时，将 output_dir 改成对应分组目录，并传入该组任务列表。这里的 `--overwrite` 仅重算评分；在推理入口使用它会重新生成结果。默认会复用已有生成文件，换模型或参数时应更换结果目录。

## 6. 结果位置与验证范围

```text
结果目录/
  模型名/任务名/test_generated.json
  eval_results.json
```

`eval_results.json` 为聚合结果。已完成参数解析、任务注册和离线验证；目标机器的 GPU 推理、vLLM/CUDA 版本组合及 NLI 模型实际加载仍需在该机器通过试跑确认。
