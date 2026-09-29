# 多处理器并行评估系统使用指南

## 概述

多处理器并行评估系统支持在N个处理器上自动分配和执行AIR-Bench评估任务，实现最大化的资源利用。每个处理器由多台机器组成独立的Ray集群。

## 核心特性

- **多处理器支持**：支持任意数量的处理器，通过环境变量配置
- **独立Ray集群**：每个处理器内的多台机器组成独立的Ray集群
- **集中式队列**：任务队列在处理器0的head节点上，统一调度
- **动态任务分配**：根据处理器空闲状态自动分配任务
- **处理器单任务**：每个处理器同时只运行1个任务，独占该处理器的所有GPU资源
- **自动化调度**：无需手动干预，一键启动所有任务

## 前提条件

1. **机器配置**：
   - 在 `/etc/mpi/hostfile` 中配置所有机器的地址
   - 机器数量 = NUM_PROCESSORS × MACHINES_PER_PROCESSOR
   - 例如：5个处理器，每个处理器5台机器，需要25台机器

2. **SSH配置**：
   - 从处理器0的head节点可以SSH免密登录到所有其他机器
   - 测试：`ssh <机器IP> "echo test"`

3. **环境一致性**：
   - 所有机器的项目路径相同
   - Conda环境名称相同
   - 代码版本同步

4. **环境变量配置**：
   - `NUM_PROCESSORS`：处理器数量（默认：2）
   - `MACHINES_PER_PROCESSOR`：每个处理器包含的机器数量（默认：1）

## 快速开始

### 1. 准备任务配置文件

创建任务配置文件（例如 `my_tasks.txt`），每行一个评估命令：

```bash
# my_tasks.txt
python scripts/ray-vllm/evaluate.py --model_path Qwen/Qwen2-7B --data_dir ./data --output_dir ./results/task1 --task_types recommendation
python scripts/ray-vllm/evaluate.py --model_path Qwen/Qwen2-7B --data_dir ./data --output_dir ./results/task2 --task_types classification
python scripts/ray-vllm/evaluate.py --model_path Qwen/Qwen2-7B --data_dir ./data --output_dir ./results/task3 --task_types generation
```

**注意**：
- 每行是完整的 `python evaluate.py ...` 命令
- 可以包含注释（以 `#` 开头）
- 空行会被自动忽略
- `--ray_address local` 会被自动添加（如果未指定）

### 2. 配置处理器

设置环境变量以配置处理器架构：

```bash
# 例如：5个处理器，每个处理器包含5台机器
export NUM_PROCESSORS=5
export MACHINES_PER_PROCESSOR=5
```

### 3. 启动多处理器并行评估

```bash
cd /path/to/AIR-Bench/auto_eval/executor

bash eval_parallel.sh --config my_tasks.txt
```

### 4. 监控执行进度

主日志：
```bash
tail -f main.log
```

单个任务日志：
```bash
tail -f task_1_<task_name>.log
tail -f task_2_<task_name>.log
```

处理器Ray集群状态：
```bash
# 在处理器0的head节点上查看
ray status

# 在其他处理器的head节点上查看
ssh <processor_N_head_node> ray status
```

GPU使用情况：
```bash
# 查看某台机器的GPU使用情况
ssh <machine_ip> nvidia-smi
```

## 高级选项

### 指定日志目录

```bash
bash eval_parallel.sh \
    --config my_tasks.txt \
    --log_dir /path/to/custom/logs
```

### 修改Ray端口

```bash
bash eval_parallel.sh \
    --config my_tasks.txt \
    --ray_port 6380
```

### 配置不同数量的处理器

```bash
# 2个处理器，每个处理器1台机器（类似原双机模式）
export NUM_PROCESSORS=2
export MACHINES_PER_PROCESSOR=1
bash eval_parallel.sh --config my_tasks.txt

# 4个处理器，每个处理器2台机器
export NUM_PROCESSORS=4
export MACHINES_PER_PROCESSOR=2
bash eval_parallel.sh --config my_tasks.txt
```

### 指定Conda环境

```bash
bash eval_parallel.sh \
    --config my_tasks.txt \
    --conda_env your_env_name
```

## 工作流程

1. **初始化阶段**
   - 读取 `/etc/mpi/hostfile` 获取所有机器列表
   - 按 NUM_PROCESSORS 和 MACHINES_PER_PROCESSOR 将机器分组
   - 为每个处理器组创建专用的 hostfile
   - 在每个处理器组上调用 `init_ray_cluster.sh` 启动独立的Ray集群
   - 从配置文件加载任务队列（在处理器0的head节点）

2. **调度阶段**
   - 每30秒检查一次处理器状态
   - 当有处理器空闲时，从队列中取出任务并分配
   - 通过SSH在处理器的head节点上启动 `evaluate.py` 进程

3. **监控阶段**
   - 持续监控每个处理器的任务执行状态
   - 通过检查SSH进程PID判断任务是否完成
   - 任务完成后自动标记处理器为空闲

4. **完成阶段**
   - 等待所有任务执行完毕
   - 自动关闭所有处理器的Ray集群
   - 生成执行摘要

## 日志文件说明

- `main.log` - 主调度器日志，包含所有调度决策
- `ray_processor_0.log`, `ray_processor_1.log`, ... - 各处理器的Ray集群日志
- `task_<id>_<task_name>.log` - 每个任务的执行日志
- `processor_<id>_status.txt` - 处理器状态文件（idle/busy）
- `processor_<id>_task_id.txt` - 处理器当前运行的任务ID
- `processor_<id>_task_name.txt` - 处理器当前运行的任务名称
- `processor_<id>_task.pid` - 处理器当前运行任务的PID

## 故障排查

### 1. SSH连接失败

**问题**：无法连接到远程机器

**解决方案**：
```bash
# 测试SSH连接
ssh <远程机器> "echo test"

# 检查SSH密钥
ssh-copy-id <远程机器>
```

### 2. Ray集群启动失败

**问题**：Ray集群无法启动或状态异常

**解决方案**：
```bash
# 在所有机器上停止现有Ray实例
for machine in $(awk '{print $1}' /etc/mpi/hostfile); do
    ssh $machine "ray stop; rm -rf /tmp/ray"
done

# 重新运行脚本
```

### 3. 任务执行失败

**问题**：任务日志显示错误

**解决方案**：
1. 查看任务日志：`cat task_X.log`
2. 检查错误信息
3. 确认模型路径、数据路径是否正确
4. 验证两台机器上的环境一致性

### 4. 端口冲突

**问题**：Ray端口已被占用

**解决方案**：
```bash
# 使用不同的端口
bash eval_parallel.sh --config my_tasks.txt --ray_port 6380
```

## 最佳实践

1. **任务粒度**：
   - 建议每个任务耗时相近，以实现负载均衡
   - 避免一个任务远长于其他任务

2. **资源规划**：
   - 确保每台机器有足够的GPU内存
   - 监控磁盘空间（日志和结果文件）

3. **测试流程**：
   - 先用少量任务测试（2-3个）
   - 验证无误后再运行完整任务集

4. **日志管理**：
   - 定期清理旧日志文件
   - 重要日志及时备份

## 示例场景

### 场景1：评估多个模型

```bash
# task_config.txt
python scripts/ray-vllm/evaluate.py --model_path Qwen/Qwen2-7B --output_dir ./results/qwen2-7b
python scripts/ray-vllm/evaluate.py --model_path Qwen/Qwen2-14B --output_dir ./results/qwen2-14b
python scripts/ray-vllm/evaluate.py --model_path meta-llama/Llama-2-7b --output_dir ./results/llama2-7b
python scripts/ray-vllm/evaluate.py --model_path meta-llama/Llama-2-13b --output_dir ./results/llama2-13b
```

### 场景2：评估不同任务类型

```bash
# task_config.txt
python scripts/ray-vllm/evaluate.py --model_path Qwen/Qwen2-7B --task_types recommendation --output_dir ./results/recommendation
python scripts/ray-vllm/evaluate.py --model_path Qwen/Qwen2-7B --task_types classification --output_dir ./results/classification
python scripts/ray-vllm/evaluate.py --model_path Qwen/Qwen2-7B --task_types generation --output_dir ./results/generation
```

### 场景3：评估不同配置

```bash
# task_config.txt
python scripts/ray-vllm/evaluate.py --model_path Qwen/Qwen2-7B --temperature 0.7 --output_dir ./results/temp_0.7
python scripts/ray-vllm/evaluate.py --model_path Qwen/Qwen2-7B --temperature 0.9 --output_dir ./results/temp_0.9
python scripts/ray-vllm/evaluate.py --model_path Qwen/Qwen2-7B --num_beams 5 --output_dir ./results/beam_5
```

## 相关文件

- `eval_parallel.sh` - 主控调度脚本
- `scripts/init_ray_cluster.sh` - Ray集群启动脚本
- `task_config_example.txt` - 任务配置示例

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│  主控脚本 (eval_parallel.sh) - 运行在处理器0的head节点          │
│  ├─ 读取机器列表 (/etc/mpi/hostfile)                             │
│  ├─ 按处理器数量分组机器                                         │
│  ├─ 初始化N个处理器的Ray集群                                    │
│  ├─ 初始化集中式任务队列                                         │
│  └─ 启动调度循环（轮询分配任务）                                 │
└──────┬──────────────────┬──────────────────┬────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  处理器0      │   │  处理器1      │   │  处理器N      │
│  (任务队列)   │   │              │   │              │
├──────────────┤   ├──────────────┤   ├──────────────┤
│ 机器1(Head)  │   │ 机器6(Head)  │   │ 机器X(Head)  │
│ 机器2(Worker)│   │ 机器7(Worker)│   │ 机器Y(Worker)│
│ 机器3(Worker)│   │ 机器8(Worker)│   │ 机器Z(Worker)│
│ ...          │   │ ...          │   │ ...          │
│ Ray集群      │   │ Ray集群      │   │ Ray集群      │
│ 运行1个任务   │   │ 运行1个任务   │   │ 运行1个任务   │
└──────────────┘   └──────────────┘   └──────────────┘
```

**说明**：
- 每个处理器包含多台机器，组成独立的Ray集群
- 第一台机器是Ray head节点，其余是worker节点
- 任务队列在处理器0上，通过SSH分配任务到各处理器
- 每个处理器同时只运行1个任务，独占该处理器的所有GPU

## 联系与支持

如有问题，请查看日志文件或联系开发团队。
