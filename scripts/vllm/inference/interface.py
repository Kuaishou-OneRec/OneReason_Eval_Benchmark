"""
vLLM 批量推理接口

支持单模型和多模型并行评估，提供简单易用的接口用于批量文本生成。
支持从 JSON 文件读取输入数据和生成参数，使用 vLLM 进行高效批量推理，并将结果保存到文件。

主要功能:
- 单模型评估：使用 VllmInferenceInterface
- 多模型并行评估：使用 MultiModelEvaluator（智能 GPU 调度）
- 支持从 JSON 文件加载批量输入数据
- 支持自定义生成参数
- 支持模型实例复用
- 支持单机多卡并行

使用示例:
    # 单模型评估
    from interface import VllmInferenceInterface

    interface = VllmInferenceInterface(model_path="Qwen/Qwen2-7B")
    results = interface.generate(prompts=["prompt1", "prompt2"])

    # 多模型评估
    from interface import MultiModelEvaluator

    evaluator = MultiModelEvaluator(available_gpus=[0, 1, 2, 3])
    results = evaluator.evaluate(
        models=["Qwen/Qwen2-7B", "Qwen/Qwen2-14B"],
        prompts=["prompt1", "prompt2"]
    )
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp

from scripts.vllm.utils.generator import VllmGenerator


# ============================================================================
# 单模型推理接口
# ============================================================================

class VllmInferenceInterface:
    """
    vLLM 单模型推理接口

    提供简单易用的批量文本生成功能，支持模型复用和自定义生成参数。
    """

    def __init__(
        self,
        model_path: str,
        checkpoint_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = False,
        dtype: str = "bfloat16",
        max_model_len: Optional[int] = None,
        force_enable_optimizations: bool = False,
        force_disable_optimizations: bool = False,
        **kwargs
    ):
        """
        初始化 vLLM 推理接口

        Args:
            model_path: 模型路径或 HuggingFace 模型名称
            checkpoint_path: PT checkpoint 路径（可选）
            tensor_parallel_size: 张量并行大小
            gpu_memory_utilization: GPU 内存利用率 (0-1)
            trust_remote_code: 是否信任远程代码
            dtype: 模型数据类型
            max_model_len: 最大模型长度
            force_enable_optimizations: 强制启用优化
            force_disable_optimizations: 强制禁用优化
            **kwargs: 其他 vLLM 参数
        """
        self.model_path = model_path
        self.generator = None
        self._generator_kwargs = {
            "model_name_or_path": model_path,
            "checkpoint_path": checkpoint_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "force_enable_optimizations": force_enable_optimizations,
            "force_disable_optimizations": force_disable_optimizations,
            **kwargs
        }

        # 初始化生成器（加载模型）
        self._initialize_generator()

    def _initialize_generator(self):
        """初始化或重新初始化生成器"""
        if self.generator is not None:
            del self.generator
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.generator = VllmGenerator(**self._generator_kwargs)

    def generate(
        self,
        prompts: Union[List[str], Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        批量生成文本

        Args:
            prompts: 输入提示列表或字典
            temperature: 采样温度
            top_p: nucleus sampling 参数
            top_k: top-k sampling 参数
            max_new_tokens: 最大生成 token 数
            num_return_sequences: 每个提示生成的候选数量
            repetition_penalty: 重复惩罚
            **kwargs: 其他生成参数

        Returns:
            Dict[str, List[str]]: {id: [generated_text_1, ...]}
        """
        # 转换输入格式
        if isinstance(prompts, list):
            prompts_dict = {str(i): prompt for i, prompt in enumerate(prompts)}
        else:
            prompts_dict = prompts

        # 构建生成参数
        gen_kwargs = {}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if num_return_sequences is not None:
            gen_kwargs["num_return_sequences"] = num_return_sequences
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        gen_kwargs.update(kwargs)

        # 执行生成
        results = self.generator.generate(prompts_dict, **gen_kwargs)
        return results

    def generate_from_json(
        self,
        input_json_path: str,
        output_path: Optional[str] = None,
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        从 JSON 文件读取输入并执行生成

        Args:
            input_json_path: 输入 JSON 文件路径
            output_path: 输出文件路径（可选）
            output_format: 输出格式 ("json" 或 "jsonl")

        Returns:
            Dict[str, Any]: 生成结果
        """
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        prompts = input_data.get("prompts", [])
        params = input_data.get("params", {})

        if not prompts:
            raise ValueError("输入 JSON 中未找到 'prompts' 字段或其为空")

        # 执行生成
        results_dict = self.generate(prompts, **params)

        # 组织输出格式
        results_list = []
        for idx, prompt in enumerate(prompts):
            key = str(idx)
            generated_texts = results_dict.get(key, [])
            results_list.append({
                "prompt": prompt,
                "generated_texts": generated_texts
            })

        output_data = {
            "results": results_list,
            "params": params
        }

        # 保存到文件
        if output_path:
            self.save_results(output_data, output_path, output_format)

        return output_data

    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        output_format: str = "json"
    ):
        """保存结果到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif output_format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results.get("results", []):
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")

        print(f"✓ 结果已保存到: {output_path}")

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'generator') and self.generator is not None:
            del self.generator
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================================
# 多模型并行评估接口
# ============================================================================

@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str
    name: str = None
    tensor_parallel_size: int = 1
    gpu_ids: List[int] = None
    checkpoint_path: Optional[str] = None
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self._generate_model_name(self.model_path)

    def _generate_model_name(self, model_path: str) -> str:
        """
        Generate model name from model path.

        Rule 0 (qwen): If last part contains 'qwen', use last part directly
        Example: .../Qwen2.5-7B-Instruct
        Output: Qwen2.5-7B-Instruct

        Format 1 (verl/actor): Extract directory before 'global_step' + step number
        Example: .../label_cond_reco_1112.../global_step_1000/actor/...
        Output: label_cond_reco_1112..._step1000

        Format 2 (pretrain): Extract 3 directories before 'global_step'
        Example: .../sft/v0.0.1_from_stg2_v0.1.1_8b/step20/global_step20/...
        Output: sft_v0.0.1_from_stg2_v0.1.1_8b_step20

        Default: Use last 3 parts of path
        """
        try:
            parts = model_path.strip().split('/')
            parts = [p for p in parts if p]

            # Rule 0: If last part contains 'qwen', use it directly
            if parts and 'qwen' in parts[-1].lower():
                return parts[-1]

            is_format1 = ('verl' in model_path and 'actor' in model_path)

            idx = -1
            global_step_num = ""
            for i in range(len(parts) - 1, -1, -1):
                if parts[i].startswith("global_step"):
                    idx = i
                    global_step_num = parts[i].replace("global_step_", "").replace("global_step", "")
                    break

            if is_format1 and idx > 0:
                dir_before_global_step = parts[idx - 1]
                return f"{dir_before_global_step}_step{global_step_num}"
            elif idx != -1 and idx >= 3:
                suffix_parts = parts[idx - 3 : idx]
                return "_".join(suffix_parts)
            else:
                return "_".join(parts[-3:]) if len(parts) >= 3 else "_".join(parts)
        except Exception:
            return os.path.basename(model_path)


class GPUScheduler:
    """GPU 调度器 - 管理 GPU 资源分配"""

    def __init__(self, available_gpus: Optional[List[int]] = None):
        """
        Args:
            available_gpus: 可用的 GPU ID 列表
        """
        if available_gpus is None:
            import torch
            if torch.cuda.is_available():
                self.available_gpus = list(range(torch.cuda.device_count()))
            else:
                raise RuntimeError("未检测到可用的 GPU")
        else:
            self.available_gpus = available_gpus

        print(f"✓ 检测到 {len(self.available_gpus)} 张 GPU: {self.available_gpus}")

        self.manager = mp.Manager()
        self.gpu_status = self.manager.dict({gpu_id: "free" for gpu_id in self.available_gpus})
        self.lock = self.manager.Lock()

    def get_status(self) -> Dict[int, str]:
        """获取当前 GPU 状态"""
        with self.lock:
            return dict(self.gpu_status)


def _evaluate_single_model_worker(
    model_config: ModelConfig,
    prompts: List[str],
    params: Dict[str, Any],
    gpu_scheduler_state: Any,
    lock: Any
) -> Dict[str, Any]:
    """
    单个模型评估的工作函数（在独立进程中运行）
    """
    import os
    import sys
    import time
    from pathlib import Path

    # Subprocesses will automatically have access to installed packages

    model_name = model_config.name
    print(f"\n[{model_name}] 开始评估...")

    try:
        # 分配 GPU
        print(f"[{model_name}] 等待分配 {model_config.tensor_parallel_size} 张 GPU...")

        allocated_gpus = None
        while allocated_gpus is None:
            with lock:
                free_gpus = [gpu_id for gpu_id, status in gpu_scheduler_state.items() if status == "free"]
                if len(free_gpus) >= model_config.tensor_parallel_size:
                    allocated_gpus = free_gpus[:model_config.tensor_parallel_size]
                    for gpu_id in allocated_gpus:
                        gpu_scheduler_state[gpu_id] = "busy"
            if allocated_gpus is None:
                time.sleep(1)

        print(f"[{model_name}] 已分配 GPU: {allocated_gpus}")

        # 设置 CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, allocated_gpus))

        # 初始化推理接口
        print(f"[{model_name}] 正在加载模型...")
        interface = VllmInferenceInterface(
            model_path=model_config.model_path,
            checkpoint_path=model_config.checkpoint_path,
            tensor_parallel_size=model_config.tensor_parallel_size,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
            dtype=model_config.dtype,
            max_model_len=model_config.max_model_len,
        )

        # 执行生成
        print(f"[{model_name}] 正在生成...")
        results_dict = interface.generate(prompts, **params)

        # 组织输出格式
        results_list = []
        for idx, prompt in enumerate(prompts):
            key = str(idx)
            generated_texts = results_dict.get(key, [])
            results_list.append({
                "prompt": prompt,
                "generated_texts": generated_texts
            })

        print(f"[{model_name}] ✓ 评估完成")

        # 清理
        del interface
        import gc
        gc.collect()

        # 释放 GPU
        with lock:
            for gpu_id in allocated_gpus:
                gpu_scheduler_state[gpu_id] = "free"
        print(f"[{model_name}] GPU 已释放: {allocated_gpus}")

        return {
            "model_name": model_name,
            "results": results_list,
            "success": True,
            "error": None
        }

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[{model_name}] ✗ 评估失败: {e}")

        if 'allocated_gpus' in locals() and allocated_gpus:
            with lock:
                for gpu_id in allocated_gpus:
                    gpu_scheduler_state[gpu_id] = "free"

        return {
            "model_name": model_name,
            "results": [],
            "success": False,
            "error": error_msg
        }


class MultiModelEvaluator:
    """多模型并行评估器"""

    def __init__(
        self,
        available_gpus: Optional[List[int]] = None,
        default_tensor_parallel_size: int = 1,
        default_dtype: str = "bfloat16",
        default_gpu_memory_utilization: float = 0.9
    ):
        """
        Args:
            available_gpus: 可用的 GPU ID 列表
            default_tensor_parallel_size: 默认张量并行大小
            default_dtype: 默认数据类型
            default_gpu_memory_utilization: 默认 GPU 内存利用率
        """
        self.gpu_scheduler = GPUScheduler(available_gpus)
        self.default_tensor_parallel_size = default_tensor_parallel_size
        self.default_dtype = default_dtype
        self.default_gpu_memory_utilization = default_gpu_memory_utilization

    def evaluate(
        self,
        models: List[str],
        prompts: List[str],
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        评估多个模型

        Args:
            models: 模型路径列表
            prompts: 输入提示列表
            params: 生成参数

        Returns:
            {model_name: [{"prompt": "...", "generated_texts": [...]}, ...]}
        """
        if params is None:
            params = {}

        # 创建模型配置
        model_configs = []
        for model_path in models:
            config = ModelConfig(
                model_path=model_path,
                tensor_parallel_size=self.default_tensor_parallel_size,
                dtype=self.default_dtype,
                gpu_memory_utilization=self.default_gpu_memory_utilization
            )
            model_configs.append(config)

        print(f"\n{'='*80}")
        print(f"开始评估 {len(model_configs)} 个模型")
        print(f"{'='*80}")
        for config in model_configs:
            print(f"  - {config.name} (需要 {config.tensor_parallel_size} 张 GPU)")
        print()

        # 使用 ProcessPoolExecutor 并行执行
        results = {}
        failed_models = []

        max_workers = max(1, len(self.gpu_scheduler.available_gpus) // self.default_tensor_parallel_size)
        print(f"最大并行模型数: {max_workers}\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {}
            for config in model_configs:
                future = executor.submit(
                    _evaluate_single_model_worker,
                    config,
                    prompts,
                    params,
                    self.gpu_scheduler.gpu_status,
                    self.gpu_scheduler.lock
                )
                future_to_model[future] = config.name

            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    if result["success"]:
                        results[model_name] = result["results"]
                    else:
                        failed_models.append((model_name, result["error"]))
                        print(f"\n✗ 模型 {model_name} 评估失败")
                except Exception as e:
                    failed_models.append((model_name, str(e)))
                    print(f"\n✗ 模型 {model_name} 发生异常: {e}")

        # 打印总结
        print(f"\n{'='*80}")
        print(f"评估完成!")
        print(f"{'='*80}")
        print(f"成功: {len(results)}/{len(model_configs)} 个模型")
        if failed_models:
            print(f"\n失败的模型:")
            for model_name, error in failed_models:
                print(f"  - {model_name}: {error[:100]}...")
        print()

        return results

    def evaluate_from_json(
        self,
        input_json_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从 JSON 文件读取配置并评估

        Args:
            input_json_path: 输入 JSON 文件路径
            output_path: 输出文件路径（可选）

        Returns:
            完整的评估结果
        """
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        models = input_data.get("models", [])
        prompts = input_data.get("prompts", [])
        params = input_data.get("params", {})

        if not models:
            raise ValueError("输入 JSON 中未找到 'models' 字段或其为空")
        if not prompts:
            raise ValueError("输入 JSON 中未找到 'prompts' 字段或其为空")

        # 执行评估
        results = self.evaluate(models, prompts, params)

        # 组织输出
        output_data = {
            "results": results,
            "params": params,
            "models": models
        }

        # 保存到文件
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"✓ 结果已保存到: {output_path}")

        return output_data


# ============================================================================
# 便捷函数
# ============================================================================

def evaluate_models_from_file(
    input_json_path: str,
    output_path: str,
    available_gpus: Optional[List[int]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：从 JSON 文件评估多个模型

    Args:
        input_json_path: 输入 JSON 文件路径
        output_path: 输出文件路径
        available_gpus: 可用的 GPU ID 列表
        **kwargs: 其他评估器参数

    Returns:
        评估结果
    """
    evaluator = MultiModelEvaluator(available_gpus=available_gpus, **kwargs)
    return evaluator.evaluate_from_json(input_json_path, output_path)
