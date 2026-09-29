"""
断点续传管理模块

按 prompt 粒度记录完成状态，支持任务中断后恢复。
进度文件保存在输出目录下。
"""

import json
import os
from pathlib import Path
from typing import Dict, Set, Optional, Any
from filelock import FileLock


class CheckpointManager:
    """
    断点续传管理器

    功能：
    - 记录已完成的 prompt 索引
    - 支持增量保存结果
    - 任务恢复时跳过已完成的 prompt
    """

    def __init__(self, output_dir: str, checkpoint_name: str = "progress.json"):
        """
        Args:
            output_dir: 输出目录路径
            checkpoint_name: 进度文件名
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.output_dir / checkpoint_name
        self.lock_path = self.output_dir / f"{checkpoint_name}.lock"

        # 已完成的 prompt 索引集合
        self.completed_indices: Set[int] = set()

        # 加载已有进度
        self._load_checkpoint()

    def _load_checkpoint(self):
        """加载已有的进度文件"""
        if self.checkpoint_path.exists():
            try:
                with FileLock(str(self.lock_path)):
                    with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.completed_indices = set(data.get("completed_indices", []))
                print(f"[Checkpoint] 已加载进度: {len(self.completed_indices)} 个 prompt 已完成")
            except Exception as e:
                print(f"[Checkpoint] 加载进度文件失败: {e}，将从头开始")
                self.completed_indices = set()

    def _save_checkpoint(self):
        """保存进度到文件"""
        with FileLock(str(self.lock_path)):
            data = {
                "completed_indices": sorted(list(self.completed_indices)),
                "total_completed": len(self.completed_indices)
            }
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def mark_completed(self, indices: list):
        """
        标记一批 prompt 为已完成

        Args:
            indices: 已完成的 prompt 索引列表
        """
        self.completed_indices.update(indices)
        self._save_checkpoint()

    def is_completed(self, index: int) -> bool:
        """检查某个 prompt 是否已完成"""
        return index in self.completed_indices

    def get_pending_indices(self, total: int) -> list:
        """
        获取待处理的 prompt 索引列表

        Args:
            total: 总 prompt 数量

        Returns:
            待处理的索引列表
        """
        all_indices = set(range(total))
        pending = all_indices - self.completed_indices
        return sorted(list(pending))

    def get_progress(self, total: int) -> Dict[str, Any]:
        """
        获取当前进度信息

        Args:
            total: 总 prompt 数量

        Returns:
            进度信息字典
        """
        completed = len(self.completed_indices)
        return {
            "completed": completed,
            "total": total,
            "pending": total - completed,
            "progress_percent": round(completed / total * 100, 2) if total > 0 else 0
        }

    def reset(self):
        """重置进度（清空已完成记录）"""
        self.completed_indices = set()
        if self.checkpoint_path.exists():
            os.remove(self.checkpoint_path)
        print("[Checkpoint] 进度已重置")
