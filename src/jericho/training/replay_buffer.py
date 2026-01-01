"""
Prioritized Replay Buffer for Wake-Sleep Training

实现基于优先级的经验回放缓冲区，用于睡眠阶段的记忆巩固。

Author: Jericho Team
Date: 2026-01-02
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import random
import numpy as np
import torch


@dataclass
class Experience:
    """经验样本"""
    data: Any  # 原始数据（batch dict）
    loss: float  # 该样本的损失
    priority: float  # 采样优先级
    step: int  # 记录时的训练步数
    metadata: Optional[Dict] = None


class ReplayBuffer:
    """
    基于优先级的经验回放缓冲区
    
    使用 Reservoir Sampling 保持固定大小，
    采样时使用优先级加权。
    
    Parameters
    ----------
    capacity : int
        缓冲区最大容量
    alpha : float
        优先级指数，0 表示均匀采样，1 表示完全按优先级
    beta : float
        重要性采样权重指数
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.total_added = 0
    
    def add(
        self,
        data: Any,
        loss: float,
        step: int,
        metadata: Optional[Dict] = None,
    ):
        """
        添加经验到缓冲区
        
        Parameters
        ----------
        data : Any
            经验数据
        loss : float
            该样本的损失
        step : int
            当前训练步数
        metadata : dict, optional
            额外元数据
        """
        priority = (abs(loss) + 1e-6) ** self.alpha  # 高 loss 样本优先
        
        exp = Experience(
            data=data,
            loss=loss,
            priority=priority,
            step=step,
            metadata=metadata,
        )
        
        self.buffer.append(exp)
        self.priorities.append(priority)
        self.total_added += 1
    
    def sample(self, batch_size: int) -> Tuple[List[Any], torch.Tensor]:
        """
        按优先级采样
        
        Parameters
        ----------
        batch_size : int
            采样数量
        
        Returns
        -------
        samples : List[Any]
            采样的数据列表
        weights : torch.Tensor
            重要性采样权重
        """
        if len(self.buffer) == 0:
            return [], torch.tensor([])
        
        n = min(batch_size, len(self.buffer))
        
        # 计算采样概率
        priorities = np.array(list(self.priorities))
        probs = priorities / priorities.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), size=n, replace=False, p=probs)
        
        # 计算重要性采样权重
        min_prob = probs.min()
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        
        samples = [self.buffer[i].data for i in indices]
        weights = torch.tensor(weights, dtype=torch.float32)
        
        return samples, weights
    
    def sample_recent(self, batch_size: int, recency_window: int = 1000) -> List[Any]:
        """
        采样最近的经验
        
        Parameters
        ----------
        batch_size : int
            采样数量
        recency_window : int
            最近窗口大小
        
        Returns
        -------
        List[Any]
            采样的数据列表
        """
        if len(self.buffer) == 0:
            return []
        
        window = min(recency_window, len(self.buffer))
        n = min(batch_size, window)
        
        # 从最近的样本中随机采样
        recent_indices = list(range(len(self.buffer) - window, len(self.buffer)))
        indices = random.sample(recent_indices, n)
        
        return [self.buffer[i].data for i in indices]
    
    def update_priority(self, indices: List[int], new_losses: List[float]):
        """更新优先级"""
        for i, loss in zip(indices, new_losses):
            if 0 <= i < len(self.buffer):
                new_priority = (abs(loss) + 1e-6) ** self.alpha
                self.priorities[i] = new_priority
                self.buffer[i].priority = new_priority
                self.buffer[i].loss = loss
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def stats(self) -> Dict:
        """获取缓冲区统计"""
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "total_added": self.total_added,
            }
        
        losses = [exp.loss for exp in self.buffer]
        priorities = list(self.priorities)
        
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "total_added": self.total_added,
            "mean_loss": np.mean(losses),
            "max_loss": np.max(losses),
            "min_loss": np.min(losses),
            "mean_priority": np.mean(priorities),
        }


class WakeSleepScheduler:
    """
    Wake-Sleep 训练调度器
    
    管理 Wake（正常训练）、Sleep-Replay（记忆回放）、Sleep-Prune（稀疏化）阶段的切换。
    
    Parameters
    ----------
    cycle_epochs : int
        一个完整 Wake-Sleep 循环的 epoch 数
    wake_ratio : float
        Wake 阶段占比
    replay_ratio : float
        Sleep-Replay 阶段占比
    prune_ratio : float
        Sleep-Prune 阶段占比（自动计算）
    """
    
    def __init__(
        self,
        cycle_epochs: int = 5,
        wake_ratio: float = 0.8,
        replay_ratio: float = 0.1,
        warmup_epochs: int = 5,
    ):
        self.cycle_epochs = cycle_epochs
        self.wake_ratio = wake_ratio
        self.replay_ratio = replay_ratio
        self.prune_ratio = 1.0 - wake_ratio - replay_ratio
        self.warmup_epochs = warmup_epochs
        
        assert self.prune_ratio >= 0, "wake_ratio + replay_ratio must be <= 1"
    
    def get_phase(self, epoch: int) -> str:
        """
        获取当前训练阶段
        
        Parameters
        ----------
        epoch : int
            当前 epoch
        
        Returns
        -------
        str
            "warmup", "wake", "sleep_replay", 或 "sleep_prune"
        """
        if epoch < self.warmup_epochs:
            return "warmup"
        
        effective_epoch = epoch - self.warmup_epochs
        position_in_cycle = (effective_epoch % self.cycle_epochs) / self.cycle_epochs
        
        if position_in_cycle < self.wake_ratio:
            return "wake"
        elif position_in_cycle < self.wake_ratio + self.replay_ratio:
            return "sleep_replay"
        else:
            return "sleep_prune"
    
    def get_phase_info(self, epoch: int) -> Dict:
        """获取当前阶段的详细信息"""
        phase = self.get_phase(epoch)
        
        effective_epoch = max(0, epoch - self.warmup_epochs)
        cycle_num = effective_epoch // self.cycle_epochs
        position_in_cycle = effective_epoch % self.cycle_epochs
        
        return {
            "phase": phase,
            "epoch": epoch,
            "cycle": cycle_num,
            "position_in_cycle": position_in_cycle,
            "is_warmup": epoch < self.warmup_epochs,
        }
    
    def should_enable_l0_gate(self, epoch: int) -> bool:
        """是否应该启用 L0 Gate（仅在非 warmup 阶段）"""
        return epoch >= self.warmup_epochs
    
    def should_update_weights(self, epoch: int) -> bool:
        """是否应该更新权重（warmup, wake, sleep_replay）"""
        phase = self.get_phase(epoch)
        return phase in ["warmup", "wake", "sleep_replay"]
    
    def should_use_replay(self, epoch: int) -> bool:
        """是否应该使用回放数据"""
        return self.get_phase(epoch) == "sleep_replay"


__all__ = ["ReplayBuffer", "WakeSleepScheduler", "Experience"]

