"""
L0 Sparse Gate for Learnable Pruning

基于 L0 正则化的可学习稀疏门控。
在训练期自动学习哪些通道应该保留（keep≈0.7）。

Reference: Louizos et al., "Learning Sparse Neural Networks through L0 Regularization" (ICLR 2018)

Author: Jericho Team
Date: 2026-01-02
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class L0Gate(nn.Module):
    """
    L0 稀疏门控层
    
    使用 Hard Concrete 分布实现可微的 L0 正则化。
    训练时通过重参数化技巧采样门控值，推理时使用确定性阈值。
    
    Parameters
    ----------
    dim : int
        输入/输出维度
    droprate_init : float
        初始丢弃率，默认 0.5（即初始 keep ≈ 0.5）
    temperature : float
        Hard Concrete 分布的温度参数
    stretch : Tuple[float, float]
        Hard Concrete 分布的拉伸范围 (gamma, zeta)
    min_keep_ratio : float
        最小保留比例约束，防止过度稀疏化
    target_keep_ratio : float
        目标保留比例，用于约束型正则化
    """
    
    def __init__(
        self,
        dim: int,
        droprate_init: float = 0.5,
        temperature: float = 2.0 / 3.0,
        stretch: Tuple[float, float] = (-0.1, 1.1),
        min_keep_ratio: float = 0.5,
        target_keep_ratio: float = 0.7,
    ):
        super().__init__()
        
        self.dim = dim
        self.temperature = temperature
        self.gamma, self.zeta = stretch
        self.min_keep_ratio = min_keep_ratio
        self.target_keep_ratio = target_keep_ratio
        
        # 可学习的 log_alpha 参数
        # 初始化使得 sigmoid(log_alpha) ≈ 1 - droprate_init
        init_mean = math.log(1 - droprate_init) - math.log(droprate_init)
        self.log_alpha = nn.Parameter(
            torch.empty(dim).normal_(init_mean, 0.01)
        )
        
        # 用于统计的缓冲区
        self.register_buffer("_sparsity_history", torch.zeros(100))
        self.register_buffer("_history_idx", torch.tensor(0, dtype=torch.long))
    
    def _hard_concrete_sample(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """采样 Hard Concrete 分布"""
        u = torch.rand_like(log_alpha).clamp(1e-8, 1 - 1e-8)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / self.temperature)
        # 拉伸到 (gamma, zeta) 并裁剪到 [0, 1]
        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(s_stretched, 0.0, 1.0)
    
    def _hard_concrete_mean(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Hard Concrete 分布的期望（用于推理）"""
        return torch.sigmoid(log_alpha)
    
    def get_expected_sparsity(self) -> float:
        """获取期望的稀疏度（0 表示全保留，1 表示全丢弃）"""
        # P(gate = 0) = sigmoid(log_alpha - temperature * log(-gamma / zeta))
        threshold = self.temperature * math.log(-self.gamma / self.zeta)
        p_zero = torch.sigmoid(self.log_alpha - threshold).mean()
        return 1.0 - p_zero.item()  # keep ratio
    
    def get_l0_loss(self) -> torch.Tensor:
        """计算 L0 正则化损失（期望的非零参数数量）"""
        # P(gate > 0) = 1 - P(gate <= 0) = sigmoid(log_alpha - threshold)
        threshold = self.temperature * math.log(-self.gamma / self.zeta)
        p_nonzero = torch.sigmoid(self.log_alpha - threshold)
        return p_nonzero.sum()
    
    def get_constrained_l0_loss(self) -> torch.Tensor:
        """
        计算约束型 L0 正则化损失
        
        惩罚偏离 target_keep_ratio 的情况，同时保证不低于 min_keep_ratio。
        使用 Huber-like 损失：在目标附近平滑，远离目标时线性增长。
        """
        current_keep = self.get_expected_sparsity()
        target = self.target_keep_ratio
        min_k = self.min_keep_ratio
        
        # 距离目标的偏差
        deviation = abs(current_keep - target)
        
        # 低于最小值的惩罚（更强）
        below_min_penalty = max(0, min_k - current_keep) * 10.0
        
        # 总损失 = 偏差 + 强惩罚
        return torch.tensor(deviation + below_min_penalty, device=self.log_alpha.device)
    
    def get_target_aware_l0_loss(self) -> torch.Tensor:
        """
        目标感知的 L0 正则化损失
        
        鼓励 keep_ratio 接近 target_keep_ratio，而不是简单地最小化非零参数。
        """
        threshold = self.temperature * math.log(-self.gamma / self.zeta)
        p_nonzero = torch.sigmoid(self.log_alpha - threshold)
        current_keep = p_nonzero.mean()
        
        # 双向惩罚：既惩罚过于稀疏，也惩罚不够稀疏
        target = torch.tensor(self.target_keep_ratio, device=self.log_alpha.device)
        loss = (current_keep - target) ** 2
        
        # 低于最小值的额外惩罚
        min_k = torch.tensor(self.min_keep_ratio, device=self.log_alpha.device)
        below_min_penalty = torch.relu(min_k - current_keep) * 10.0
        
        return loss + below_min_penalty
    
    def forward(
        self,
        x: torch.Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状 (batch, seq_len, dim) 或 (batch, dim)
        training : bool, optional
            是否训练模式，默认使用 self.training
        
        Returns
        -------
        gated_x : torch.Tensor
            门控后的张量
        gate_values : torch.Tensor
            门控值，形状 (dim,)
        """
        if training is None:
            training = self.training
        
        if training:
            # 训练时采样
            gate = self._hard_concrete_sample(self.log_alpha)
        else:
            # 推理时使用期望
            gate = self._hard_concrete_mean(self.log_alpha)
            gate = torch.clamp(gate * (self.zeta - self.gamma) + self.gamma, 0.0, 1.0)
        
        # 应用最小保留比例约束：保证至少 min_keep_ratio 的通道被保留
        if self.min_keep_ratio > 0:
            # 计算需要保留的最小通道数
            min_keep_count = int(self.dim * self.min_keep_ratio)
            
            # 如果当前激活的通道太少，强制保留 top-k
            active_count = (gate > 0.5).sum()
            if active_count < min_keep_count:
                # 找到 top-k 个最大的 gate 值
                _, top_indices = torch.topk(gate, min_keep_count)
                # 创建强制保留掩码
                force_keep = torch.zeros_like(gate)
                force_keep[top_indices] = 1.0
                # 混合：使用较大的值
                gate = torch.maximum(gate, force_keep * 0.5)
        
        # 记录稀疏度历史
        if training:
            with torch.no_grad():
                sparsity = (gate > 0.5).float().mean()
                idx = self._history_idx.item() % 100
                self._sparsity_history[idx] = sparsity
                self._history_idx.add_(1)
        
        # 广播门控值
        if x.dim() == 3:
            gate_broadcast = gate.view(1, 1, -1)
        elif x.dim() == 2:
            gate_broadcast = gate.view(1, -1)
        else:
            gate_broadcast = gate
        
        return x * gate_broadcast, gate
    
    def get_sparsity_stats(self) -> dict:
        """获取稀疏度统计"""
        valid_count = min(self._history_idx.item(), 100)
        if valid_count == 0:
            return {"mean_keep": 1.0, "std_keep": 0.0, "expected_keep": self.get_expected_sparsity()}
        
        history = self._sparsity_history[:valid_count]
        return {
            "mean_keep": history.mean().item(),
            "std_keep": history.std().item() if valid_count > 1 else 0.0,
            "expected_keep": self.get_expected_sparsity(),
        }
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, temp={self.temperature}, gamma={self.gamma}, zeta={self.zeta}"


class GatedLinear(nn.Module):
    """
    带 L0 Gate 的线性层
    
    在输出通道上应用 L0 Gate，实现可学习的通道稀疏化。
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        droprate_init: float = 0.5,
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.gate = L0Gate(out_features, droprate_init=droprate_init)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        output : torch.Tensor
        gate_values : torch.Tensor
        """
        h = self.linear(x)
        return self.gate(h)
    
    def get_l0_loss(self) -> torch.Tensor:
        return self.gate.get_l0_loss()
    
    def get_effective_dim(self) -> int:
        """获取有效（非零）维度数"""
        with torch.no_grad():
            gate = self.gate._hard_concrete_mean(self.gate.log_alpha)
            gate_clipped = torch.clamp(gate * (self.gate.zeta - self.gate.gamma) + self.gate.gamma, 0, 1)
            return (gate_clipped > 0.5).sum().item()


class L0GatedBlock(nn.Module):
    """
    可热插拔的 L0 Gate 包装器
    
    用于包装现有的 SSM/Attention block，添加输出门控。
    """
    
    def __init__(
        self,
        block: nn.Module,
        dim: int,
        droprate_init: float = 0.5,
        enabled: bool = True,
    ):
        super().__init__()
        
        self.block = block
        self.gate = L0Gate(dim, droprate_init=droprate_init)
        self.enabled = enabled
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        h = self.block(x, *args, **kwargs)
        
        if self.enabled:
            h, _ = self.gate(h)
        
        return h
    
    def get_l0_loss(self) -> torch.Tensor:
        if self.enabled:
            return self.gate.get_l0_loss()
        return torch.tensor(0.0)


__all__ = ["L0Gate", "GatedLinear", "L0GatedBlock"]

