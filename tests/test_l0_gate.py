"""
Tests for L0 Sparse Gate module

Author: Jericho Team
Date: 2026-01-02
"""

import pytest
import torch
import torch.nn as nn

from jericho.models.modules.l0_gate import L0Gate, GatedLinear, L0GatedBlock


class TestL0Gate:
    """L0Gate 单元测试"""
    
    def test_init(self):
        """测试初始化"""
        gate = L0Gate(dim=128, droprate_init=0.5)
        
        assert gate.dim == 128
        assert gate.log_alpha.shape == (128,)
        assert gate.temperature == 2.0 / 3.0
    
    def test_forward_train(self):
        """测试训练模式前向传播"""
        gate = L0Gate(dim=64)
        gate.train()
        
        x = torch.randn(2, 10, 64)  # (batch, seq_len, dim)
        out, gate_values = gate(x)
        
        assert out.shape == x.shape
        assert gate_values.shape == (64,)
        assert (gate_values >= 0).all()
        assert (gate_values <= 1).all()
    
    def test_forward_eval(self):
        """测试推理模式前向传播"""
        gate = L0Gate(dim=64)
        gate.eval()
        
        x = torch.randn(2, 10, 64)
        out1, gate1 = gate(x)
        out2, gate2 = gate(x)
        
        # 推理模式应该是确定性的
        assert torch.allclose(out1, out2)
        assert torch.allclose(gate1, gate2)
    
    def test_l0_loss(self):
        """测试 L0 正则化损失"""
        gate = L0Gate(dim=128)
        
        l0_loss = gate.get_l0_loss()
        
        assert l0_loss.shape == ()
        assert l0_loss.item() > 0
        assert l0_loss.item() <= 128  # 最多 128 个非零参数
    
    def test_sparsity_stats(self):
        """测试稀疏度统计"""
        gate = L0Gate(dim=64, droprate_init=0.3)
        gate.train()
        
        # 运行几次前向传播以收集统计
        x = torch.randn(2, 10, 64)
        for _ in range(10):
            gate(x)
        
        stats = gate.get_sparsity_stats()
        
        assert "mean_keep" in stats
        assert "std_keep" in stats
        assert "expected_keep" in stats
        assert 0 <= stats["expected_keep"] <= 1
    
    def test_gradient_flow(self):
        """测试梯度流动"""
        gate = L0Gate(dim=32)
        gate.train()
        
        x = torch.randn(2, 5, 32, requires_grad=True)
        out, _ = gate(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert gate.log_alpha.grad is not None


class TestGatedLinear:
    """GatedLinear 单元测试"""
    
    def test_init(self):
        """测试初始化"""
        layer = GatedLinear(64, 128)
        
        assert layer.linear.in_features == 64
        assert layer.linear.out_features == 128
        assert layer.gate.dim == 128
    
    def test_forward(self):
        """测试前向传播"""
        layer = GatedLinear(64, 128)
        
        x = torch.randn(2, 10, 64)
        out, gate_values = layer(x)
        
        assert out.shape == (2, 10, 128)
        assert gate_values.shape == (128,)
    
    def test_l0_loss(self):
        """测试 L0 损失"""
        layer = GatedLinear(32, 64)
        
        l0_loss = layer.get_l0_loss()
        assert l0_loss.shape == ()
        assert l0_loss.item() > 0
    
    def test_effective_dim(self):
        """测试有效维度计算"""
        layer = GatedLinear(32, 64, droprate_init=0.5)
        
        eff_dim = layer.get_effective_dim()
        assert 0 <= eff_dim <= 64


class TestL0GatedBlock:
    """L0GatedBlock 单元测试"""
    
    def test_wrap_linear(self):
        """测试包装线性层"""
        linear = nn.Linear(64, 64)
        gated = L0GatedBlock(linear, dim=64)
        
        x = torch.randn(2, 10, 64)
        out = gated(x)
        
        assert out.shape == x.shape
    
    def test_enabled_flag(self):
        """测试启用/禁用门控"""
        linear = nn.Linear(64, 64)
        
        # 启用门控
        gated_enabled = L0GatedBlock(linear, dim=64, enabled=True)
        x = torch.randn(2, 10, 64)
        out_enabled = gated_enabled(x)
        
        # 禁用门控
        gated_disabled = L0GatedBlock(linear, dim=64, enabled=False)
        out_disabled = gated_disabled(x)
        
        # 禁用时应该等于原始线性层输出
        assert out_disabled.shape == x.shape
        
        # L0 loss 在禁用时应该为 0
        assert gated_disabled.get_l0_loss().item() == 0
        assert gated_enabled.get_l0_loss().item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

