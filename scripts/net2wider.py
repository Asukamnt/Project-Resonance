#!/usr/bin/env python3
"""
Net2Wider: 无损网络宽度扩张

实现 Net2Net 论文中的 Net2Wider 操作，用于增加隐藏层维度。
支持批量扩张 MiniJMamba 的 SSM block。

Reference: Chen et al., "Net2Net: Accelerating Learning via Knowledge Transfer" (ICLR 2016)

Author: Jericho Team
Date: 2026-01-02
"""

import math
import copy
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn


def widen_linear(
    old: nn.Linear,
    factor: float,
    noise_std: float = 0.01,
) -> nn.Linear:
    """
    扩张 Linear 层的输出维度
    
    Parameters
    ----------
    old : nn.Linear
        原始线性层
    factor : float
        扩张因子，例如 1.5 表示 128 -> 192
    noise_std : float
        添加到新通道的噪声标准差，防止梯度完全对称
    
    Returns
    -------
    nn.Linear
        扩张后的线性层
    """
    d_in = old.in_features
    d_out_old = old.out_features
    d_out_new = int(d_out_old * factor)
    
    new = nn.Linear(d_in, d_out_new, bias=old.bias is not None)
    
    with torch.no_grad():
        # 复制原有权重
        new.weight.data[:d_out_old] = old.weight.data.clone()
        
        # 新通道使用 Kaiming 初始化 + 小噪声
        nn.init.kaiming_uniform_(new.weight.data[d_out_old:], a=math.sqrt(5))
        new.weight.data[d_out_old:] *= noise_std  # 缩小初始影响
        
        if old.bias is not None:
            new.bias.data[:d_out_old] = old.bias.data.clone()
            new.bias.data[d_out_old:] = 0.0  # 新通道 bias 从 0 开始
    
    return new


def widen_linear_input(
    old: nn.Linear,
    factor: float,
) -> nn.Linear:
    """
    扩张 Linear 层的输入维度（用于下游层）
    
    Parameters
    ----------
    old : nn.Linear
        原始线性层
    factor : float
        扩张因子
    
    Returns
    -------
    nn.Linear
        输入维度扩张后的线性层
    """
    d_in_old = old.in_features
    d_in_new = int(d_in_old * factor)
    d_out = old.out_features
    
    new = nn.Linear(d_in_new, d_out, bias=old.bias is not None)
    
    with torch.no_grad():
        # 复制原有权重
        new.weight.data[:, :d_in_old] = old.weight.data.clone()
        # 新输入维度的权重初始化为 0（保持功能等价）
        new.weight.data[:, d_in_old:] = 0.0
        
        if old.bias is not None:
            new.bias.data = old.bias.data.clone()
    
    return new


def widen_layernorm(
    old: nn.LayerNorm,
    factor: float,
) -> nn.LayerNorm:
    """
    扩张 LayerNorm 的维度
    
    Parameters
    ----------
    old : nn.LayerNorm
        原始 LayerNorm
    factor : float
        扩张因子
    
    Returns
    -------
    nn.LayerNorm
        扩张后的 LayerNorm
    """
    if isinstance(old.normalized_shape, int):
        d_old = old.normalized_shape
    else:
        d_old = old.normalized_shape[0]
    
    d_new = int(d_old * factor)
    new = nn.LayerNorm(d_new, eps=old.eps, elementwise_affine=old.elementwise_affine)
    
    if old.elementwise_affine:
        with torch.no_grad():
            new.weight.data[:d_old] = old.weight.data.clone()
            new.weight.data[d_old:] = 1.0  # 新通道 gamma = 1
            new.bias.data[:d_old] = old.bias.data.clone()
            new.bias.data[d_old:] = 0.0  # 新通道 beta = 0
    
    return new


def widen_embedding(
    old: nn.Embedding,
    factor: float,
    noise_std: float = 0.01,
) -> nn.Embedding:
    """
    扩张 Embedding 层的维度
    """
    num_embeddings = old.num_embeddings
    d_old = old.embedding_dim
    d_new = int(d_old * factor)
    
    new = nn.Embedding(num_embeddings, d_new, padding_idx=old.padding_idx)
    
    with torch.no_grad():
        new.weight.data[:, :d_old] = old.weight.data.clone()
        nn.init.normal_(new.weight.data[:, d_old:], std=noise_std)
    
    return new


class Net2Wider:
    """
    Net2Wider 批量扩张工具
    
    用于扩张整个 MiniJMamba 模型的隐藏维度。
    """
    
    def __init__(self, factor: float = 1.5, noise_std: float = 0.01):
        """
        Parameters
        ----------
        factor : float
            扩张因子，例如 1.5 表示 128 -> 192
        noise_std : float
            添加到新通道的噪声标准差
        """
        self.factor = factor
        self.noise_std = noise_std
        self.widen_log = []
    
    def widen_module(self, module: nn.Module, name: str = "") -> nn.Module:
        """
        递归扩张模块
        
        Parameters
        ----------
        module : nn.Module
            要扩张的模块
        name : str
            模块名称（用于日志）
        
        Returns
        -------
        nn.Module
            扩张后的模块
        """
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child, nn.Linear):
                # 扩张线性层的输出维度
                new_child = widen_linear(child, self.factor, self.noise_std)
                setattr(module, child_name, new_child)
                self.widen_log.append({
                    "name": full_name,
                    "type": "Linear",
                    "old_shape": f"{child.in_features} -> {child.out_features}",
                    "new_shape": f"{new_child.in_features} -> {new_child.out_features}",
                })
            
            elif isinstance(child, nn.LayerNorm):
                new_child = widen_layernorm(child, self.factor)
                setattr(module, child_name, new_child)
                self.widen_log.append({
                    "name": full_name,
                    "type": "LayerNorm",
                    "old_dim": child.normalized_shape,
                    "new_dim": new_child.normalized_shape,
                })
            
            elif isinstance(child, nn.Embedding):
                new_child = widen_embedding(child, self.factor, self.noise_std)
                setattr(module, child_name, new_child)
                self.widen_log.append({
                    "name": full_name,
                    "type": "Embedding",
                    "old_dim": child.embedding_dim,
                    "new_dim": new_child.embedding_dim,
                })
            
            else:
                # 递归处理子模块
                self.widen_module(child, full_name)
        
        return module
    
    def widen_mini_jmamba(
        self,
        model: nn.Module,
        widen_ssm: bool = True,
        widen_attn: bool = False,  # 默认不扩张 Attention
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        扩张 MiniJMamba 模型
        
        Parameters
        ----------
        model : nn.Module
            MiniJMamba 模型
        widen_ssm : bool
            是否扩张 SSM block
        widen_attn : bool
            是否扩张 Attention block（建议保持 False 节省显存）
        
        Returns
        -------
        model : nn.Module
            扩张后的模型
        info : dict
            扩张信息
        """
        self.widen_log = []
        
        # 扩张输入投影
        if hasattr(model, 'input_proj'):
            old_proj = model.input_proj
            if isinstance(old_proj, nn.Linear):
                new_proj = widen_linear(old_proj, self.factor, self.noise_std)
                model.input_proj = new_proj
                self.widen_log.append({
                    "name": "input_proj",
                    "type": "Linear",
                    "old_shape": f"{old_proj.in_features} -> {old_proj.out_features}",
                    "new_shape": f"{new_proj.in_features} -> {new_proj.out_features}",
                })
            elif isinstance(old_proj, nn.Sequential):
                for i, layer in enumerate(old_proj):
                    if isinstance(layer, nn.Linear):
                        new_layer = widen_linear(layer, self.factor, self.noise_std)
                        old_proj[i] = new_layer
        
        # 扩张 SSM layers
        if widen_ssm and hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                layer_name = f"layers.{i}"
                # SSMLikeBlock 内部的线性层
                self.widen_module(layer, layer_name)
        
        # 扩张最终 LayerNorm
        if hasattr(model, 'final_norm'):
            old_norm = model.final_norm
            if isinstance(old_norm, nn.LayerNorm):
                new_norm = widen_layernorm(old_norm, self.factor)
                model.final_norm = new_norm
                self.widen_log.append({
                    "name": "final_norm",
                    "type": "LayerNorm",
                    "old_dim": old_norm.normalized_shape,
                    "new_dim": new_norm.normalized_shape,
                })
        
        # 扩张符号头（输入维度）
        if hasattr(model, 'symbol_head'):
            old_head = model.symbol_head
            if isinstance(old_head, nn.Linear):
                new_head = widen_linear_input(old_head, self.factor)
                model.symbol_head = new_head
                self.widen_log.append({
                    "name": "symbol_head",
                    "type": "Linear (input)",
                    "old_shape": f"{old_head.in_features} -> {old_head.out_features}",
                    "new_shape": f"{new_head.in_features} -> {new_head.out_features}",
                })
        
        info = {
            "factor": self.factor,
            "noise_std": self.noise_std,
            "widen_log": self.widen_log,
            "num_widened": len(self.widen_log),
        }
        
        return model, info


def widen_checkpoint(
    checkpoint_path: str,
    output_path: str,
    factor: float = 1.5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    扩张 checkpoint 文件
    
    Parameters
    ----------
    checkpoint_path : str
        原始 checkpoint 路径
    output_path : str
        输出 checkpoint 路径
    factor : float
        扩张因子
    device : str
        设备
    
    Returns
    -------
    dict
        扩张信息
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from jericho.models.mini_jmamba import MiniJMamba, MiniJMambaConfig
    
    # 加载原始 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = ckpt["config"]
    
    # 创建原始模型
    d_model_old = saved_config.get("d_model", 128)
    d_model_new = int(d_model_old * factor)
    
    config = MiniJMambaConfig(
        frame_size=saved_config.get("frame_size", 160),
        hop_size=saved_config.get("hop_size", 160),
        symbol_vocab_size=saved_config.get("symbol_vocab_size", 12),
        d_model=d_model_old,
        num_ssm_layers=saved_config.get("num_ssm_layers", 10),
        num_attn_layers=saved_config.get("num_attn_layers", 2),
        max_frames=saved_config.get("max_frames", 256),
        use_rope=saved_config.get("use_rope", True),
    )
    
    model = MiniJMamba(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    # 扩张模型
    widener = Net2Wider(factor=factor)
    model, widen_info = widener.widen_mini_jmamba(model)
    
    # 更新 config
    new_config = dict(saved_config)
    new_config["d_model"] = d_model_new
    new_config["widened_from"] = d_model_old
    new_config["widen_factor"] = factor
    
    # 保存新 checkpoint
    new_ckpt = {
        "model_state_dict": model.state_dict(),
        "config": new_config,
        "backbone": ckpt.get("backbone", "mini_jmamba"),
        "widen_info": widen_info,
        "original_checkpoint": checkpoint_path,
    }
    
    # 复制其他元数据
    for key in ["symbol_to_id", "id_to_symbol", "task", "epochs", "seed"]:
        if key in ckpt:
            new_ckpt[key] = ckpt[key]
    
    torch.save(new_ckpt, output_path)
    
    print(f"Widened checkpoint saved to {output_path}")
    print(f"  d_model: {d_model_old} -> {d_model_new}")
    print(f"  Widened {widen_info['num_widened']} modules")
    
    return widen_info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Net2Wider: Widen MiniJMamba checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Input checkpoint path")
    parser.add_argument("--output", type=str, required=True,
                       help="Output checkpoint path")
    parser.add_argument("--factor", type=float, default=1.5,
                       help="Widen factor (default: 1.5)")
    args = parser.parse_args()
    
    widen_checkpoint(args.checkpoint, args.output, args.factor)

