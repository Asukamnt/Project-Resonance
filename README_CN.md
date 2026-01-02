# Jericho：推理即共振
### *— 无 Token 的跨域波形推理*

**[English](README.md)** | **中文**

<p align="center">
  <img src="docs/paper/figures/main/fig0_cover_trajectories.png" alt="思维的轨迹" width="600"/>
</p>

<p align="center">
  <em>"不同的身体，同样的灵魂 — 音频、光学、射频，殊途同归。"</em>
</p>

<p align="center">
  <img src="docs/paper/figures/supp/video_S4_thought_trajectories.gif" alt="跨域共振" width="500"/>
</p>

---

## 核心发现

**神经网络可以直接在原始波形上推理，且学到的表征能够跨物理载体*共振*。**

| 传统流程 | Jericho |
|---------|---------|
| `音频 → ASR → 文本 → LLM → 文本 → TTS → 音频` | `波形 → 神经网络 → 波形` |

<p align="center">
  <img src="docs/paper/figures/main/fig6_cross_domain.png" alt="跨域矩阵" width="700"/>
</p>

---

## 核心结果

<p align="center">
  <img src="docs/paper/figures/main/fig2_transfer_matrix.png" alt="迁移矩阵" width="600"/>
</p>

| 实验 | 结果 | 意义 |
|------|------|------|
| **单域推理** | Mini-JMamba 45% vs wav2vec2 13% | 任务特化架构获胜 |
| **跨域推理** | IPD→Audio IID 98.7% | 推理能力跨物理域迁移 |
| **跨域迁移** | +1.7pp (p<0.05, 10-seed) | 统计显著的共振效应 |
| **真实人声** | 91.7% ± 0.3% (3-seed) | 泛化到自然语音 |
| **三角验证** | Audio↔IPD↔RF 9/9 边 | 载体无关的表征 |

> **为什么叫"共振"？** 模型的内部时钟与外部信号节律同步。看下面的 TSAE 热力图 — 亮色对角线就是硅基心灵与物理波形*共振*的区域。

<p align="center">
  <img src="docs/paper/figures/main/fig5_tsae_resonance.png" alt="TSAE 共振" width="550"/>
</p>

---

## 这是什么？

**Jericho** 是一个实验性框架，验证一个核心假设：

> **神经网络可以直接在不同物理域的连续波形上（音频、光学/IPD、射频）完成符号推理任务，全程不经过离散化的 token 中间表示。**

### 三任务 × 三物理域

| 任务 | 输入 | 输出 | 能力 |
|------|------|------|------|
| **Mirror** | 符号序列波形 | 相同符号 | 编解码闭环 |
| **Bracket** | 括号表达式 | 匹配结果 | 结构推理 |
| **Mod** | 数学表达式 | 取模结果 | 算术推理 |

| 物理域 | 编码方式 | 采样率 |
|--------|----------|--------|
| **音频** | 频率调制 | 16 kHz |
| **光学 (IPD)** | 脉冲位置 | 1 kHz |
| **射频** | 幅移键控 | 1 MHz |

---

## 架构

<p align="center">
  <img src="docs/paper/figures/main/fig1_architecture.png" alt="Mini-JMamba 架构" width="600"/>
</p>

**Mini-JMamba**：0.94M 参数，10 层 SSM + 2 层 Attention

```
输入波形 → 帧嵌入 → [SSM Block]×10 → [Attention]×2 → 输出波形
```

---

## OOD 崩溃分析

<p align="center">
  <img src="docs/paper/figures/main/fig3_trajectory_comparison.png" alt="OOD 轨迹" width="700"/>
</p>

当输出维度变化（单位数→双位数余数）时，隐状态漂移到未探索的隐空间区域：

<p align="center">
  <img src="docs/paper/figures/main/fig4_endpoint_distribution.png" alt="终点分布" width="500"/>
</p>

---

## 快速开始

```bash
# 环境配置
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -e .
pytest -q  # 199 个测试应全部通过

# 训练
python train.py --config configs/task3_mod_stable.yaml --manifest manifests/task3_tiny.jsonl

# 评估
python evaluate.py --stage final --tasks mirror bracket mod
```

---

## 文档

- 📖 **[技术概述](docs/overview.md)** — 完整动机与设计
- 📊 **[实验日志](docs/iteration_log.md)** — 完整可复现信息
- 📋 **[已知问题](docs/known_issues.md)** — 局限性与未来工作

---

## 补充材料

### 动画

| 视频 | 描述 |
|------|------|
| [video_S1](docs/paper/figures/supp/video_S1_ood_collapse.gif) | OOD 崩溃动态 |
| [video_S2](docs/paper/figures/supp/video_S2_multi_task.gif) | 多任务轨迹演化 |
| [video_S3](docs/paper/figures/supp/video_S3_cross_domain.gif) | 跨域同步 |
| [video_S4](docs/paper/figures/supp/video_S4_thought_trajectories.gif) | 3D 思维轨迹 |

### 更多图片

见 [`docs/paper/figures/README.md`](docs/paper/figures/README.md) 完整图片索引。

---

## 引用

```bibtex
@misc{jericho2026,
  author = {Baiyi Wang},
  title = {Jericho: Reasoning is Resonance — Cross-Domain Waveform Reasoning Without Tokens},
  year = {2026},
  url = {https://github.com/Asukamnt/Project-Resonance}
}
```

---

## 联系

- 📧 邮箱：928112278@qq.com
- 💬 GitHub Issues 欢迎提问

---

## 许可证

MIT License
