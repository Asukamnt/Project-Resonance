# 论文证据索引

本文档汇总所有可用于论文的实验证据位置。

---

## 核心结果

### 1. 单域推理（Phase 1）

| 证据 | 位置 | 关键数字 |
|------|------|----------|
| Task1 Mirror EM=100% | `docs/iteration_log.md` Week 3 | EM=1.0 |
| Task2 Bracket IID=96% | `docs/iteration_log.md` Week 5 | audio_acc=0.96 |
| Task2 OOD-length=84% | `docs/iteration_log.md` Week 5 | +34pp vs baseline |
| Task3 Mod EM=45% | `docs/iteration_log.md` Week 4 | disjoint holdout |

### 2. 跨域推理（Phase 3）

| 证据 | 位置 | 关键数字 |
|------|------|----------|
| IPD→Audio IID=98.7% | `docs/iteration_log.md` #26 | 3-seed mean |
| IPD→Audio OOD=67.3% | `docs/iteration_log.md` #26 | 3/3 seeds pass |

### 3. 跨域迁移（Phase 4）

| 证据 | 位置 | 关键数字 |
|------|------|----------|
| Audio→IPD +1.7pp | `reports/bootstrap_ci_audio_to_ipd.json` | 95% CI [+0.1, +3.4] |
| Audio→RF +9 epochs | `docs/iteration_log.md` Phase 4+ | 收敛加速 |
| 三角验证 6/6 | `docs/iteration_log.md` Phase 4+ | 完整矩阵 |

### 4. 统计显著性（P0 修复）

| 证据 | 位置 | 关键数字 |
|------|------|----------|
| 10-seed Bootstrap CI | `reports/bootstrap_ci_audio_to_ipd.json` | p<0.05 |
| 随机映射负对照 | `reports/random_mapping_control.json` | ΔEM=48pp |
| STFT-SDR 波形质量 | `reports/stft_sdr_results.json` | 32.84 dB |

### 5. Baseline 对比

| 证据 | 位置 | 关键数字 |
|------|------|----------|
| wav2vec2 vs Mini-JMamba | `reports/wav2vec2_baseline_summary.json` | 22% vs 45% |
| Transformer vs Mini-JMamba | `docs/iteration_log.md` #19 | 41% vs 45% |
| LSTM vs Mini-JMamba | `docs/iteration_log.md` #19 | 42% vs 45% |

### 6. Ablation Studies

| 证据 | 位置 | 关键数字 |
|------|------|----------|
| Thinking Gap | `reports/ablation_thinking_gap.json` | 0.5s 最优 |
| Architecture (SSM/Attn) | `reports/ablation_architecture.json` | 10+2 最优 |
| Channel Noise | `reports/ablation_channel_noise.json` | AWGN 鲁棒 |

### 7. TSAE 发现

| 证据 | 位置 | 关键数字 |
|------|------|----------|
| 频率偏差分析 | `reports/tsae_analysis.json` | -636Hz → -396Hz |
| Hybrid 验证 | `reports/tsae_hybrid_verify.json` | 模型本体效应 |
| 多 Checkpoint | `reports/tsae_multi_checkpoint.json` | 5/5 可复现 |
| Task2 验证 | `reports/tsae_task2.json` | 方向相反 |

---

## 代码证据

### 核心模型
- `src/jericho/models/mini_jmamba.py` - 架构定义

### 训练 Pipeline
- `src/jericho/pipelines/task3_mod_audio.py` - Task3 完整流程
- `src/jericho/pipelines/task2_bracket_audio.py` - Task2 流程

### 评测脚本
- `evaluate.py` - Oracle EM（协议验证）
- `evaluate_model.py` - Model EM（能力评估）

### 复现脚本
- `scripts/repro_tiny.py` - 5分钟最小复现

---

## bin 文件夹有用内容

| 文件 | 可能价值 |
|------|----------|
| `docs/bin/main_track_sprint/04_paper_outline.md` | 论文大纲草稿 |
| `docs/bin/main_track_sprint/review_results.md` | 自我审稿结果 |
| `docs/bin/phase2-4/o3pro_feedback.md` | o3-pro 审稿反馈 |

---

## 推荐论文结构

1. **Introduction**: Wave Reasoning 动机
2. **Related Work**: ASR→NLU vs End-to-End
3. **Method**: Mini-JMamba 架构
4. **Experiments**:
   - 4.1 单域推理（Task1/2/3）
   - 4.2 跨域推理（IPD→Audio）
   - 4.3 跨域迁移（三角验证）
   - 4.4 Ablations
5. **Analysis**: TSAE 发现
6. **Conclusion**

---

## arXiv 发布流程

1. **注册账号**: https://arxiv.org/user/register
2. **获取 endorsement**: 需要已发表作者推荐
3. **选择分类**: cs.LG / cs.CL / cs.SD
4. **上传**: LaTeX/PDF + 源码链接
5. **等待审核**: 1-2 工作日

**注意**: 你需要找一个有 arXiv 发表经历的人帮你做 endorsement。

