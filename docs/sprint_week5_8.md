# Week 5-8 两周冲刺清单

> **目标**：把"成功"变成"可发表/可吸引资源"  
> **策略**：负对照 + 消融 + 一键总报告 + 发布资产化

---

## 📊 冲刺总览

| 阶段 | 天数 | 核心目标 | 验收标准 |
|------|------|----------|----------|
| Sprint 1 (Day 1-3) | 3天 | 负对照 + 消融套件 | S7 + S22 完成 |
| Sprint 2 (Day 4-6) | 3天 | Task3 多步 + OOD | 组合 mod + 3轴 OOD |
| Sprint 3 (Day 7-10) | 4天 | 一键总评估 + 发布资产 | S1/S19-S26 完成 |

---

## 🔴 Sprint 1: 科研硬证据（Day 1-3）

### Day 1: S7 负对照套件

**目标**：证明模型不走捷径

| 任务 | 落点 | 验收 | 止损 |
|------|------|------|------|
| label_shuffle 负对照 | `scripts/negative_controls.py` | 模型在 shuffle 数据上 acc ≈ random | 如果 acc > random+0.1，说明有信息泄露 |
| phase_scramble 负对照 | 同上 | 打乱相位后模型无法解码 | 如果仍能解码，说明未用相位信息 |
| random_mapping 负对照 | 同上 | 符号→频率随机映射后 acc ≈ random | 证明依赖真实映射 |

**输出产物**：
```
reports/negative_controls.md
- Task1 label_shuffle: acc = 0.xx (expected ~0.10)
- Task1 phase_scramble: acc = 0.xx (expected ~0.10)
- Task2 label_shuffle: acc = 0.xx (expected ~0.50)
- Task3 label_shuffle: acc = 0.xx (expected ~0.10)
```

**命令模板**：
```bash
python scripts/negative_controls.py --task mirror --control label_shuffle --model runs/best_task1/checkpoint.pt --output reports/negative_controls.md
```

### Day 2-3: S22 消融套件

**目标**：证明关键组件必要性

| 消融实验 | 变量 | 预期影响 | 验收 |
|----------|------|----------|------|
| 无 Attention | 删除 2 层 Attention | OOD 显著下降 | IID/OOD 对比表 |
| 无 RoPE | 换回 learnable pos | OOD-length 崩溃 | OOD-length 曲线 |
| 无 CTC 辅助 | `ctc_weight=0` | 收敛变慢/不稳 | 训练曲线对比 |
| 无课程 | 直接混合训练 | 多任务干扰 | 最终指标对比 |
| 输入表示 | Mel vs STFT | 略微差异 | 指标对比 |

**输出产物**：
```
reports/ablations.csv
experiments/ablation_configs/
  - no_attention.yaml
  - no_rope.yaml
  - no_ctc.yaml
  - no_curriculum.yaml
  - mel_input.yaml
```

**命令模板**：
```bash
python experiments/run_ablations.py --suite core5 --report reports/ablations.csv
```

---

## 🟡 Sprint 2: Task3 多步 + OOD（Day 4-6）

### Day 4: Task3 多步组合

**目标**：A%B%C 多步 mod

| 任务 | 落点 | 验收 | 止损 |
|------|------|------|------|
| 数据生成器支持多步 | `make_task3_manifest.py` | `steps=2,3` 参数 | 如果训练崩，先做 steps=2 |
| 训练管线适配 | `task3_mod_audio.py` | 多步样本可训练 | 如果 OOM，减小 batch |
| 评测口径明确 | `evaluate.py` | 分步骤 EM 报告 | 如果全错，检查渲染 |

**输出产物**：
```
manifests/task3_compose.jsonl  # 含 1-3 步组合
runs/task3_compose/metrics.json
```

### Day 5-6: Task3 多轴 OOD

**目标**：3 个 OOD 轴

| OOD 轴 | 定义 | 预期 | 止损 |
|--------|------|------|------|
| ood_digits | 训练 0-7，测试 8-9 | EM > baseline | 如果崩，检查频率映射 |
| ood_length | 训练 len≤2，测试 len=3,4 | EM > 0.5 × IID | 如果崩，检查 RoPE |
| ood_compose | 训练 steps=1，测试 steps=2 | EM > 0.3 × IID | 需要多步训练作为对照 |

**输出产物**：
```
reports/task3_ood_summary.md
- ood_digits: EM = 0.xx
- ood_length: EM = 0.xx
- ood_compose: EM = 0.xx
```

---

## 🟢 Sprint 3: 发布资产化（Day 7-10）

### Day 7: S1/S19/S20 一键总评估

**目标**：外部复现者一条命令看全貌

| 任务 | 落点 | 验收 |
|------|------|------|
| 总评估脚本 | `evaluate.py --stage final` | 输出 JSON + Markdown |
| 全 split 报告 | `reports/ood_summary.md` | 3 任务 × N splits |
| 指标协议文档 | `docs/metrics_protocol.md` | EM/Token/Edit 定义 |

**输出产物**：
```
reports/
  - system_overview_final.json
  - ood_summary.md
  - test_metrics.json
```

**命令模板**：
```bash
python evaluate.py --stage final --tasks mirror bracket mod --no_text --report reports/system_overview_final.json
```

### Day 8: S23 风险日志

**目标**：10 大风险 + 缓解状态

| 风险 | 缓解措施 | 状态 |
|------|----------|------|
| 过拟合/捷径 | 负对照 + OOD | ✅ 已验证 |
| 辅助作弊 | CTC 权重控制 | ✅ 已实现 |
| OOM | 梯度累积 + batch 调节 | ✅ 已实现 |
| 训练不稳 | 梯度裁剪 + warmup | ✅ 已实现 |
| 容量不足 | 可扩展架构 | ⏳ 待验证 |
| 多任务干扰 | Subject-Selector | ⏳ 待实现 |
| 评分器偏差 | 多口径验证 | ✅ 已验证 |
| 进度延误 | 周计划跟踪 | ✅ 已控制 |
| SSM 实现问题 | Mamba-2 原版 | ✅ 已验证 |
| 音频质量 | STFT + L1 损失 | ⏳ 可改进 |

**输出产物**：
```
docs/risk_log.md
```

### Day 9: S25/S26 里程碑 + 复现资产

**目标**：发布级完整性

| 任务 | 落点 | 验收 |
|------|------|------|
| 里程碑日志 | `docs/milestone_log.md` | Week1-8 条目 |
| 复现种子 | `docs/repro_seeds.json` | 固定种子列表 |
| 检查点 | `artifacts/checkpoints/` | 最佳模型 |
| 示例音频 | `artifacts/audio_examples/` | 3 任务各 5 例 |

**输出产物**：
```
docs/milestone_log.md
docs/repro_seeds.json
artifacts/
  - checkpoints/
    - task1_best.pt
    - task2_best.pt
    - task3_best.pt
  - audio_examples/
    - task1_mirror_01.wav
    - task2_bracket_01.wav
    - task3_mod_01.wav
```

### Day 10: 最终整合 + README 更新

**目标**：对外发布就绪

| 任务 | 落点 | 验收 |
|------|------|------|
| README 结果表 | `README.md` | 3 任务 IID/OOD 表格 |
| 快速开始指南 | `README.md` | 3 条复现命令 |
| 仓库结构校验 | 手动检查 | S24 条目全满足 |

---

## 📋 每日 Checklist 模板

```markdown
### Day X Checklist

- [ ] 任务 1: xxx
  - 落点: xxx
  - 命令: `xxx`
  - 验收: xxx
  - 状态: ⏳/✅/❌

- [ ] 任务 2: xxx
  ...

**产出物**:
- [ ] `reports/xxx.md`
- [ ] `scripts/xxx.py`

**止损决策点**:
- 如果 xxx，则 xxx
```

---

## 🎯 关键决策点

### Sprint 1 决策点
- **负对照失败**（acc >> random）：检查数据生成是否有信息泄露
- **消融无显著差异**：可能需要更极端的消融设置

### Sprint 2 决策点
- **多步组合 OOM**：降低 batch size 或 sequence length
- **OOD 全崩**：检查 RoPE 实现或训练分布

### Sprint 3 决策点
- **复现失败**：固定更多随机源（numpy/torch/cuda）
- **发布阻断**：优先修复阻断项，延后锦上添花

---

## 📊 冲刺结束验收表

| Spec 条目 | 状态 | 产物 |
|-----------|------|------|
| S1 Final Gate | ⏳ | `reports/system_overview_final.json` |
| S7 负对照 | ⏳ | `reports/negative_controls.md` |
| S19 指标报告 | ⏳ | `reports/test_metrics.json` |
| S20 OOD 汇总 | ⏳ | `reports/ood_summary.md` |
| S21 错误分析 | ⏳ | `reports/error_analysis.md` |
| S22 消融 | ⏳ | `reports/ablations.csv` |
| S23 风险日志 | ⏳ | `docs/risk_log.md` |
| S25 里程碑 | ⏳ | `docs/milestone_log.md` |
| S26 复现资产 | ⏳ | `artifacts/` |

---

## 📅 时间线

```
Week 1 (Day 1-7):
  Day 1: S7 负对照
  Day 2-3: S22 消融套件
  Day 4: Task3 多步
  Day 5-6: Task3 OOD
  Day 7: S1/S19/S20 一键总评估

Week 2 (Day 8-10):
  Day 8: S23 风险日志
  Day 9: S25/S26 里程碑 + 资产
  Day 10: 最终整合
```

---

## 🚀 开始命令

```bash
# 确认当前状态
pytest tests/ -q  # 应该全绿

# 开始 Sprint 1 - Day 1
python scripts/negative_controls.py --task mirror --control label_shuffle
```

