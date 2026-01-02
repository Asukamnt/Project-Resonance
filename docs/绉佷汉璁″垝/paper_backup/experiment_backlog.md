# 论文必补实验清单

**状态**：P0 实验进行中  
**预计周期**：2-3 周  
**更新时间**：2026-01-01 02:30

---

## 🔴 P0 必做（阻塞论文提交）

### 1. wav2vec2 / HuBERT 公平微调
**预计时间**：3 天  
**状态**：🔄 进行中

| 项目 | 要求 |
|------|------|
| 任务覆盖 | Task1 Mirror, Task2 Bracket, Task3 Mod |
| 训练设置 | 与 Mini-JMamba 相同（数据量、epoch、早停） |
| 报告指标 | EM ± std（3-5 seeds）, 95% CI |
| 对比维度 | 参数量、训练时间、GPU 显存 |

**目的**：确保 baseline 对比公平，堵住"设置不同"的质疑

**当前状态**：
- [x] wav2vec2 Task3 冻结版已跑（22% EM）
- [x] 公平微调脚本 `scripts/wav2vec2_finetune_fair.py`
- [🔄] **3设置×3种子实验运行中**（frozen/partial/full）
  - 终端: `terminals/9.txt`
  - 输出: `reports/wav2vec2_finetune_fair.json`
- [ ] HuBERT 待补（如果 wav2vec2 结果不足以说服）
- [ ] Task1/Task2 待补（如有时间）

---

### 2. 真实硬件最小验证
**预计时间**：3 天  
**状态**：⚠️ Protocol 验证完成，Model 验证待做

| 项目 | 要求 |
|------|------|
| 链路 | Audio: 扬声器 → 空气传播 → 话筒 |
| 任务 | Task1 Mirror（最简单） |
| 报告 | 通过/失败 + SNR/延迟测量 |
| 设备 | 普通 USB 话筒/扬声器即可 |

**目的**：证明不仅仅是"合成数据上的玩具"

**当前状态**：
- [x] 信道模拟脚本 `scripts/real_hardware_test.py`
- [x] SNR 扫描测试完成（**但这只是 Protocol 验证！**）
- [ ] **Model 信道鲁棒性验证**（需要 checkpoint + 模型 forward）
- [ ] 硬件录放测试（可选，如果有设备）

**⚠️ 重要澄清**：

`real_hardware_test.py` 测试的是：
```
Oracle 编码 → 信道干扰 → FFT 解码
```
**这验证的是 Protocol，不是 Model！**

**初步结果（Protocol 验证）**：

| SNR (dB) | Protocol EM | 说明 |
|----------|-------------|------|
| 40-0 dB | **100%** | FFT 解码器对噪声极其鲁棒 |

**结论**：编码/解码协议本身可以在极端噪声下工作。这为真实硬件部署提供了理论基础。

**Model 鲁棒性**（已在 `ablation_channel_noise.py` 验证）：
- AWGN 5-30dB: 无性能下降 ✅
- 相位偏移: 无性能下降 ✅
- 时间拉伸: 敏感（发现 TSAE 效应）

**脚本**：
- Protocol: `scripts/real_hardware_test.py`
- Model: `scripts/ablation_channel_noise.py`

---

## 🟡 P1 应做（显著加强论文）

### 3. 长程 OOD 分析 ✅
**预计时间**：2 天  
**状态**：✅ 完成

| 项目 | 要求 | 状态 |
|------|------|------|
| 分析内容 | 为什么 OOD-Length 崩溃？ | ✅ |
| 方法 | 输出维度 vs 输入长度解耦分析 | ✅ |
| 产出 | 失败模式分析 + Limitations 声明 | ✅ |

**关键发现**：
- 崩溃主因：**输出维度外推**（训练集 100% 单位数 → OOD 77.5% 双位数）
- `ood_digits`（更长输入但单位数输出）EM = 39.7%，无衰减
- `ood_length`（更长输入 + 双位数输出）EM = 2.7%，崩溃

**产出文件**：
- 分析脚本：`scripts/ood_length_analysis.py`
- 报告：`reports/ood_length_analysis.json`
- Limitations：`docs/paper/limitations.md`

---

### 4. 机制可视化 ✅
**预计时间**：2 天  
**状态**：✅ 完成

| 项目 | 要求 | 状态 |
|------|------|------|
| SSM 状态 | 可视化隐状态轨迹（PCA） | ✅ |
| Attention | 热力图（符号对齐模式） | ✅ |
| 产出 | 2 张图用于 Analysis 部分 | ✅ |

**产出文件**：
- 脚本：`scripts/visualize_mechanisms.py`
- Attention 热力图：`reports/mechanism_viz/attention_heatmap.png`
- 隐状态轨迹：`reports/mechanism_viz/hidden_trajectory.png`

---

## 🟢 P2 可选（锦上添花）

### 5. 更多架构消融
- [ ] 纯 SSM vs 纯 Attention vs Hybrid
- [ ] 层数敏感性（6/10/14 层）

### 6. 更多负对照
- [ ] 噪声输入 baseline（模型输入纯噪声）
- [ ] 随机标签 baseline（打乱 input-target 对应）

### 7. 跨域机制分析
- [ ] CKA 分析（Transfer vs Scratch 的表示相似度）
- [ ] 迁移层选择性分析（哪些层最重要）

---

## 📅 时间线建议

```
Week 1: P0 实验（wav2vec2 公平对比 + 硬件验证）
Week 2: P1 实验（OOD 分析 + 可视化）
Week 3: 论文完稿 + 内部审阅
Week 4: 外部审阅 + 提交
```

---

## 📝 当前论文材料状态

| 部分 | 状态 | 备注 |
|------|------|------|
| Abstract | ✅ 初稿 | 3 版本可选 |
| Introduction | ✅ 扩展版 | 待压缩 |
| Related Work | ✅ 扩展版 | 31 引用 |
| Method | ✅ 扩展版 | 完整 |
| Experiments | ⚠️ 待补 | 需要 P0 实验数据 |
| Analysis | ⚠️ 待补 | 需要可视化 |
| Limitations | ⚠️ 待补 | 需要 OOD 分析 |
| Conclusion | ✅ 初稿 | 简短 |

---

*最后更新：2026-01-01*

