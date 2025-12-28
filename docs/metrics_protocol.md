# 评测口径协议

> 最后更新: 2025-12-29

本文档定义项目的两类核心评测指标及其区别，避免口径漂移。

## 两类 EM 指标

| 指标 | 全称 | 定义 | 验证脚本 |
|------|------|------|----------|
| **Oracle EM** | Oracle Exact Match | 系统闭环正确性：目标波形→FFT 解码→符号 == 原始符号 | `evaluate.py` |
| **Model EM** | Model Exact Match | 训练模型能力：输入波形→模型预测→FFT 解码→符号 == 目标符号 | `train.py --split` |

## Oracle EM（系统闭环验证）

### 定义

```
Oracle EM = (目标波形解码后的符号 == 原始目标符号) 的比例
```

### 流程

```
输入符号 → 目标计算 → 目标符号 → 波形编码 → 目标波形 → FFT 解码 → 解码符号
                                    ↓
                              Oracle EM = (解码符号 == 目标符号)?
```

### 意义

- **Oracle EM = 1.0** 表示：
  - ✅ 符号→波形编码正确
  - ✅ FFT 解码器正确
  - ✅ 评测协议无漏洞
  - ✅ 没有评分器偏差

- **Oracle EM < 1.0** 表示：
  - ❌ 系统存在 bug
  - ❌ 编码/解码不一致
  - ❌ 需要修复后才能评估模型

### 验证命令

```bash
python evaluate.py --stage final --tasks mirror bracket mod
```

---

## Model EM（训练模型能力）

### 定义

```
Model EM = (模型输出波形解码后的符号 == 目标符号) 的比例
```

### 流程

```
输入符号 → 波形编码 → 输入波形 → 模型推理 → 输出波形 → FFT 解码 → 预测符号
                                                         ↓
目标符号 ← 目标计算 ← 输入符号                    Model EM = (预测符号 == 目标符号)?
```

### 意义

- **Model EM** 反映模型真正学到的能力
- 需要在 Oracle EM = 1.0 的基础上评估
- IID EM 和 OOD EM 分开报告

### 验证命令

```bash
# 训练后评估
python train.py --task mirror --split iid_test --model mini_jmamba
python train.py --task bracket --split ood_length --model mini_jmamba
```

---

## 报告规范

### 报告中必须明确标注

1. **报告类型**：Oracle/Protocol 还是 Model
2. **评估对象**：闭环验证 还是 模型能力
3. **适用场景**：系统调试 还是 论文结果

### 示例

```markdown
## 评估结果

### Oracle/Protocol 验证 (系统闭环)
| Task | Oracle EM |
|------|-----------|
| mirror | 1.0000 |
| bracket | 1.0000 |
| mod | 1.0000 |

### Model 能力 (训练后)
| Task | Split | Model EM |
|------|-------|----------|
| mirror | iid_test | 0.98 |
| mirror | ood_length | 0.84 |
| bracket | iid_test | 0.96 |
| mod | iid_test | 0.97 |
```

---

## 常见误区

### ❌ 错误理解

"Oracle EM=1.0 说明模型很强"

### ✅ 正确理解

"Oracle EM=1.0 说明评测系统闭环正确，可以信任后续的 Model EM 结果"

---

## 更新记录

| 日期 | 更新内容 |
|------|----------|
| 2025-12-29 | 初始版本，定义 Oracle EM vs Model EM |

