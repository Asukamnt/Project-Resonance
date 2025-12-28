# 评测总表

> 生成时间: 2025-12-29T02:32:41.252246

## 口径说明

| 指标 | 定义 |
|------|------|
| **Oracle EM** | 系统闭环验证：目标波形→FFT解码→符号 == 原始符号 |
| **Model EM** | 训练模型能力：输入→模型→FFT解码→符号 == 目标符号 |

---

## Oracle/Protocol 验证

| Task | Split | Oracle EM | Token Acc | Total |
|------|-------|-----------|-----------|-------|
| mirror | iid_test | 1.0000 ✅ | 1.0000 | 50 |
| mirror | ood_length | 1.0000 ✅ | 1.0000 | 50 |
| bracket | iid_test | 1.0000 ✅ | 1.0000 | 50 |
| bracket | ood_length | 1.0000 ✅ | 1.0000 | 50 |
| mod | iid_test | 1.0000 ✅ | 1.0000 | 50 |
| mod | ood_length | 1.0000 ✅ | 1.0000 | 50 |

---

## Model 能力（训练后）

| Task | Split | Model EM | Token Acc | Total |
|------|-------|----------|-----------|-------|
| mod | iid_test | 0.0000 ❌ | 0.2105 | 50 |
| mod | ood_length | 0.0000 ❌ | 0.0343 | 50 |

---

## 对比汇总

| Task | Split | Oracle EM | Model EM | Gap |
|------|-------|-----------|----------|-----|
| mod | iid_test | 1.0000 | 0.0000 | +1.0000 |
| mod | ood_length | 1.0000 | 0.0000 | +1.0000 |