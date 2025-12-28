# OOD 评估汇总报告

> 生成时间: 2025-12-29T02:47:12.020787

## 口径说明

**本报告评估的是 Oracle/Protocol 闭环正确性，而非训练模型能力。**

| 指标类型 | 含义 | 本报告 |
|----------|------|--------|
| **Oracle EM** | 系统闭环验证：编码→解码一致性 | 验证 |
| **Model EM** | 训练模型能力：模型预测准确率 | 未包含 |

Oracle EM=1.0 证明：编码正确、解码正确、评测协议无漏洞。

---

## 评估配置

- Stage: final (Oracle/Protocol)
- Tasks: mirror, bracket, mod
- Splits: iid_test, ood_length

## 结果矩阵

| Task | Split | EM | Token Acc | Edit Dist | Total |
|------|-------|-----|-----------|-----------|-------|
| mirror | iid_test | 1.0000 ✅ | 1.0000 | 0.00 | 50 |
| mirror | ood_length | 1.0000 ✅ | 1.0000 | 0.00 | 50 |
| bracket | iid_test | 1.0000 ✅ | 1.0000 | 0.00 | 50 |
| bracket | ood_length | 1.0000 ✅ | 1.0000 | 0.00 | 50 |
| mod | iid_test | 1.0000 ✅ | 1.0000 | 0.00 | 50 |
| mod | ood_length | 1.0000 ✅ | 1.0000 | 0.00 | 50 |

## 任务汇总

| Task | IID EM | OOD Min EM | Avg EM |
|------|--------|------------|--------|
| mirror | 1.0000 | 1.0000 | 1.0000 |
| bracket | 1.0000 | 1.0000 | 1.0000 |
| mod | 1.0000 | 1.0000 | 1.0000 |

## 总体结论

✅ **评估通过**: 平均 EM >= 95%