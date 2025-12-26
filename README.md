# Project Resonance — Jericho

## 状态
- ✅ Week1 / Stage A：Task1 (Mirror) 符号→音频编码器与 scorer 解码器已实现。
- ✅ `tests/test_task1_roundtrip.py` 覆盖随机序列往返闭环，pytest 通过即视为 100% roundtrip。
- ⏳ 后续阶段：Mini-JMamba 模型集成、Task2/Task3、训练/评估流水线。

## 目录结构
- `src/jericho/symbols.py`：符号表、频率映射与正弦音频合成。
- `src/jericho/scorer.py`：基于 FFT 的频率识别与 exact match 评分。
- `src/jericho/data/make_manifest.py`：Task1 manifest 生成器。
- `tests/test_task1_roundtrip.py`：Task1 往返闭环单测。
- `docs/spec_from_pdf.md`：PDF 规约提炼。

## 快速开始（Windows PowerShell）
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest -q
```

运行 `pytest -q` 将在本地生成随机符号序列，验证 “符号→音频→符号” 闭环的 100% 精确度。关闭虚拟环境，可执行 `deactivate`。

## 数据 manifest 与基线命令
- 生成可复现 Task1 manifest（JSONL，每行含 split/symbols/length/difficulty_tag/example_id 等字段）：
  ```powershell
  python -m jericho.data.make_manifest --out manifests/task1.jsonl --seed 42
  ```
- 对指定拆分进行批量 sanity check（Exact Match 应为 1.0）：
  ```powershell
  python .\evaluate.py --manifest manifests/task1.jsonl --split iid_test
  ```
- 运行恒等基线（将符号序列编码→解码并写入 `artifacts/baselines`）：
  ```powershell
  python .\train.py --manifest manifests/task1.jsonl --splits train val --out artifacts/baselines/trivial.jsonl
  ```

生成的 manifest 默认遵循 Stage A 规范：`train/val/iid_test` 仅使用符号 A–E 长度 1–8，`ood_length` 使用长度 9–12，`ood_symbol` 引入新符号 F（训练集内不会出现）。
