# S26 复现资产

本目录包含项目复现所需的资产。

## 目录结构

```
artifacts/
├── README.md           # 本文件
├── checkpoints/        # 模型检查点（.pt）
└── audio_examples/     # 音频示例（.wav + index.json）
```

## 检查点

训练完成后的模型检查点存放在 `checkpoints/` 目录。

### 命名规范

```
{task}_{stage}_{seed}_epoch{N}.pt
```

示例:
- `mirror_mvp_123_epoch30.pt`
- `bracket_final_123_epoch50.pt`
- `mod_final_123_epoch100.pt`

当前仓库内的 demo 示例（用于外部快速复现，不代表"最优模型"）：
- `mirror_demo_seed42_epoch15.pt` — **Model EM = 1.0** (IID & OOD-length)

> **注意**：
> - **Mirror** checkpoint 可直接用 `evaluate_model.py` 验证 Model EM
> - **Mod/Bracket** 任务需要完整 pipeline（含答案窗口定位）才能正确评估
> - 如需完整 Task3 训练和评估，请使用 `train.py --config configs/task3_mod_stable.yaml`

### 加载方式

```python
import torch
checkpoint = torch.load("artifacts/checkpoints/mirror_mvp_123_epoch30.pt")
model.load_state_dict(checkpoint["model_state_dict"])
```

## 音频示例

代表性音频示例存放在 `audio_examples/` 目录。

### 示例类型

1. **成功案例**: 模型正确生成的音频
2. **失败案例**: 有助于诊断的错误示例
3. **OOD 示例**: 分布外泛化示例

### 索引文件

- `audio_examples/index.json`：记录每条样例的 `input_symbols/target_symbols/predicted_symbols/correct` 与对应 wav 文件名。

### 命名规范

```
{task}_{split}_{example_id}_{input|output|target}.wav
```

示例:
- `mirror_iid_test_000001_input.wav`
- `bracket_ood_length_000042_output.wav`
- `mod_ood_compose_000123_target.wav`

## 复现说明

1. 确保安装所有依赖: `pip install -r requirements.txt`
2. 使用固定种子: `--seed 123`
3. 参考 `docs/repro_seeds.json` 获取完整种子配置

## 验证

```bash
# 验证资产完整性
python -c "from pathlib import Path; assert Path('artifacts/checkpoints').exists()"
python -c "from pathlib import Path; assert Path('artifacts/audio_examples').exists()"
```

