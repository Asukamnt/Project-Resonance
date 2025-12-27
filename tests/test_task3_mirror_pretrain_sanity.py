from __future__ import annotations

import torch

from jericho.data.make_task3_manifest import build_task3_manifest
from jericho.pipelines.task3_mod_audio import Task3TrainingConfig, mini_jmamba_task3_pipeline


def test_task3_mirror_pretrain_sanity_tiny():
    split_sizes = {"train": 8, "val": 0, "iid_test": 0, "ood_digits": 0}
    entries = build_task3_manifest(seed=2025, split_sizes=split_sizes, preset="tiny", balance_remainder=True)
    train_entries = [e for e in entries if e.split == "train"]

    _preds, metrics = mini_jmamba_task3_pipeline(
        train_entries,
        train_entries,
        seed=7,
        epochs=1,
        batch_size=4,
        lr=3e-3,
        device=torch.device("cpu"),
        config=Task3TrainingConfig(
            frame_size=160,
            hop_size=160,
            d_model=64,
            pretrain_mirror_epochs=80,
            pretrain_mirror_ctc_weight=2.0,
            pretrain_mirror_frame_ce_weight=3.0,
            pretrain_mirror_blank_penalty_weight=2.0,
            pretrain_mirror_audio_weight=0.0,
            symbol_warmup_epochs=1,
            frame_recon_weight=0.0,
            render_mode="none",
        ),
    )

    blank_rate = metrics["mirror_expression_blank_rate"]
    token_acc = metrics["mirror_expression_token_accuracy"]
    assert blank_rate is not None
    assert token_acc is not None
    assert blank_rate < 0.9
    assert token_acc > 0.5

