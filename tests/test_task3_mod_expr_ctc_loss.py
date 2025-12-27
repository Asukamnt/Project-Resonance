from __future__ import annotations

import torch

from jericho.data.make_task3_manifest import build_task3_manifest
from jericho.pipelines.task3_mod_audio import Task3TrainingConfig, mini_jmamba_task3_pipeline


def test_mod_expr_ctc_loss_changes_total_loss():
    split_sizes = {"train": 4, "val": 0, "iid_test": 2, "ood_digits": 0}
    entries = build_task3_manifest(seed=999, split_sizes=split_sizes, preset="tiny", balance_remainder=True)
    train_entries = [e for e in entries if e.split == "train"]
    eval_entries = [e for e in entries if e.split == "iid_test"]

    # Run with zero mod expr ctc weight
    _preds0, metrics0 = mini_jmamba_task3_pipeline(
        train_entries,
        eval_entries,
        seed=1,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device=torch.device("cpu"),
        config=Task3TrainingConfig(
            frame_size=160,
            hop_size=160,
            d_model=32,
            pretrain_mirror_epochs=0,
            mod_expr_ctc_weight_start=0.0,
            mod_expr_ctc_weight=0.0,
            render_mode="none",
        ),
        mod_lr_factor=1.0,
    )

    _preds1, metrics1 = mini_jmamba_task3_pipeline(
        train_entries,
        eval_entries,
        seed=1,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device=torch.device("cpu"),
        config=Task3TrainingConfig(
            frame_size=160,
            hop_size=160,
            d_model=32,
            pretrain_mirror_epochs=0,
            mod_expr_ctc_weight_start=1.0,
            mod_expr_ctc_weight=0.5,
            render_mode="none",
        ),
        mod_lr_factor=1.0,
    )

    # Ensure losses differ when mod expr ctc is enabled
    assert metrics1["loss_post"] != metrics0["loss_post"]

