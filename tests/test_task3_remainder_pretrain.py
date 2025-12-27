import torch

from jericho.data.make_task3_manifest import build_task3_manifest
from jericho.pipelines.task3_mod_audio import Task3TrainingConfig, mini_jmamba_task3_pipeline


def test_remainder_pretrain_runs_and_metrics():
    split_sizes = {"train": 4, "val": 0, "iid_test": 2, "ood_digits": 0}
    entries = build_task3_manifest(seed=4321, split_sizes=split_sizes, preset="tiny", balance_remainder=True)
    train_entries = [e for e in entries if e.split == "train"]
    eval_entries = [e for e in entries if e.split == "iid_test"]

    _, metrics = mini_jmamba_task3_pipeline(
        train_entries,
        eval_entries,
        seed=0,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device=torch.device("cpu"),
        config=Task3TrainingConfig(
            frame_size=160,
            hop_size=160,
            d_model=32,
            pretrain_mirror_epochs=0,
            pretrain_remainder_epochs=1,
            pretrain_remainder_lr=1e-3,
            pretrain_remainder_freeze_backbone=True,
            remainder_head="gru_token",
            remainder_gru_hidden=16,
            render_mode="none",
            mod_expr_ctc_weight_start=0.5,
            mod_expr_ctc_weight=0.2,
            remainder_ce_weight=2.0,
        ),
        mod_lr_factor=0.5,
    )

    assert "remainder_acc_eval_mid2" in metrics
    assert "remainder_ce_loss_mid2" in metrics
    assert metrics["pretrain_remainder_epochs"] == 1

