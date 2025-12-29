from __future__ import annotations

import torch

from jericho.data.make_task3_manifest import build_task3_manifest
from jericho.pipelines.task3_mod_audio import Task3TrainingConfig, mini_jmamba_task3_pipeline


def test_task3_metrics_fields_exist_and_in_range():
    split_sizes = {"train": 4, "val": 0, "iid_test": 2, "ood_digits": 0}
    entries = build_task3_manifest(seed=1234, split_sizes=split_sizes, preset="tiny", balance_remainder=True)
    train_entries = [e for e in entries if e.split == "train"]
    eval_entries = [e for e in entries if e.split == "iid_test"]

    _, metrics, _ = mini_jmamba_task3_pipeline(
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
            pretrain_mirror_epochs=1,
            thinking_gap_s=0.5,
            render_mode="none",
            mod_expr_ctc_weight_start=0.5,
            mod_expr_ctc_weight=0.2,
        ),
        mod_lr_factor=0.5,
    )

    for key in [
        "expression_ctc_em_eval_pre",
        "expression_ctc_em_eval_post",
        "expression_token_acc_eval_pre",
        "expression_token_acc_eval_post",
        "expression_ctc_em_eval_mid",
        "expression_token_acc_eval_mid",
        "hybrid_em_overall_pre",
        "hybrid_em_overall_post",
        "hybrid_em_cond_pre",
        "hybrid_em_cond_post",
        "hybrid_parse_ok_rate_pre",
        "hybrid_parse_ok_rate_post",
        "hybrid_em_overall_mid",
        "hybrid_em_cond_mid",
        "hybrid_parse_ok_rate_mid",
        "answer_blank_rate_pre",
        "answer_blank_rate_post",
        "answer_ctc_empty_rate_pre",
        "answer_ctc_empty_rate_post",
        "answer_blank_rate_mid",
        "answer_ctc_empty_rate_mid",
        "remainder_acc_eval_pre",
        "remainder_acc_eval_post",
        "remainder_acc_eval_mid",
        "remainder_acc_eval_mid2",
        "remainder_guidance_blank_floor",
        "answer_digit_mass_floor",
        "answer_blank_margin",
        "answer_blank_margin_weight",
        "remainder_head",
        "remainder_gru_hidden",
        "pretrain_remainder_epochs",
        "pretrain_remainder_lr",
        "pretrain_remainder_freeze_backbone",
        "remainder_ce_loss_mid2",
    ]:
        assert key in metrics
        val = metrics[key]
        if key.endswith("_floor") or key.startswith("answer_blank_margin"):
            assert val >= 0
        elif key == "remainder_head":
            assert val in {"gru", "pooled", "gru_frame", "gru_token", "attn_hidden"}
        elif key == "remainder_gru_hidden":
            assert val > 0
        elif key in {"pretrain_remainder_epochs"}:
            assert val >= 0
        elif key in {"pretrain_remainder_lr"}:
            assert val >= 0
        elif key in {"pretrain_remainder_freeze_backbone"}:
            assert isinstance(val, bool)
        elif key == "remainder_ce_loss_mid2":
            if val is not None:
                assert val >= 0
        else:
            if val is not None:
                assert 0.0 <= val <= 1.0

