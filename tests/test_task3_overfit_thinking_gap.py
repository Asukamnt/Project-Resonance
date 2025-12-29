from __future__ import annotations

import torch

from jericho.data.make_task3_manifest import build_task3_manifest, write_manifest
from jericho.data.manifest import read_manifest
from jericho.pipelines import Task3TrainingConfig, mini_jmamba_task3_pipeline


def test_task3_mini_jmamba_overfits_small_set(tmp_path):
    split_sizes = {"train": 8, "val": 0, "iid_test": 4, "ood_digits": 0}
    entries = build_task3_manifest(
        seed=2024,
        split_sizes=split_sizes,
        preset="easy",
        balance_remainder=True,
    )
    manifest_path = tmp_path / "task3.jsonl"
    write_manifest(entries, manifest_path)

    train_entries = read_manifest(manifest_path, split="train")
    eval_entries = read_manifest(manifest_path, split="train")

    _, metrics, _ = mini_jmamba_task3_pipeline(
        train_entries,
        eval_entries,
        seed=7,
        epochs=80,
        batch_size=4,
        lr=3e-3,
        device=torch.device("cpu"),
        config=Task3TrainingConfig(
            frame_size=160,
            hop_size=160,
            d_model=64,
            ctc_weight=0.6,
            ctc_weight_start=1.5,
            ctc_weight_end=0.3,
            frame_ce_weight=2.5,
            blank_penalty_weight=1.5,
            symbol_warmup_epochs=20,
            thinking_gap_s=0.1,
            thinking_gap_align=160,
            answer_window_only=True,
        ),
    )

    assert metrics["ctc_em_post"] > metrics["ctc_em_pre"]
    assert metrics["ctc_blank_rate_post"] <= metrics["ctc_blank_rate_pre"] + 1e-6
    assert metrics["em_post"] >= metrics["baseline_em"]

