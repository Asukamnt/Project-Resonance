from __future__ import annotations

from pathlib import Path

import torch

from jericho.data.make_task3_manifest import build_task3_manifest, write_manifest
from jericho.data.manifest import read_manifest
from jericho.pipelines import Task3TrainingConfig, mini_jmamba_task3_pipeline


def test_task3_mini_jmamba_pipeline_improves_metrics(tmp_path: Path):
    split_sizes = {"train": 12, "val": 0, "iid_test": 6, "ood_digits": 0}
    entries = build_task3_manifest(
        seed=99,
        split_sizes=split_sizes,
        preset="easy",
        balance_remainder=True,
    )
    manifest_path = tmp_path / "task3.jsonl"
    write_manifest(entries, manifest_path)

    train_entries = read_manifest(manifest_path, split="train")
    eval_entries = read_manifest(manifest_path, split="iid_test")

    predictions, metrics = mini_jmamba_task3_pipeline(
        train_entries,
        eval_entries,
        seed=7,
        epochs=12,
        batch_size=4,
        lr=3e-3,
        device=torch.device("cpu"),
        config=Task3TrainingConfig(frame_size=160, hop_size=160, ctc_weight=0.3, d_model=64),
    )

    assert metrics["loss_post"] <= metrics["loss_pre"]
    assert metrics["baseline_em"] >= 0.0
    assert metrics["ctc_em_post"] >= 0.0
    assert 0.0 <= metrics["ctc_blank_rate_post"] <= 1.0
    assert "em_baseline_margin" in metrics
    assert "pass_baseline_margin" in metrics
    assert "pass_ctc_threshold" in metrics
    assert len(predictions) == len(eval_entries)

