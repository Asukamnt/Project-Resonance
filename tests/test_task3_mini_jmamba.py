from __future__ import annotations

from pathlib import Path

import torch

from jericho.data.make_task3_manifest import build_task3_manifest, write_manifest
from jericho.pipelines import Task3TrainingConfig, mini_jmamba_task3_pipeline


def test_mini_jmamba_task3_smoke(tmp_path: Path):
    manifest_entries = build_task3_manifest(
        seed=11,
        split_sizes={"train": 8, "val": 0, "iid_test": 4, "ood_digits": 0},
        preset="easy",
        balance_remainder=True,
    )
    manifest_path = tmp_path / "task3.jsonl"
    write_manifest(manifest_entries, manifest_path)

    train_entries = [e for e in manifest_entries if e.split == "train"]
    eval_entries = [e for e in manifest_entries if e.split == "iid_test"]

    predictions, metrics, _ = mini_jmamba_task3_pipeline(
        train_entries,
        eval_entries,
        seed=0,
        epochs=6,
        batch_size=4,
        lr=5e-4,
        device=torch.device("cpu"),
        config=Task3TrainingConfig(frame_size=160, hop_size=160, ctc_weight=0.3),
    )

    assert metrics["loss_post"] <= metrics["loss_pre"]
    assert metrics["ctc_em_post"] >= 0.0
    assert "baseline_em" in metrics
    assert "em_baseline_margin" in metrics
    assert isinstance(metrics.get("pass_ctc_threshold"), bool)
    assert len(predictions) == len(eval_entries)

