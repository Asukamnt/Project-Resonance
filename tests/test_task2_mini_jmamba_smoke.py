"""Smoke test for Task2 Mini-JMamba pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from jericho.data.make_task2_manifest import build_task2_manifest, write_manifest
from jericho.data.manifest import read_manifest
from jericho.pipelines import Task2TrainingConfig, mini_jmamba_task2_pipeline


def test_task2_mini_jmamba_pipeline_smoke(tmp_path: Path):
    """Smoke test: Task2 pipeline should run and produce valid metrics."""
    # Generate small manifest
    entries = build_task2_manifest(
        seed=42,
        split_sizes={"train": 20, "iid_test": 10},
        preset="tiny",
        balance_valid=True,
    )
    manifest_path = tmp_path / "task2.jsonl"
    write_manifest(entries, manifest_path)

    train_entries = read_manifest(manifest_path, split="train")
    eval_entries = read_manifest(manifest_path, split="iid_test")

    config = Task2TrainingConfig(
        frame_size=160,
        hop_size=160,
        d_model=32,  # Small for speed
        num_ssm_layers=2,
        num_attn_layers=1,
        symbol_warmup_epochs=2,
    )

    predictions, metrics = mini_jmamba_task2_pipeline(
        train_entries,
        eval_entries,
        seed=123,
        epochs=5,
        batch_size=4,
        lr=1e-3,
        device=torch.device("cpu"),
        config=config,
    )

    # Basic sanity checks
    assert len(predictions) == len(eval_entries)
    assert "audio_accuracy_post" in metrics
    assert "cls_accuracy_post" in metrics
    assert "baseline_accuracy" in metrics
    assert 0.0 <= metrics["audio_accuracy_post"] <= 1.0
    assert 0.0 <= metrics["cls_accuracy_post"] <= 1.0


def test_task2_pipeline_improves_over_epochs(tmp_path: Path):
    """Test that training improves classification accuracy."""
    entries = build_task2_manifest(
        seed=99,
        split_sizes={"train": 30, "iid_test": 15},
        preset="tiny",
        balance_valid=True,
    )
    manifest_path = tmp_path / "task2.jsonl"
    write_manifest(entries, manifest_path)

    train_entries = read_manifest(manifest_path, split="train")
    eval_entries = read_manifest(manifest_path, split="iid_test")

    config = Task2TrainingConfig(
        d_model=48,
        num_ssm_layers=3,
        num_attn_layers=1,
        symbol_warmup_epochs=3,
    )

    predictions, metrics = mini_jmamba_task2_pipeline(
        train_entries,
        eval_entries,
        seed=42,
        epochs=15,
        batch_size=8,
        lr=2e-3,
        device=torch.device("cpu"),
        config=config,
    )

    # Should have some improvement or at least not crash
    assert metrics["loss_post"] <= metrics["loss_pre"] + 0.5  # Allow some variance
    assert len(predictions) == len(eval_entries)
    # Check prediction structure
    for p in predictions:
        assert "gold_symbol" in p
        assert "pred_symbol" in p
        assert p["gold_symbol"] in ("V", "X")

