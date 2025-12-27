from __future__ import annotations

from pathlib import Path

import torch

from jericho.data.make_manifest import build_manifest
from jericho.data.manifest import read_manifest, write_manifest
from jericho.models.mini_jmamba import (
    AttentionBlock,
    MiniJMamba,
    MiniJMambaConfig,
    SSMLikeBlock,
)
from jericho.pipelines import MiniJMambaTrainingConfig, mini_jmamba_pipeline


def _random_frames(batch: int, seq_len: int, frame_size: int):
    torch.manual_seed(0)
    features = torch.randn(batch, seq_len, frame_size)
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    return features, mask


def test_mini_jmamba_forward_shapes():
    config = MiniJMambaConfig(
        frame_size=16,
        hop_size=16,
        symbol_vocab_size=8,
        d_model=32,
        num_ssm_layers=2,
        num_attn_layers=1,
        num_heads=4,
        max_frames=10,
        dropout=0.0,
        attn_dropout=0.0,
    )
    model = MiniJMamba(config)
    features, mask = _random_frames(batch=3, seq_len=5, frame_size=config.frame_size)
    frame_outputs, symbol_logits = model(features, mask)
    assert frame_outputs.shape == (3, 5, config.frame_size)
    assert symbol_logits.shape == (3, 5, config.symbol_vocab_size)


def test_mini_jmamba_structure_counts():
    config = MiniJMambaConfig(frame_size=8, hop_size=8, symbol_vocab_size=6)
    model = MiniJMamba(config)
    ssm_layers = sum(isinstance(layer, SSMLikeBlock) for layer in model.layers)
    attn_layers = sum(isinstance(layer, AttentionBlock) for layer in model.layers)
    assert ssm_layers == config.num_ssm_layers
    assert attn_layers == config.num_attn_layers
    assert len(model.layers) == config.num_ssm_layers + config.num_attn_layers


def test_mini_jmamba_training_step_decreases_loss():
    config = MiniJMambaConfig(
        frame_size=12,
        hop_size=12,
        symbol_vocab_size=6,
        d_model=32,
        num_ssm_layers=2,
        num_attn_layers=1,
        num_heads=4,
        max_frames=8,
        dropout=0.0,
        attn_dropout=0.0,
    )
    model = MiniJMamba(config)
    features, mask = _random_frames(batch=6, seq_len=6, frame_size=config.frame_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    frame_outputs, symbol_logits = model(features, mask)
    loss_before = ((frame_outputs - features) ** 2).mean()

    for _ in range(3):
        frame_outputs, symbol_logits = model(features, mask)
        loss = ((frame_outputs - features) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    frame_outputs, symbol_logits = model(features, mask)
    loss_after = ((frame_outputs - features) ** 2).mean()

    assert torch.isfinite(loss_before)
    assert torch.isfinite(loss_after)
    assert loss_after.item() <= loss_before.item()


def test_mini_jmamba_pipeline_improves_metrics(tmp_path: Path):
    manifest_entries = build_manifest(
        seed=7,
        split_sizes={"train": 8, "val": 0, "iid_test": 4, "ood_length": 0, "ood_symbol": 0},
    )
    manifest_path = tmp_path / "task1.jsonl"
    write_manifest(manifest_entries, manifest_path)

    train_entries = read_manifest(manifest_path, split="train")
    eval_entries = read_manifest(manifest_path, split="iid_test")

    predictions, metrics = mini_jmamba_pipeline(
        train_entries,
        eval_entries,
        seed=0,
        epochs=10,
        batch_size=4,
        lr=1e-3,
        device=torch.device("cpu"),
        config=MiniJMambaTrainingConfig(frame_size=160, hop_size=160, ctc_weight=0.3),
    )

    assert metrics["loss_post"] <= metrics["loss_pre"]
    assert metrics["em_post"] >= metrics["em_pre"]
    assert len(predictions) == len(eval_entries)
