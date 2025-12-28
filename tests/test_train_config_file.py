from __future__ import annotations

from pathlib import Path

import pytest

import train


def test_train_config_loads_and_cli_overrides(tmp_path: Path) -> None:
    cfg = tmp_path / "task3_mod_stable.yaml"
    cfg.write_text(
        "\n".join(
            [
                "task: mod",
                "model: mini_jmamba",
                "epochs: 5",
                "thinking_gap_s: 0.7",
                "pretrain_remainder_freeze_backbone: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    args = train.parse_args(
        [
            "--config",
            str(cfg),
            "--manifest",
            "manifests/task3_tiny.jsonl",
            "--thinking-gap-s",
            "0.5",
            "--no-pretrain-remainder-freeze-backbone",
        ]
    )

    assert args.task == "mod"
    assert args.model == "mini_jmamba"
    assert args.epochs == 5
    assert args.thinking_gap_s == pytest.approx(0.5)
    assert args.pretrain_remainder_freeze_backbone is False


def test_train_config_unknown_key_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("unknown_key: 123\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        train.parse_args(["--config", str(cfg)])

    assert "Unknown config keys" in str(excinfo.value)


def test_task3_mod_stable_config_file_loads() -> None:
    """Ensure the repo-shipped stable config stays in sync with train.py argparse."""
    cfg = Path("configs/task3_mod_stable.yaml")
    assert cfg.exists(), "Expected configs/task3_mod_stable.yaml to exist"

    args = train.parse_args(["--config", str(cfg), "--manifest", "manifests/task3_tiny.jsonl"])
    assert args.task == "mod"
    assert args.model == "mini_jmamba"
    assert args.thinking_gap_s == pytest.approx(0.5)
    assert args.pretrain_mirror_epochs == 15
    assert args.remainder_head == "attn_hidden"
    assert args.epochs == 50


def test_task2_bracket_stable_config_file_loads() -> None:
    """Ensure the repo-shipped Task2 config stays in sync with train.py argparse."""
    cfg = Path("configs/task2_bracket_stable.yaml")
    assert cfg.exists(), "Expected configs/task2_bracket_stable.yaml to exist"

    args = train.parse_args(
        [
            "--config",
            str(cfg),
            "--task",
            "bracket",
            "--model",
            "mini_jmamba",
            "--manifest",
            "manifests/task2_tiny.jsonl",
        ]
    )
    assert args.task == "bracket"
    assert args.model == "mini_jmamba"
    assert args.epochs == 50
    assert args.lr == pytest.approx(0.001)
    assert args.batch_size == 16
    assert args.seed == 123
    assert args.task2_binary_ce_weight == pytest.approx(1.0)
    assert args.task2_symbol_guidance_weight == pytest.approx(1.0)
    assert args.task2_thinking_gap_s == pytest.approx(0.3)
    assert args.task2_answer_window_only is True
    assert args.task2_audio_ramp_epochs == 10
    assert args.symbol_warmup_epochs == 5


