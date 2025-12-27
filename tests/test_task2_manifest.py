"""Tests for Task2 manifest generation."""

from __future__ import annotations

from pathlib import Path

from jericho.data.make_task2_manifest import build_task2_manifest, PRESETS
from jericho.data.manifest import read_manifest, write_manifest
from jericho.task2 import is_balanced, VALID_SYMBOL, INVALID_SYMBOL


class TestBuildTask2Manifest:
    """Tests for build_task2_manifest function."""

    def test_default_split_sizes(self):
        entries = build_task2_manifest(seed=42)
        splits = {e.split for e in entries}
        assert "train" in splits
        assert "iid_test" in splits

    def test_balance_valid(self):
        entries = build_task2_manifest(
            seed=42,
            split_sizes={"train": 100},
            balance_valid=True,
        )
        valid_count = sum(1 for e in entries if is_balanced(e.symbols))
        invalid_count = len(entries) - valid_count
        # Should be approximately 50/50
        assert abs(valid_count - invalid_count) <= 2

    def test_preset_tiny(self):
        entries = build_task2_manifest(
            seed=42,
            split_sizes={"train": 20},
            preset="tiny",
        )
        assert len(entries) == 20
        for e in entries:
            assert len(e.symbols) <= 10  # tiny preset max

    def test_preset_easy(self):
        entries = build_task2_manifest(
            seed=42,
            split_sizes={"train": 20},
            preset="easy",
        )
        assert len(entries) == 20

    def test_reproducible(self):
        e1 = build_task2_manifest(seed=123, split_sizes={"train": 10})
        e2 = build_task2_manifest(seed=123, split_sizes={"train": 10})
        assert [e.symbols for e in e1] == [e.symbols for e in e2]

    def test_ood_length_split(self):
        entries = build_task2_manifest(
            seed=42,
            split_sizes={"ood_length": 10},
            preset="tiny",
        )
        config = PRESETS["tiny"]
        for e in entries:
            assert e.split == "ood_length"
            assert e.difficulty_tag == "ood"
            assert len(e.symbols) >= config.ood_length_range[0]


class TestTask2ManifestRoundtrip:
    """Test writing and reading Task2 manifest."""

    def test_roundtrip(self, tmp_path: Path):
        manifest_path = tmp_path / "task2.jsonl"
        entries = build_task2_manifest(
            seed=42,
            split_sizes={"train": 10, "iid_test": 5},
        )
        write_manifest(entries, manifest_path)

        loaded_train = read_manifest(manifest_path, split="train")
        loaded_test = read_manifest(manifest_path, split="iid_test")

        assert len(loaded_train) == 10
        assert len(loaded_test) == 5

