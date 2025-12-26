from __future__ import annotations

from pathlib import Path

from jericho.data.make_manifest import DEFAULT_SPLIT_SIZES, build_manifest
from jericho.data.manifest import read_manifest, write_manifest


def test_manifest_generation_and_constraints(tmp_path):
    split_sizes = {
        "train": 20,
        "val": 10,
        "iid_test": 10,
        "ood_length": 5,
        "ood_symbol": 5,
    }
    entries = build_manifest(seed=123, split_sizes=split_sizes)
    assert len(entries) == sum(split_sizes.values())

    seen_sequences = set()
    train_sequences = []
    ood_symbol_sequences = []

    for entry in entries:
        seq_tuple = tuple(entry.symbols)
        assert seq_tuple not in seen_sequences
        seen_sequences.add(seq_tuple)

        if entry.split == "train":
            train_sequences.append(entry)
        if entry.split == "ood_symbol":
            ood_symbol_sequences.append(entry)

    assert all("F" not in s.symbols for s in train_sequences)
    assert all("F" in s.symbols for s in ood_symbol_sequences)
    assert all(1 <= e.length <= 8 for e in entries if e.split in {"train", "val", "iid_test", "ood_symbol"})
    assert all(9 <= e.length <= 12 for e in entries if e.split == "ood_length")

    manifest_path = tmp_path / "task1.jsonl"
    write_manifest(entries, manifest_path)
    reloaded = read_manifest(manifest_path)
    assert len(reloaded) == len(entries)

