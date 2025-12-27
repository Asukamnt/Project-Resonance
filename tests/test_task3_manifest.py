from __future__ import annotations

from pathlib import Path

from jericho.data.make_task3_manifest import build_task3_manifest, write_manifest
from jericho.task3.utils import parse_mod_expression


def test_task3_manifest_generation(tmp_path: Path):
    split_sizes = {"train": 10, "val": 4, "iid_test": 4, "ood_digits": 4}
    entries = build_task3_manifest(seed=42, split_sizes=split_sizes)
    assert len(entries) == sum(split_sizes.values())

    for entry in entries:
        assert "%" in entry.symbols
        _, b = parse_mod_expression(entry.symbols)
        assert b != 0

    manifest_path = tmp_path / "task3.jsonl"
    write_manifest(entries, manifest_path)
    entries_again = build_task3_manifest(seed=42, split_sizes=split_sizes)
    assert [e.symbols for e in entries] == [e.symbols for e in entries_again]

