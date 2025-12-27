from __future__ import annotations

from pathlib import Path

from jericho.data.make_manifest import DEFAULT_SPLIT_SIZES, build_manifest
from jericho.data.manifest import read_manifest, write_manifest


def test_manifest_defaults_and_reproducibility(tmp_path: Path) -> None:
    entries_first = build_manifest(seed=2024, split_sizes=None)
    entries_second = build_manifest(seed=2024, split_sizes=None)

    assert len(entries_first) == sum(DEFAULT_SPLIT_SIZES.values())
    assert [e.symbols for e in entries_first] == [e.symbols for e in entries_second]

    counts: dict[str, int] = {split: 0 for split in DEFAULT_SPLIT_SIZES}
    seen_sequences: set[tuple[str, ...]] = set()
    ood_symbol_entries = []

    for entry in entries_first:
        counts[entry.split] += 1
        seq_tuple = tuple(entry.symbols)
        assert seq_tuple not in seen_sequences
        seen_sequences.add(seq_tuple)

        if entry.split != "ood_symbol":
            assert "F" not in entry.symbols
        else:
            ood_symbol_entries.append(entry)
            assert "F" in entry.symbols

        if entry.split in {"train", "val", "iid_test", "ood_symbol"}:
            assert 1 <= entry.length <= 8
        if entry.split == "ood_length":
            assert 9 <= entry.length <= 12

    assert counts == DEFAULT_SPLIT_SIZES
    assert ood_symbol_entries, "OOD symbol split must contain entries with 'F'."

    manifest_path = tmp_path / "task1.jsonl"
    manifest_copy_path = tmp_path / "task1_dup.jsonl"
    write_manifest(entries_first, manifest_path)
    write_manifest(entries_second, manifest_copy_path)

    assert manifest_path.read_text(encoding="utf-8") == manifest_copy_path.read_text(
        encoding="utf-8"
    )

    reloaded = read_manifest(manifest_path)
    assert len(reloaded) == len(entries_first)

