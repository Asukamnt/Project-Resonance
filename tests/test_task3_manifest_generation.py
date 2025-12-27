from __future__ import annotations

from collections import Counter
from pathlib import Path

from jericho.data.make_task3_manifest import build_task3_manifest
from jericho.data.manifest import read_manifest, write_manifest
from jericho.task3 import MOD_OPERATOR, parse_mod_expression, target_symbols_for_task3


def _check_no_divisor_zero(entries):
    for entry in entries:
        assert MOD_OPERATOR in entry.symbols
        op_index = entry.symbols.index(MOD_OPERATOR)
        right = entry.symbols[op_index + 1 :]
        assert right
        assert "".join(right) != "0"


def test_task3_manifest_generation_and_reproducibility(tmp_path: Path):
    split_sizes = {"train": 10, "val": 5, "iid_test": 5, "ood_digits": 5}
    entries_a = build_task3_manifest(seed=123, split_sizes=split_sizes)
    entries_b = build_task3_manifest(seed=123, split_sizes=split_sizes)
    assert len(entries_a) == sum(split_sizes.values())
    assert [e.symbols for e in entries_a] == [e.symbols for e in entries_b]

    counts = {split: 0 for split in split_sizes}
    seen_per_split: dict[str, set[tuple[str, ...]]] = {split: set() for split in split_sizes}
    for entry in entries_a:
        counts[entry.split] += 1
        token_tuple = tuple(entry.symbols)
        assert token_tuple not in seen_per_split[entry.split]
        seen_per_split[entry.split].add(token_tuple)
        assert entry.length == len(entry.symbols)
        assert MOD_OPERATOR in entry.symbols
    assert counts == split_sizes

    _check_no_divisor_zero(entries_a)

    manifest_path = tmp_path / "task3.jsonl"
    write_manifest(entries_a, manifest_path)
    reloaded = read_manifest(manifest_path)
    assert [e.symbols for e in reloaded] == [e.symbols for e in entries_a]


def test_task3_manifest_easy_preset_single_digit_remainders():
    split_sizes = {"train": 20, "val": 0, "iid_test": 0, "ood_digits": 0}
    entries = build_task3_manifest(seed=77, split_sizes=split_sizes, preset="easy")
    train_entries = [entry for entry in entries if entry.split == "train"]
    assert len(train_entries) == split_sizes["train"]
    for entry in train_entries:
        digits = target_symbols_for_task3(entry.symbols)
        assert len(digits) == 1
        assert digits[0] in "0123456789"
        assert int("".join(digits)) < 10


def test_task3_manifest_balance_remainder_spreads_distribution():
    split_sizes = {"train": 100, "val": 0, "iid_test": 0, "ood_digits": 0}
    entries = build_task3_manifest(
        seed=10,
        split_sizes=split_sizes,
        preset="easy",
        balance_remainder=True,
    )
    train_entries = [entry for entry in entries if entry.split == "train"]
    remainders = ["".join(target_symbols_for_task3(entry.symbols)) for entry in train_entries]
    counts = Counter(remainders)
    assert len(counts) >= 8
    max_bucket = max(counts.values())
    min_bucket = min(counts.values())
    assert max_bucket - min_bucket <= 6


def test_task3_manifest_tiny_preset_has_small_operands():
    split_sizes = {"train": 20, "val": 0, "iid_test": 0, "ood_digits": 0}
    entries = build_task3_manifest(seed=55, split_sizes=split_sizes, preset="tiny")
    train_entries = [entry for entry in entries if entry.split == "train"]
    assert len(train_entries) == split_sizes["train"]
    for entry in train_entries:
        dividend, divisor = parse_mod_expression(entry.symbols)
        assert 0 <= dividend <= 99
        assert 2 <= divisor <= 9
        remainder_digits = target_symbols_for_task3(entry.symbols)
        assert len(remainder_digits) == 1
