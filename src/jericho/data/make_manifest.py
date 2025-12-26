"""Manifest generator for Task1 datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from ..symbols import SYMBOLS
from .manifest import ManifestEntry, write_manifest

IID_SYMBOLS: Sequence[str] = SYMBOLS
OOD_EXTRA_SYMBOL: str = "F"

DEFAULT_SPLIT_SIZES: Dict[str, int] = {
    "train": 500,
    "val": 100,
    "iid_test": 100,
    "ood_length": 100,
    "ood_symbol": 100,
}

IID_LENGTH_RANGE = (1, 8)
OOD_LENGTH_RANGE = (9, 12)


def _sample_sequence(
    rng: np.random.Generator,
    length_range: tuple[int, int],
    alphabet: Sequence[str],
    *,
    must_include: str | None = None,
) -> tuple[str, ...]:
    min_len, max_len = length_range
    length = int(rng.integers(min_len, max_len + 1))
    seq = list(rng.choice(alphabet, size=length, replace=True))
    if must_include is not None and must_include not in seq:
        # Enforce presence by replacing a random position.
        idx = int(rng.integers(0, length))
        seq[idx] = must_include
    return tuple(seq)


def _generate_split(
    split: str,
    count: int,
    rng: np.random.Generator,
    *,
    existing_sequences: set[tuple[str, ...]],
    length_range: tuple[int, int],
    alphabet: Sequence[str],
    difficulty_tag: str,
    seed: int,
    must_include: str | None = None,
) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    attempts = 0
    max_attempts = count * 20
    while len(entries) < count:
        if attempts > max_attempts:
            raise RuntimeError(f"Unable to generate enough unique sequences for {split}")
        attempts += 1
        sequence_seed = int(rng.integers(0, 2**63))
        seq = _sample_sequence(
            rng,
            length_range,
            alphabet,
            must_include=must_include,
        )
        if seq in existing_sequences:
            continue
        existing_sequences.add(seq)
        example_id = f"{split}-{len(entries):06d}"
        entry = ManifestEntry(
            split=split,
            symbols=list(seq),
            length=len(seq),
            difficulty_tag=difficulty_tag,
            example_id=example_id,
            seed=seed,
            sequence_seed=sequence_seed,
        )
        entries.append(entry)
    return entries


def build_manifest(
    *,
    seed: int,
    split_sizes: Dict[str, int] | None = None,
) -> list[ManifestEntry]:
    """Construct manifest entries for all splits."""
    sizes = split_sizes or DEFAULT_SPLIT_SIZES
    rng = np.random.default_rng(seed)
    seen: set[tuple[str, ...]] = set()
    entries: list[ManifestEntry] = []

    entries.extend(
        _generate_split(
            "train",
            sizes.get("train", 0),
            rng,
            existing_sequences=seen,
            length_range=IID_LENGTH_RANGE,
            alphabet=IID_SYMBOLS,
            difficulty_tag="iid",
            seed=seed,
        )
    )
    entries.extend(
        _generate_split(
            "val",
            sizes.get("val", 0),
            rng,
            existing_sequences=seen,
            length_range=IID_LENGTH_RANGE,
            alphabet=IID_SYMBOLS,
            difficulty_tag="iid",
            seed=seed,
        )
    )
    entries.extend(
        _generate_split(
            "iid_test",
            sizes.get("iid_test", 0),
            rng,
            existing_sequences=seen,
            length_range=IID_LENGTH_RANGE,
            alphabet=IID_SYMBOLS,
            difficulty_tag="iid",
            seed=seed,
        )
    )
    entries.extend(
        _generate_split(
            "ood_length",
            sizes.get("ood_length", 0),
            rng,
            existing_sequences=seen,
            length_range=OOD_LENGTH_RANGE,
            alphabet=IID_SYMBOLS,
            difficulty_tag="ood_length",
            seed=seed,
        )
    )
    entries.extend(
        _generate_split(
            "ood_symbol",
            sizes.get("ood_symbol", 0),
            rng,
            existing_sequences=seen,
            length_range=IID_LENGTH_RANGE,
            alphabet=tuple(IID_SYMBOLS) + (OOD_EXTRA_SYMBOL,),
            difficulty_tag="ood_symbol",
            seed=seed,
            must_include=OOD_EXTRA_SYMBOL,
        )
    )

    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task1 manifest JSONL.")
    parser.add_argument("--out", type=Path, default=Path("manifests/task1.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=DEFAULT_SPLIT_SIZES["train"])
    parser.add_argument("--val-size", type=int, default=DEFAULT_SPLIT_SIZES["val"])
    parser.add_argument(
        "--iid-test-size", type=int, default=DEFAULT_SPLIT_SIZES["iid_test"]
    )
    parser.add_argument(
        "--ood-length-size", type=int, default=DEFAULT_SPLIT_SIZES["ood_length"]
    )
    parser.add_argument(
        "--ood-symbol-size", type=int, default=DEFAULT_SPLIT_SIZES["ood_symbol"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "iid_test": args.iid_test_size,
        "ood_length": args.ood_length_size,
        "ood_symbol": args.ood_symbol_size,
    }
    entries = build_manifest(seed=args.seed, split_sizes=split_sizes)
    write_manifest(entries, args.out)
    print(f"Wrote {len(entries)} entries to {args.out}")


if __name__ == "__main__":
    main()

