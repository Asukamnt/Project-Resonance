"""Manifest generator for Task2 (Bracket Validity)."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .manifest import ManifestEntry, write_manifest
from ..task2 import (
    BRACKET_OPEN,
    BRACKET_CLOSE,
    VALID_SYMBOL,
    INVALID_SYMBOL,
    is_balanced,
    generate_balanced_brackets,
    generate_unbalanced_brackets,
)

DEFAULT_SPLIT_SIZES: Dict[str, int] = {
    "train": 800,
    "val": 200,
    "iid_test": 200,
    "ood_length": 200,
}


@dataclass(frozen=True)
class Task2Preset:
    """Configuration for Task2 curriculum presets."""

    name: str
    iid_length_range: Tuple[int, int]  # min, max length (inclusive)
    ood_length_range: Tuple[int, int]


PRESETS: Dict[str, Task2Preset] = {
    "full": Task2Preset(
        name="full",
        iid_length_range=(2, 12),
        ood_length_range=(14, 20),
    ),
    "easy": Task2Preset(
        name="easy",
        iid_length_range=(2, 8),
        ood_length_range=(10, 14),
    ),
    "tiny": Task2Preset(
        name="tiny",
        iid_length_range=(2, 6),
        ood_length_range=(8, 10),
    ),
}


def _generate_sample(
    rng: random.Random,
    length: int,
    is_valid: bool,
) -> List[str]:
    """Generate a bracket sequence of given length and validity."""
    if is_valid:
        # Balanced requires even length >= 2
        if length < 2:
            length = 2
        if length % 2 != 0:
            length += 1
        return generate_balanced_brackets(length, rng)
    else:
        return generate_unbalanced_brackets(length, rng)


def build_task2_manifest(
    seed: int,
    split_sizes: Dict[str, int] | None = None,
    preset: str = "full",
    balance_valid: bool = True,
) -> List[ManifestEntry]:
    """Build Task2 manifest entries.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    split_sizes:
        Dict mapping split names to counts.
    preset:
        One of 'full', 'easy', 'tiny'.
    balance_valid:
        If True, ensure 50/50 split between valid and invalid samples.

    Returns
    -------
    List[ManifestEntry]
        Generated manifest entries.
    """
    if split_sizes is None:
        split_sizes = DEFAULT_SPLIT_SIZES.copy()

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'; available: {sorted(PRESETS.keys())}")
    config = PRESETS[preset]

    rng = random.Random(seed)
    entries: List[ManifestEntry] = []

    for split, count in split_sizes.items():
        if count <= 0:
            continue

        is_ood = split.startswith("ood")
        length_range = config.ood_length_range if is_ood else config.iid_length_range
        difficulty_tag = "ood" if is_ood else "iid"

        for idx in range(count):
            # Determine validity
            if balance_valid:
                is_valid = idx % 2 == 0
            else:
                is_valid = rng.random() < 0.5

            # Sample length
            length = rng.randint(length_range[0], length_range[1])

            # Generate sequence
            seq_seed = rng.randint(0, 2**62)
            seq_rng = random.Random(seq_seed)
            symbols = _generate_sample(seq_rng, length, is_valid)

            # Sanity check
            actual_valid = is_balanced(symbols)
            if actual_valid != is_valid:
                # Retry with different seed
                seq_seed = rng.randint(0, 2**62)
                seq_rng = random.Random(seq_seed)
                symbols = _generate_sample(seq_rng, length, is_valid)

            entry = ManifestEntry(
                split=split,
                symbols=symbols,
                length=len(symbols),
                difficulty_tag=difficulty_tag,
                example_id=f"{split}-{idx:06d}",
                seed=seed,
                sequence_seed=seq_seed,
            )
            entries.append(entry)

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task2 bracket manifest")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="full",
        help="Difficulty preset",
    )
    parser.add_argument(
        "--train", type=int, default=DEFAULT_SPLIT_SIZES["train"], help="Train split size"
    )
    parser.add_argument(
        "--val", type=int, default=DEFAULT_SPLIT_SIZES["val"], help="Val split size"
    )
    parser.add_argument(
        "--iid-test",
        type=int,
        default=DEFAULT_SPLIT_SIZES["iid_test"],
        help="IID test split size",
    )
    parser.add_argument(
        "--ood-length",
        type=int,
        default=DEFAULT_SPLIT_SIZES["ood_length"],
        help="OOD length test split size",
    )
    parser.add_argument(
        "--balance-valid",
        action="store_true",
        default=True,
        help="Balance valid/invalid samples",
    )

    args = parser.parse_args()

    split_sizes = {
        "train": args.train,
        "val": args.val,
        "iid_test": args.iid_test,
        "ood_length": args.ood_length,
    }

    entries = build_task2_manifest(
        seed=args.seed,
        split_sizes=split_sizes,
        preset=args.preset,
        balance_valid=args.balance_valid,
    )

    write_manifest(entries, args.out)
    print(f"Wrote {len(entries)} Task2 entries to {args.out}")


if __name__ == "__main__":
    main()

