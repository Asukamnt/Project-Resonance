"""Manifest generator for Task3 (Arithmetic Mod)."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .manifest import ManifestEntry, write_manifest
from ..task3 import MOD_OPERATOR

DEFAULT_SPLIT_SIZES: Dict[str, int] = {
    "train": 800,
    "val": 200,
    "iid_test": 200,
    "ood_digits": 200,
    "ood_compose": 200,  # Multi-step mod expressions (A%B%C)
    "ood_length": 200,   # Longer expressions
}

BALANCE_BUCKET_MOD = 10
DEFAULT_PRESET = "full"


@dataclass(frozen=True)
class Task3Preset:
    """Configuration for Task3 curriculum presets."""

    name: str
    iid_dividend_range: Tuple[int, int]
    iid_divisor_range: Tuple[int, int]
    ood_dividend_range: Tuple[int, int]
    ood_divisor_range: Tuple[int, int]

    def ranges_for_split(self, split: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if split == "ood_digits":
            return self.ood_dividend_range, self.ood_divisor_range
        return self.iid_dividend_range, self.iid_divisor_range


PRESETS: Dict[str, Task3Preset] = {
    "full": Task3Preset(
        name="full",
        iid_dividend_range=(0, 99),
        iid_divisor_range=(1, 99),
        ood_dividend_range=(100, 999),
        ood_divisor_range=(100, 999),
    ),
    "easy": Task3Preset(
        name="easy",
        iid_dividend_range=(0, 9999),
        iid_divisor_range=(2, 9),
        ood_dividend_range=(10000, 99999),
        ood_divisor_range=(2, 9),
    ),
    "tiny": Task3Preset(
        name="tiny",
        iid_dividend_range=(0, 99),
        iid_divisor_range=(2, 9),
        ood_dividend_range=(100, 999),
        ood_divisor_range=(2, 9),
    ),
}


def _int_to_tokens(value: int) -> List[str]:
    return list(str(value))


def _expression_tokens(dividend: int, divisor: int) -> List[str]:
    """Generate single-step expression: A % B"""
    tokens = _int_to_tokens(dividend) + [MOD_OPERATOR] + _int_to_tokens(divisor)
    return tokens


def _multi_step_expression_tokens(operands: List[int]) -> List[str]:
    """Generate multi-step expression: A % B % C % ...
    
    Example: [17, 5, 3] -> ['1', '7', '%', '5', '%', '3']
    """
    if len(operands) < 2:
        raise ValueError("Need at least 2 operands for multi-step expression")
    
    tokens: List[str] = []
    for i, op in enumerate(operands):
        if i > 0:
            tokens.append(MOD_OPERATOR)
        tokens.extend(_int_to_tokens(op))
    
    return tokens


def _sample_numbers(
    rng: np.random.Generator,
    dividend_range: Tuple[int, int],
    divisor_range: Tuple[int, int],
) -> Tuple[int, int]:
    div_low, div_high = dividend_range
    divisor_low, divisor_high = divisor_range
    dividend = int(rng.integers(div_low, div_high + 1))
    divisor = int(rng.integers(max(1, divisor_low), max(1, divisor_high) + 1))
    if divisor == 0:
        divisor = 1
    return dividend, divisor


def _valid_remainder_buckets(
    dividend_range: Tuple[int, int],
    divisor_range: Tuple[int, int],
    bucket_mod: int,
) -> List[int]:
    div_low, div_high = dividend_range
    divisor_low, divisor_high = divisor_range
    divisor_low = max(1, divisor_low)
    buckets: set[int] = set()
    for divisor in range(divisor_low, max(divisor_low, divisor_high) + 1):
        for dividend in range(div_low, div_high + 1):
            remainder = dividend % divisor
            buckets.add(remainder % bucket_mod)
            if len(buckets) == bucket_mod:
                return sorted(buckets)
    return sorted(buckets)


def build_task3_manifest(
    *,
    seed: int,
    split_sizes: Dict[str, int] | None = None,
    preset: str = DEFAULT_PRESET,
    balance_remainder: bool = False,
) -> List[ManifestEntry]:
    sizes = split_sizes or DEFAULT_SPLIT_SIZES
    if preset not in PRESETS:
        raise ValueError(f"Unknown Task3 preset '{preset}'. Choices: {sorted(PRESETS)}")
    preset_cfg = PRESETS[preset]
    rng = np.random.default_rng(seed)
    entries: List[ManifestEntry] = []
    seen_per_split: Dict[str, set[Tuple[str, ...]]] = {}

    def _add_samples(
        split: str,
        count: int,
        dividend_range: Tuple[int, int],
        divisor_range: Tuple[int, int],
        difficulty: str,
    ) -> None:
        attempts = 0
        max_attempts = max(count * 200, 1000)
        local_rng = np.random.default_rng(rng.integers(0, 2**63))
        added = 0
        bucket_limits: Dict[int, int] | None = None
        if balance_remainder and count > 0:
            buckets = _valid_remainder_buckets(dividend_range, divisor_range, BALANCE_BUCKET_MOD)
            if buckets:
                cap = math.ceil(count / len(buckets))
                bucket_limits = {bucket: cap for bucket in buckets}
        while added < count:
            attempts += 1
            if attempts > max_attempts:
                if bucket_limits is not None:
                    # Relax remainder balancing to avoid infinite loops when buckets cannot be filled.
                    bucket_limits = None
                    attempts = 0
                    continue
                raise RuntimeError(f"Unable to generate enough unique expressions for split '{split}'")
            dividend, divisor = _sample_numbers(local_rng, dividend_range, divisor_range)
            tokens = tuple(_expression_tokens(dividend, divisor))
            split_seen = seen_per_split.setdefault(split, set())
            if tokens in split_seen:
                continue
            remainder_bucket_ok = True
            if bucket_limits is not None:
                remainder = dividend % divisor
                bucket = remainder % BALANCE_BUCKET_MOD
                limit = bucket_limits.get(bucket)
                if limit is None:
                    limit = math.ceil(count / max(1, len(bucket_limits)))
                    bucket_limits[bucket] = limit
                if limit <= 0:
                    remainder_bucket_ok = False
                else:
                    bucket_limits[bucket] = limit - 1
            if not remainder_bucket_ok:
                continue
            split_seen.add(tokens)
            example_id = f"{split}-{added:06d}"
            entry = ManifestEntry(
                split=split,
                symbols=list(tokens),
                length=len(tokens),
                difficulty_tag=difficulty,
                example_id=example_id,
                seed=seed,
                sequence_seed=int(local_rng.integers(0, 2**63)),
            )
            entries.append(entry)
            added += 1

    for split_name in ("train", "val", "iid_test", "ood_digits"):
        count = sizes.get(split_name, 0)
        if count <= 0:
            continue
        dividend_range, divisor_range = preset_cfg.ranges_for_split(split_name)
        difficulty = "iid" if split_name != "ood_digits" else "ood_digits"
        _add_samples(split_name, count, dividend_range, divisor_range, difficulty)

    # Generate multi-step (compose) samples: A % B % C
    def _add_compose_samples(
        split: str,
        count: int,
        num_steps: int,
        operand_range: Tuple[int, int],
        difficulty: str,
    ) -> None:
        """Generate multi-step mod expressions with left-to-right evaluation."""
        local_rng = np.random.default_rng(rng.integers(0, 2**63))
        split_seen = seen_per_split.setdefault(split, set())
        added = 0
        attempts = 0
        max_attempts = count * 200
        
        while added < count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Unable to generate enough compose samples for '{split}'")
            
            # Generate num_steps + 1 operands
            operands = [
                int(local_rng.integers(operand_range[0], operand_range[1] + 1))
                for _ in range(num_steps + 1)
            ]
            
            # Ensure no zero divisors
            for i in range(1, len(operands)):
                if operands[i] == 0:
                    operands[i] = 1
            
            tokens = tuple(_multi_step_expression_tokens(operands))
            if tokens in split_seen:
                continue
            
            split_seen.add(tokens)
            example_id = f"{split}-{added:06d}"
            entry = ManifestEntry(
                split=split,
                symbols=list(tokens),
                length=len(tokens),
                difficulty_tag=difficulty,
                example_id=example_id,
                seed=seed,
                sequence_seed=int(local_rng.integers(0, 2**63)),
            )
            entries.append(entry)
            added += 1

    # ood_compose: 2-step expressions (A % B % C)
    compose_count = sizes.get("ood_compose", 0)
    if compose_count > 0:
        # Use small numbers for compose to focus on multi-step logic
        _add_compose_samples(
            "ood_compose",
            compose_count,
            num_steps=2,  # 2 '%' operators -> 3 operands
            operand_range=(1, 99),
            difficulty="ood_compose",
        )

    # ood_length: longer single-step expressions (larger numbers)
    ood_length_count = sizes.get("ood_length", 0)
    if ood_length_count > 0:
        dividend_range = (1000, 9999)  # 4-digit dividends
        divisor_range = (10, 99)
        _add_samples("ood_length", ood_length_count, dividend_range, divisor_range, "ood_length")

    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task3 manifest JSONL.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("manifests/task3.jsonl"),
        help="Output file path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=DEFAULT_SPLIT_SIZES["train"])
    parser.add_argument("--val-size", type=int, default=DEFAULT_SPLIT_SIZES["val"])
    parser.add_argument("--iid-test-size", type=int, default=DEFAULT_SPLIT_SIZES["iid_test"])
    parser.add_argument("--ood-digits-size", type=int, default=DEFAULT_SPLIT_SIZES["ood_digits"])
    parser.add_argument("--ood-compose-size", type=int, default=DEFAULT_SPLIT_SIZES["ood_compose"],
                       help="Size of ood_compose split (multi-step A%%B%%C expressions).")
    parser.add_argument("--ood-length-size", type=int, default=DEFAULT_SPLIT_SIZES["ood_length"],
                       help="Size of ood_length split (longer expressions with larger numbers).")
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESETS),
        default=DEFAULT_PRESET,
        help="Task3 curriculum preset (easy=single-digit remainders, full=original ranges).",
    )
    parser.add_argument(
        "--balance-remainder",
        action="store_true",
        help="Balance remainder buckets (mod 10) to avoid degenerate distributions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "iid_test": args.iid_test_size,
        "ood_digits": args.ood_digits_size,
        "ood_compose": args.ood_compose_size,
        "ood_length": args.ood_length_size,
    }
    entries = build_task3_manifest(
        seed=args.seed,
        split_sizes=split_sizes,
        preset=args.preset,
        balance_remainder=args.balance_remainder,
    )
    write_manifest(entries, args.out)
    
    # Print split statistics
    from collections import Counter
    split_counts = Counter(e.split for e in entries)
    print(f"Wrote {len(entries)} Task3 entries to {args.out}")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")


if __name__ == "__main__":
    main()
