#!/usr/bin/env python3
"""Generate manifest for Phase 3 cross-domain Task2 (IPD → Audio).

Usage:
    python -m jericho.data.make_cross_domain_manifest \\
        --output manifests/phase3_task2_bracket_cross_domain.jsonl \\
        --train-size 500 --val-size 100 --iid-test-size 100 --ood-length-size 100

Reference: docs/phase2-4/phase3_light_to_sound.md
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class CrossDomainEntry:
    """A manifest entry for cross-domain Task2."""
    id: str
    symbols: List[str]  # Input bracket sequence
    task: str = "bracket_cross"
    split: str = "iid_train"
    domain_in: str = "ipd"
    domain_out: str = "audio"


def generate_balanced_bracket(length: int, rng: random.Random) -> List[str]:
    """Generate a balanced bracket sequence of given length."""
    if length % 2 != 0:
        raise ValueError("Length must be even for balanced brackets")
    
    # Use counting method: track balance, never go negative
    result = []
    open_count = 0
    close_count = 0
    half = length // 2
    
    for _ in range(length):
        if open_count == half:
            # Must close
            result.append(")")
            close_count += 1
        elif close_count == open_count:
            # Must open (can't close if no open brackets)
            result.append("(")
            open_count += 1
        else:
            # Can do either
            if rng.random() < 0.5:
                result.append("(")
                open_count += 1
            else:
                result.append(")")
                close_count += 1
    
    return result


def generate_unbalanced_bracket(length: int, rng: random.Random) -> List[str]:
    """Generate an unbalanced bracket sequence of given length."""
    # Generate random sequence until we get unbalanced
    for _ in range(100):
        seq = [rng.choice(["(", ")"]) for _ in range(length)]
        # Check if unbalanced
        count = 0
        balanced = True
        for c in seq:
            count += 1 if c == "(" else -1
            if count < 0:
                balanced = False
                break
        if count != 0:
            balanced = False
        
        if not balanced:
            return seq
    
    # Fallback: force unbalanced
    seq = ["("] * (length - 1) + [")"]
    return seq


def generate_cross_domain_manifest(
    train_size: int = 500,
    val_size: int = 100,
    iid_test_size: int = 100,
    ood_length_size: int = 100,
    iid_lengths: List[int] = None,
    ood_lengths: List[int] = None,
    balance_ratio: float = 0.5,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> List[CrossDomainEntry]:
    """Generate cross-domain manifest.
    
    Parameters
    ----------
    train_size : int
        Number of training samples
    val_size : int
        Number of validation samples  
    iid_test_size : int
        Number of IID test samples
    ood_length_size : int
        Number of OOD length test samples
    iid_lengths : List[int]
        Bracket lengths for IID (default [4, 6, 8])
    ood_lengths : List[int]
        Bracket lengths for OOD (default [10, 12, 14, 16])
    balance_ratio : float
        Ratio of balanced (V) vs unbalanced (X) samples
    seed : int
        Random seed
    output_path : Path
        If provided, write manifest to this file
        
    Returns
    -------
    List[CrossDomainEntry]
        Generated manifest entries
    """
    if iid_lengths is None:
        iid_lengths = [4, 6, 8]
    if ood_lengths is None:
        ood_lengths = [10, 12, 14, 16]
    
    rng = random.Random(seed)
    entries: List[CrossDomainEntry] = []
    entry_id = 0
    
    def generate_samples(
        n: int,
        split: str,
        lengths: List[int],
    ) -> List[CrossDomainEntry]:
        nonlocal entry_id
        samples = []
        
        for _ in range(n):
            length = rng.choice(lengths)
            is_balanced = rng.random() < balance_ratio
            
            if is_balanced:
                symbols = generate_balanced_bracket(length, rng)
            else:
                symbols = generate_unbalanced_bracket(length, rng)
            
            entry = CrossDomainEntry(
                id=f"cross_{split}_{entry_id:06d}",
                symbols=symbols,
                split=split,
            )
            samples.append(entry)
            entry_id += 1
        
        return samples
    
    # Generate splits
    entries.extend(generate_samples(train_size, "iid_train", iid_lengths))
    entries.extend(generate_samples(val_size, "iid_val", iid_lengths))
    entries.extend(generate_samples(iid_test_size, "iid_test", iid_lengths))
    entries.extend(generate_samples(ood_length_size, "ood_length", ood_lengths))
    
    # Write to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        
        print(f"Wrote {len(entries)} entries to {output_path}")
        
        # Print split summary
        from collections import Counter
        split_counts = Counter(e.split for e in entries)
        for split, count in sorted(split_counts.items()):
            balanced = sum(1 for e in entries if e.split == split and _is_balanced(e.symbols))
            print(f"  {split}: {count} samples ({balanced} balanced, {count - balanced} unbalanced)")
    
    return entries


def _is_balanced(symbols: List[str]) -> bool:
    """Check if bracket sequence is balanced."""
    count = 0
    for s in symbols:
        count += 1 if s == "(" else -1
        if count < 0:
            return False
    return count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 3 cross-domain manifest"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("manifests/phase3_task2_bracket_cross_domain.jsonl"),
        help="Output manifest path",
    )
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--iid-test-size", type=int, default=100)
    parser.add_argument("--ood-length-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--iid-lengths",
        type=int,
        nargs="+",
        default=[4, 6, 8],
        help="Bracket lengths for IID splits",
    )
    parser.add_argument(
        "--ood-lengths",
        type=int,
        nargs="+",
        default=[10, 12, 14, 16],
        help="Bracket lengths for OOD split",
    )
    parser.add_argument(
        "--balance-ratio",
        type=float,
        default=0.5,
        help="Ratio of balanced (V) samples",
    )
    
    args = parser.parse_args()
    
    generate_cross_domain_manifest(
        train_size=args.train_size,
        val_size=args.val_size,
        iid_test_size=args.iid_test_size,
        ood_length_size=args.ood_length_size,
        iid_lengths=args.iid_lengths,
        ood_lengths=args.ood_lengths,
        balance_ratio=args.balance_ratio,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

