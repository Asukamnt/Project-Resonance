#!/usr/bin/env python3
"""Generate manifest for Phase 3 Task3 Mod cross-domain (IPD → Audio).

Usage:
    python -m jericho.data.make_task3_cross_manifest \\
        --output manifests/phase3_task3_mod_cross_domain.jsonl

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
class Task3CrossEntry:
    """Manifest entry for Task3 cross-domain."""
    id: str
    symbols: List[str]  # Expression: ["4", "2", "%", "7"]
    task: str = "mod_cross"
    split: str = "iid_train"
    domain_in: str = "ipd"
    domain_out: str = "audio"


def generate_task3_cross_manifest(
    train_size: int = 500,
    val_size: int = 100,
    iid_test_size: int = 100,
    dividend_range: tuple = (0, 99),
    divisor_range: tuple = (2, 9),
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> List[Task3CrossEntry]:
    """Generate Task3 Mod cross-domain manifest.
    
    Generates A%B expressions where:
    - A ∈ dividend_range
    - B ∈ divisor_range (B ≠ 0)
    - Remainder is computed as A % B
    """
    rng = random.Random(seed)
    entries: List[Task3CrossEntry] = []
    entry_id = 0
    
    def generate_samples(n: int, split: str) -> List[Task3CrossEntry]:
        nonlocal entry_id
        samples = []
        seen = set()
        
        attempts = 0
        while len(samples) < n and attempts < n * 10:
            attempts += 1
            
            dividend = rng.randint(dividend_range[0], dividend_range[1])
            divisor = rng.randint(divisor_range[0], divisor_range[1])
            
            expr = f"{dividend}%{divisor}"
            if expr in seen:
                continue
            seen.add(expr)
            
            # Convert to symbol list
            symbols = list(expr)
            
            entry = Task3CrossEntry(
                id=f"mod_cross_{split}_{entry_id:06d}",
                symbols=symbols,
                split=split,
            )
            samples.append(entry)
            entry_id += 1
        
        return samples
    
    entries.extend(generate_samples(train_size, "iid_train"))
    entries.extend(generate_samples(val_size, "iid_val"))
    entries.extend(generate_samples(iid_test_size, "iid_test"))
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        
        print(f"Wrote {len(entries)} entries to {output_path}")
        
        # Summary
        from collections import Counter
        split_counts = Counter(e.split for e in entries)
        for split, count in sorted(split_counts.items()):
            print(f"  {split}: {count} samples")
        
        # Remainder distribution
        remainders = []
        for e in entries:
            expr = "".join(e.symbols)
            parts = expr.split("%")
            r = int(parts[0]) % int(parts[1])
            remainders.append(r)
        
        rem_dist = Counter(remainders)
        print(f"  Remainder distribution (top 5): {rem_dist.most_common(5)}")
    
    return entries


def main():
    parser = argparse.ArgumentParser(description="Generate Task3 Mod cross-domain manifest")
    parser.add_argument("--output", "-o", type=Path,
                       default=Path("manifests/phase3_task3_mod_cross_domain.jsonl"))
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--iid-test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    generate_task3_cross_manifest(
        train_size=args.train_size,
        val_size=args.val_size,
        iid_test_size=args.iid_test_size,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

