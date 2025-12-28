#!/usr/bin/env python3
"""Verify Task3 oracle baseline on multi-step expressions.

验收目标：Oracle 在所有 split 上达到 EM=1.0
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.scorer import decode_wave_to_symbols, exact_match
from jericho.task3 import (
    target_symbols_for_task3,
    count_mod_steps,
    synthesise_task3_target_wave,
)


def verify_oracle(manifest_path: Path, split: str | None = None) -> dict:
    """Verify oracle achieves EM=1.0 on a manifest split."""
    entries = list(read_manifest(manifest_path))
    
    if split:
        entries = [e for e in entries if e.split == split]
    
    if not entries:
        return {"error": f"No entries found for split '{split}'"}
    
    results = {
        "split": split or "all",
        "total": len(entries),
        "correct": 0,
        "errors": [],
    }
    
    for entry in entries:
        try:
            # Get oracle prediction
            oracle_target = target_symbols_for_task3(entry.symbols)
            
            # Synthesize target wave and decode
            target_wave = synthesise_task3_target_wave(entry.symbols)
            decoded = decode_wave_to_symbols(target_wave)
            
            # Check if decoded matches oracle prediction
            if decoded == oracle_target:
                results["correct"] += 1
            else:
                results["errors"].append({
                    "id": entry.example_id,
                    "symbols": entry.symbols,
                    "oracle": oracle_target,
                    "decoded": decoded,
                })
        except Exception as e:
            results["errors"].append({
                "id": entry.example_id,
                "symbols": entry.symbols,
                "exception": str(e),
            })
    
    results["em"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Task3 oracle baseline")
    parser.add_argument("--manifest", type=Path, default=Path("manifests/task3_multistep.jsonl"))
    parser.add_argument("--split", type=str, default=None, help="Specific split to verify")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if not args.manifest.exists():
        print(f"Error: Manifest not found: {args.manifest}")
        sys.exit(1)
    
    # Get all splits
    entries = list(read_manifest(args.manifest))
    splits = sorted(set(e.split for e in entries))
    
    print(f"Verifying Task3 oracle on: {args.manifest}")
    print("=" * 60)
    
    all_pass = True
    
    for split in splits:
        if args.split and split != args.split:
            continue
        
        # Count multi-step entries
        split_entries = [e for e in entries if e.split == split]
        multi_step = sum(1 for e in split_entries if count_mod_steps(e.symbols) > 1)
        
        result = verify_oracle(args.manifest, split)
        
        status = "PASS" if result["em"] == 1.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        
        print(f"[{status}] {split}: EM={result['em']:.4f} ({result['correct']}/{result['total']})")
        print(f"       Multi-step: {multi_step}/{len(split_entries)}")
        
        if args.verbose and result["errors"]:
            print(f"       Errors ({len(result['errors'])}):")
            for err in result["errors"][:5]:
                print(f"         - {err}")
    
    print("=" * 60)
    if all_pass:
        print("Oracle verification PASSED")
    else:
        print("Oracle verification FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

