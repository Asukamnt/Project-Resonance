"""Sanity evaluation over Task1 manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from jericho.data.manifest import ManifestEntry, read_manifest
from jericho.scorer import decode_wave_to_symbols, exact_match
from jericho.symbols import encode_symbols_to_wave


def evaluate_entries(entries: Sequence[ManifestEntry]) -> float:
    """Compute exact match over manifest entries."""
    if not entries:
        return 0.0

    matches = 0.0
    for entry in entries:
        wave = encode_symbols_to_wave(entry.symbols)
        decoded = decode_wave_to_symbols(wave)
        matches += exact_match(decoded, entry.symbols)
    return matches / len(entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task1 manifest via roundtrip.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/task1.jsonl"), help="Manifest path."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="iid_test",
        choices=["train", "val", "iid_test", "ood_length", "ood_symbol"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = read_manifest(args.manifest, split=args.split)
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        raise SystemExit(f"No entries found for split='{args.split}' in {args.manifest}")

    em = evaluate_entries(entries)
    print(f"[{args.split}] Exact Match: {em:.4f} ({len(entries)} samples)")


if __name__ == "__main__":
    main()

