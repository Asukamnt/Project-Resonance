"""Trivial baseline to validate Task1 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from jericho.data.manifest import ManifestEntry, read_manifest
from jericho.scorer import decode_wave_to_symbols, exact_match
from jericho.symbols import encode_symbols_to_wave


def run_trivial_baseline(entries: Sequence[ManifestEntry]) -> list[dict]:
    """Produce predictions by round-tripping symbols through the encoder/decoder."""
    results: list[dict] = []
    for entry in entries:
        wave_in = encode_symbols_to_wave(entry.symbols)
        decoded = decode_wave_to_symbols(wave_in)
        wave_out = encode_symbols_to_wave(decoded)
        em = exact_match(decoded, entry.symbols)
        results.append(
            {
                "example_id": entry.example_id,
                "split": entry.split,
                "target_symbols": entry.symbols,
                "decoded_symbols": decoded,
                "exact_match": em,
                "output_wave_samples": len(wave_out),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trivial Task1 baseline.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/task1.jsonl"), help="Manifest path."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "val", "iid_test", "ood_length", "ood_symbol"],
        help="Dataset splits used for the baseline.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/baselines/trivial_predictions.jsonl"),
        help="Where to store baseline predictions (JSONL).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_entries: list[ManifestEntry] = []
    for split in args.splits:
        entries = read_manifest(args.manifest, split=split)
        if not entries:
            raise SystemExit(f"No entries found for split='{split}' in {args.manifest}")
        all_entries.extend(entries)

    predictions = run_trivial_baseline(all_entries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")

    # Report aggregate accuracy per split.
    summary: dict[str, list[float]] = {}
    for item in predictions:
        summary.setdefault(item["split"], []).append(item["exact_match"])
    for split, scores in summary.items():
        avg = sum(scores) / len(scores)
        print(f"[{split}] Exact Match: {avg:.4f} ({len(scores)} samples)")
    print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()

