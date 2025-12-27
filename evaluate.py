"""Evaluation utility for Stage A baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from jericho.data import ManifestEntry, read_manifest, synthesise_entry_wave
from jericho.scorer import decode_wave_to_symbols, exact_match

SPLIT_ORDER: list[str] = ["train", "val", "iid_test", "ood_length", "ood_symbol"]
VALID_SPLITS: set[str] = set(SPLIT_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task1 baselines.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest JSONL for symbol→audio sanity evaluation.",
    )
    parser.add_argument(
        "--preds",
        type=Path,
        default=None,
        help="Prediction JSONL (e.g. produced by train.py).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split filter (applies to manifest or predictions).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples evaluated.",
    )
    return parser.parse_args()


def summarise(split_totals: Dict[str, Tuple[float, int]]) -> None:
    overall_matches = sum(matches for matches, _ in split_totals.values())
    overall_count = sum(count for _, count in split_totals.values())

    ordered = [s for s in SPLIT_ORDER if s in split_totals]
    ordered.extend(sorted(set(split_totals.keys()) - set(ordered)))

    for split in ordered:
        matches, count = split_totals[split]
        em = matches / count if count else 0.0
        print(f"[{split}] Exact Match: {em:.4f} ({count} samples)")

    if overall_count:
        overall_em = overall_matches / overall_count
        print(f"[overall] Exact Match: {overall_em:.4f} ({overall_count} samples)")


def evaluate_manifest(manifest: Path, split: str | None, limit: int | None) -> None:
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")

    entries = read_manifest(manifest, split=split)
    if not entries:
        target = split if split else "any split"
        raise SystemExit(f"No entries found for {target} in {manifest}")

    split_map: Dict[str, List[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        split_map[entry.split].append(entry)

    totals: Dict[str, Tuple[float, int]] = {}
    for split_name, items in split_map.items():
        if limit is not None:
            items = items[:limit]
        matches = 0.0
        for entry in items:
            wave = synthesise_entry_wave(entry)
            decoded = decode_wave_to_symbols(wave)
            matches += exact_match(decoded, entry.symbols)
        totals[split_name] = (matches, len(items))

    summarise(totals)


def evaluate_predictions(preds_path: Path, split: str | None, limit: int | None) -> None:
    if not preds_path.exists():
        raise SystemExit(f"Predictions file not found: {preds_path}")

    totals: Dict[str, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    processed = 0
    with preds_path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and processed >= limit:
                break
            record = json.loads(line)
            record_split = record.get("split", "unknown")
            if split is not None and record_split != split:
                continue
            gold = list(record.get("gold_symbols", []))
            pred = list(record.get("pred_symbols", []))
            em = exact_match(pred, gold)
            match_total, count_total = totals[record_split]
            totals[record_split] = (match_total + em, count_total + 1)
            processed += 1

    if not totals:
        target = split if split else "entries"
        raise SystemExit(f"No prediction entries evaluated for {target}.")

    summarise(dict(totals))


def main() -> None:
    args = parse_args()
    if (args.manifest is None) == (args.preds is None):
        raise SystemExit("Specify exactly one of --manifest or --preds.")

    if args.manifest is not None:
        if args.split is not None and args.split not in VALID_SPLITS:
            raise SystemExit(f"Unsupported split '{args.split}'.")
        evaluate_manifest(args.manifest, args.split, args.limit)
    else:
        evaluate_predictions(args.preds, args.split, args.limit)


if __name__ == "__main__":
    main()

