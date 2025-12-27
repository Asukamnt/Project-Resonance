from __future__ import annotations

import numpy as np

from jericho.baselines.identity import predict_wave_identity
from jericho.data.make_manifest import build_manifest
from jericho.data.manifest import write_manifest, read_manifest
from jericho.data.utils import synthesise_entry_wave
from jericho.scorer import decode_wave_to_symbols
from jericho.symbols import encode_symbols_to_wave


def test_identity_baseline_roundtrip_symbols():
    sequence = ["A", "C", "B", "E"]
    wave = encode_symbols_to_wave(sequence, rng=np.random.default_rng(7))
    pred_wave = predict_wave_identity(wave, rng_seed=0)
    decoded = decode_wave_to_symbols(pred_wave)
    assert decoded == sequence


def test_identity_baseline_manifest_entry(tmp_path):
    entries = build_manifest(
        seed=321,
        split_sizes={
            "train": 1,
            "val": 0,
            "iid_test": 0,
            "ood_length": 0,
            "ood_symbol": 0,
        },
    )
    manifest_path = tmp_path / "task1.jsonl"
    write_manifest(entries, manifest_path)
    entry = read_manifest(manifest_path, split="train")[0]

    input_wave = synthesise_entry_wave(entry)
    pred_wave = predict_wave_identity(input_wave, rng_seed=0)
    decoded = decode_wave_to_symbols(pred_wave)
    assert decoded == entry.symbols

