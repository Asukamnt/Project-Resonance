from __future__ import annotations

import numpy as np

from jericho.baselines import predict_wave_identity, predict_wave_oracle_mod
from jericho.symbols import GAP_DUR, SR, TONE_DUR, encode_symbols_to_wave
from jericho.scorer import decode_wave_to_symbols


def test_oracle_mod_exact_match():
    tokens = ["1", "3", "%", "5"]
    rng = np.random.default_rng(0)
    input_wave = encode_symbols_to_wave(tokens, sr=SR, tone_dur=TONE_DUR, gap_dur=GAP_DUR, rng=rng)
    pred_wave = predict_wave_oracle_mod(input_wave)
    pred_symbols = decode_wave_to_symbols(pred_wave)
    assert pred_symbols == ["3"]


def test_identity_baseline_unchanged():
    tokens = ["A", "B", "C"]
    wave = encode_symbols_to_wave(tokens, rng=np.random.default_rng(1))
    pred_wave = predict_wave_identity(wave, rng_seed=0)
    pred_symbols = decode_wave_to_symbols(pred_wave)
    assert pred_symbols == tokens

