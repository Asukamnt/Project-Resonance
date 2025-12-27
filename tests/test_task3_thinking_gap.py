from __future__ import annotations

import numpy as np
import pytest

from jericho.data.manifest import ManifestEntry
from jericho.pipelines.task3_mod_audio import (
    Task3TrainingConfig,
    build_task3_vocab,
    prepare_task3_samples,
)
from jericho.scorer import decode_wave_to_symbols


def _make_entry() -> ManifestEntry:
    return ManifestEntry(
        split="train",
        symbols=list("13%5"),
        length=4,
        difficulty_tag="iid",
        example_id="gap-000000",
        seed=0,
        sequence_seed=0,
    )


def test_thinking_gap_inserts_silence_and_preserves_answer():
    entry = _make_entry()
    vocab = build_task3_vocab()
    config = Task3TrainingConfig(
        thinking_gap_s=0.2,
        thinking_gap_align=160,
        answer_window_only=True,
    )
    samples = prepare_task3_samples(
        [entry],
        vocab,
        config.frame_size,
        config.hop_size,
        blank_id=config.ctc_blank_id,
        config=config,
    )
    sample = samples[0]
    target_np = sample.target_wave.numpy()
    assert np.max(np.abs(target_np[: sample.answer_start_samples])) == pytest.approx(
        0.0, abs=1e-6
    )
    ans_slice = target_np[
        sample.answer_start_samples : sample.answer_start_samples + sample.answer_len_samples
    ]
    decoded = decode_wave_to_symbols(ans_slice, segmentation=config.decoder_segmentation)
    assert decoded[: len(sample.target_tokens)] == sample.target_tokens
    assert len(decoded) >= len(sample.target_tokens)


def test_thinking_gap_zero_keeps_answer_at_start():
    entry = _make_entry()
    vocab = build_task3_vocab()
    config = Task3TrainingConfig(thinking_gap_s=0.0, answer_window_only=True)
    samples = prepare_task3_samples(
        [entry],
        vocab,
        config.frame_size,
        config.hop_size,
        blank_id=config.ctc_blank_id,
        config=config,
    )
    sample = samples[0]
    assert sample.answer_start_samples == 0
    ans_slice = sample.target_wave.numpy()[: sample.answer_len_samples]
    decoded = decode_wave_to_symbols(ans_slice, segmentation=config.decoder_segmentation)
    assert decoded[: len(sample.target_tokens)] == sample.target_tokens
    assert len(decoded) >= len(sample.target_tokens)

