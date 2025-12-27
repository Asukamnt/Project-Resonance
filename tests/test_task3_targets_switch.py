from __future__ import annotations

import torch

from jericho.data.make_task3_manifest import build_task3_manifest
from jericho.pipelines.task3_mod_audio import (
    Task3TrainingConfig,
    build_task3_vocab,
    collate_task3_train,
    prepare_task3_samples,
)
from jericho.task3 import MOD_OPERATOR


def test_task3_targets_switch_between_pretrain_and_mod():
    split_sizes = {"train": 4, "val": 0, "iid_test": 0, "ood_digits": 0}
    entries = build_task3_manifest(seed=321, split_sizes=split_sizes, preset="tiny", balance_remainder=True)
    config = Task3TrainingConfig(frame_size=160, hop_size=160)
    vocab = build_task3_vocab()
    percent_id = vocab[MOD_OPERATOR]

    samples = prepare_task3_samples(
        [e for e in entries if e.split == "train"],
        vocab,
        config.frame_size,
        config.hop_size,
        blank_id=config.ctc_blank_id,
        config=config,
    )
    batch = collate_task3_train(samples, config.frame_size, blank_id=config.ctc_blank_id)
    (
        _inputs,
        _targets_frames,
        mask,
        remainder_ctc_targets,
        remainder_target_lengths,
        _wave_lengths,
        _content_lengths,
        remainder_frame_symbol_targets,
        expression_frame_symbol_targets,
        _remainder_values,
        _answer_start_samples,
        _answer_len_samples,
        expression_ctc_targets,
        expression_target_lengths,
        _expression_len_samples,
    ) = batch

    assert remainder_ctc_targets.numel() > 0
    assert int(remainder_ctc_targets.max().item()) < percent_id
    assert (remainder_target_lengths == 1).all()

    # Frame CE targets for mod (answer window) should not include '%'
    if remainder_frame_symbol_targets.numel() > 0:
        max_frame_id = int(remainder_frame_symbol_targets.max().item())
        assert max_frame_id < percent_id

    # Expression targets (pretrain) should include '%'
    assert expression_ctc_targets.numel() > 0
    assert (expression_ctc_targets == percent_id).any()

    # Expression frame labels should also contain '%'
    if expression_frame_symbol_targets.numel() > 0:
        assert (expression_frame_symbol_targets == percent_id).any()

