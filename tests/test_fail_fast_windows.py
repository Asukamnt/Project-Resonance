import pytest
import torch

from jericho.pipelines.task3_mod_audio import (
    assert_mod_targets_without_percent,
    validate_answer_frame_windows,
    validate_answer_windows,
    validate_ctc_inputs,
)


def test_validate_answer_windows_rejects_zero_or_negative():
    starts = torch.tensor([0, 5], dtype=torch.long)
    lens = torch.tensor([0, 3], dtype=torch.long)
    waves = torch.tensor([10, 10], dtype=torch.long)

    with pytest.raises(ValueError, match="length must be >0"):
        validate_answer_windows(starts, lens, waves, context="unit_test")


def test_validate_answer_frame_windows_rejects_overflow():
    starts = torch.tensor([2], dtype=torch.long)
    lens = torch.tensor([5], dtype=torch.long)
    frame_counts = torch.tensor([6], dtype=torch.long)

    with pytest.raises(ValueError, match="exceed available mask"):
        validate_answer_frame_windows(
            starts, lens, frame_counts, context="unit_test_frames"
        )


def test_validate_answer_frame_windows_rejects_misaligned_mask():
    starts = torch.tensor([4], dtype=torch.long)
    lens = torch.tensor([2], dtype=torch.long)
    frame_counts = torch.tensor([5], dtype=torch.long)

    with pytest.raises(ValueError, match="exceed available mask"):
        validate_answer_frame_windows(
            starts, lens, frame_counts, context="unit_test_frames_misaligned"
        )


def test_validate_ctc_inputs_rejects_zero_input_length():
    input_lengths = torch.tensor([0, 3], dtype=torch.long)
    target_lengths = torch.tensor([1, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="input length must be >0"):
        validate_ctc_inputs(
            input_lengths,
            target_lengths,
            context="unit_test_ctc",
            start_samples=torch.tensor([0, 0], dtype=torch.long),
            len_samples=torch.tensor([5, 5], dtype=torch.long),
        )


def test_assert_mod_targets_without_percent_rejects_percent_id():
    targets = torch.tensor([1, 7, 11], dtype=torch.long)
    with pytest.raises(ValueError, match="percent_id=5"):
        assert_mod_targets_without_percent(targets, percent_id=5)
