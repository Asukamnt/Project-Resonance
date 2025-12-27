import numpy as np
import torch

from jericho.pipelines.task3_mod_audio import TASK3_SYMBOLS, apply_tone_bank_render, render_tone_bank
from jericho.scorer import decode_wave_to_symbols
from jericho.symbols import SR, TONE_DUR


def _one_hot_digit(idx: int) -> torch.Tensor:
    probs = torch.zeros(1, 10, dtype=torch.float32)
    probs[:, idx] = 1.0
    return probs


def test_render_tone_bank_decodes_each_digit():
    num_samples = int(round(SR * TONE_DUR))
    for digit in range(10):
        probs = _one_hot_digit(digit)
        wave = render_tone_bank(probs, num_samples, sr=SR, phase=0.0)
        wave_np = wave.squeeze(0).detach().cpu().numpy().astype(np.float32)
        decoded_hard = decode_wave_to_symbols(wave_np, sr=SR, segmentation="hard")
        decoded_energy = decode_wave_to_symbols(wave_np, sr=SR, segmentation="energy")

        assert decoded_hard == [str(digit)]
        assert decoded_energy and decoded_energy[0] == str(digit)


def test_render_tone_bank_continuity_between_frames():
    frame_size = 160
    num_frames = 4
    num_samples = frame_size * num_frames
    probs = _one_hot_digit(5)
    wave = render_tone_bank(probs, num_samples, sr=SR, phase=0.0)
    wave_np = wave.squeeze(0).detach().cpu().numpy()

    boundary_indices = np.arange(frame_size, num_samples, frame_size)
    boundary_jumps = np.abs(wave_np[boundary_indices] - wave_np[boundary_indices - 1])
    max_jump = float(boundary_jumps.max())

    assert max_jump < 1.2, f"discontinuity detected: max_jump={max_jump:.4f}"


def test_apply_tone_bank_render_respects_digit_mass_floor():
    frame_size = 160
    frames = 2
    vocab_size = len(TASK3_SYMBOLS) + 1  # +1 for blank id
    symbol_probs = torch.full((1, frames, vocab_size), 1e-5, dtype=torch.float32)
    symbol_probs[..., 0] = 0.999
    symbol_probs = symbol_probs / symbol_probs.sum(dim=-1, keepdim=True)
    guided_frames = torch.zeros(1, frames, frame_size, dtype=torch.float32)
    window_mask = torch.ones(1, frames, dtype=torch.bool)
    answer_start_frames = torch.tensor([0], dtype=torch.long)
    answer_len_frames = torch.tensor([frames], dtype=torch.long)
    answer_len_samples = torch.tensor([frame_size * frames], dtype=torch.long)

    rendered = apply_tone_bank_render(
        guided_frames,
        symbol_probs,
        window_mask,
        answer_start_frames,
        answer_len_frames,
        answer_len_samples,
        mode="tone_bank_soft",
        render_weight=1.0,
        render_fixed_phase=0.0,
        frame_size=frame_size,
        hop_size=frame_size,
        sr=SR,
        answer_digit_mass_floor=0.8,
    )
    window = rendered[0, :frames]
    rms = float(torch.sqrt((window**2).mean()).item())
    assert rms > 1e-3

