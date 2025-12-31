from __future__ import annotations

import torch

from jericho.models.losses import MultiResolutionSTFTConfig, multi_resolution_stft_loss
from jericho.pipelines.task3_mod_audio import ctc_greedy_decode


def test_multi_resolution_stft_matches_when_waves_equal():
    cfg = MultiResolutionSTFTConfig(fft_sizes=(256,), hop_scale=0.25, win_scale=1.0)
    wave = torch.sin(torch.linspace(0, 2 * torch.pi, steps=512)).unsqueeze(0)
    lengths = torch.tensor([wave.size(1)])

    loss, details = multi_resolution_stft_loss(wave, wave, lengths=lengths, config=cfg)

    assert loss.item() < 1e-6
    assert details["spectral_convergence"].item() < 1e-6
    assert details["log_magnitude"].item() < 1e-6


def test_multi_resolution_stft_respects_lengths_mask():
    cfg = MultiResolutionSTFTConfig(fft_sizes=(256,), hop_scale=0.5, win_scale=1.0)
    pred = torch.zeros(1, 512)
    target = torch.zeros(1, 512)
    pred[:, 400:] = 1.0
    full_loss, _ = multi_resolution_stft_loss(pred, target, config=cfg)

    lengths = torch.tensor([384])
    masked_loss, _ = multi_resolution_stft_loss(pred, target, lengths=lengths, config=cfg)

    assert masked_loss < full_loss


def test_ctc_greedy_decode_collapses_repeats_and_blanks():
    logits = torch.tensor(
        [
            [
                [0.0, 4.0, 0.0],  # A
                [3.0, 0.0, 0.0],  # blank
                [0.0, 0.0, 4.0],  # B
                [0.0, 0.0, 4.0],  # B repeated -> collapsed
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, True, True]])
    decoded = ctc_greedy_decode(logits, mask, {1: "A", 2: "B"}, blank_id=0)

    assert decoded == [["A", "B"]]



































