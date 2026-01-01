"""Loss utilities for audio reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch


@dataclass
class MultiResolutionSTFTConfig:
    """Configuration for multi-resolution STFT loss."""

    fft_sizes: Tuple[int, ...] = (512, 1024, 2048)
    hop_scale: float = 0.25
    win_scale: float = 1.0
    spectral_convergence_weight: float = 1.0
    log_mag_weight: float = 1.0
    eps: float = 1e-7


def _mask_wave(
    wave: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    """Zero out padded samples based on lengths."""

    if lengths is None:
        return wave
    if wave.dim() != 2:
        raise ValueError(f"_mask_wave expects (batch, time), got shape={tuple(wave.shape)}")
    max_len = wave.size(1)
    lengths = lengths.to(device=wave.device).clamp(min=0, max=max_len)
    time_axis = torch.arange(max_len, device=wave.device).unsqueeze(0)
    mask = time_axis < lengths.unsqueeze(1)
    return wave * mask


def _stft(
    wave: torch.Tensor,
    fft_size: int,
    hop_scale: float,
    win_scale: float,
) -> torch.Tensor:
    hop_length = max(1, int(round(fft_size * hop_scale)))
    win_length = max(1, int(round(fft_size * win_scale)))
    window = torch.hann_window(win_length, device=wave.device, dtype=wave.dtype)
    return torch.stft(
        wave,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )


def multi_resolution_stft_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    config: MultiResolutionSTFTConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute multi-resolution STFT loss (spectral convergence + log magnitude).

    Parameters
    ----------
    pred_wave:
        Predicted waveform tensor of shape (batch, time).
    target_wave:
        Target waveform tensor of shape (batch, time).
    lengths:
        Optional per-sample valid lengths. Samples beyond each length are masked
        to zero before computing the loss to avoid padded silence dominating.
    config:
        Optional :class:`MultiResolutionSTFTConfig` controlling STFT settings
        and weights.
    """

    if config is None:
        config = MultiResolutionSTFTConfig()

    if pred_wave.shape != target_wave.shape:
        raise ValueError(
            f"multi_resolution_stft_loss expects matching shapes, "
            f"got pred={tuple(pred_wave.shape)} target={tuple(target_wave.shape)}"
        )
    if pred_wave.dim() != 2:
        raise ValueError(
            f"multi_resolution_stft_loss expects tensors of shape (batch, time), "
            f"got pred.dim()={pred_wave.dim()}"
        )

    pred_masked = _mask_wave(pred_wave, lengths)
    target_masked = _mask_wave(target_wave, lengths)

    sc_losses: list[torch.Tensor] = []
    mag_losses: list[torch.Tensor] = []

    for fft_size in config.fft_sizes:
        pred_spec = _stft(pred_masked, fft_size, config.hop_scale, config.win_scale)
        target_spec = _stft(target_masked, fft_size, config.hop_scale, config.win_scale)

        diff = pred_spec - target_spec
        sc = torch.linalg.norm(diff, dim=(-2, -1)) / (
            torch.linalg.norm(target_spec, dim=(-2, -1)) + config.eps
        )
        sc_losses.append(sc.mean())

        pred_mag = pred_spec.abs()
        target_mag = target_spec.abs()
        log_mag = (torch.log(pred_mag + config.eps) - torch.log(target_mag + config.eps)).abs()
        mag_losses.append(log_mag.mean())

    sc_loss = torch.stack(sc_losses).mean()
    mag_loss = torch.stack(mag_losses).mean()
    total = config.spectral_convergence_weight * sc_loss + config.log_mag_weight * mag_loss

    details = {"spectral_convergence": sc_loss, "log_magnitude": mag_loss}
    return total, details


__all__ = ["MultiResolutionSTFTConfig", "multi_resolution_stft_loss"]





































