"""Scoring utilities for Task1 round-trip validation."""

from __future__ import annotations

from typing import Iterable, List, Literal

import numpy as np

from .symbols import GAP_DUR, SR, SYMBOL2FREQ, SYMBOLS, TONE_DUR


def _nearest_symbol(frequency: float) -> str:
    """
    Return the symbol whose carrier frequency is closest to ``frequency``.
    Uses SYMBOL2FREQ keys for OOD.
    """
    return min(SYMBOL2FREQ.keys(), key=lambda symbol: abs(SYMBOL2FREQ[symbol] - frequency))


def _resize_segment(segment: np.ndarray, target_samples: int) -> np.ndarray:
    if segment.size == target_samples:
        return segment
    if segment.size > target_samples:
        return segment[:target_samples]
    return np.pad(segment, (0, target_samples - segment.size))


def _segment_by_energy(
    wave: np.ndarray,
    sr: int,
    tone_samples: int,
    frame_ms: float = 0.01,
    hop_ms: float = 0.005,
    percentile: float = 0.6,
    gap_merge_ms: float = 15.0,
    min_duration_ms: float = 30.0,
) -> list[np.ndarray]:
    frame_size = max(1, int(round(sr * frame_ms)))
    hop_size = max(1, int(round(sr * hop_ms)))
    if hop_size > frame_size:
        hop_size = frame_size

    num_frames = max(1, 1 + (len(wave) - frame_size) // hop_size)
    energies = np.empty(num_frames, dtype=np.float32)
    for idx in range(num_frames):
        start = idx * hop_size
        end = start + frame_size
        frame = wave[start:end]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        energies[idx] = float(np.mean(frame ** 2))

    threshold = np.quantile(energies, percentile)
    active = energies >= threshold

    segments_frames: list[tuple[int, int]] = []
    idx = 0
    while idx < num_frames:
        if not active[idx]:
            idx += 1
            continue

        start_frame = idx
        while idx < num_frames and active[idx]:
            idx += 1
        end_frame = idx

        segments_frames.append((start_frame, end_frame))

    if not segments_frames:
        return [wave]

    merged_frames: list[tuple[int, int]] = []
    max_gap_frames = int(round((gap_merge_ms / 1000.0) / hop_ms))
    for start, end in segments_frames:
        if merged_frames:
            prev_start, prev_end = merged_frames[-1]
            gap_frames = start - prev_end
            if gap_frames <= max_gap_frames:
                merged_frames[-1] = (prev_start, end)
                continue
        merged_frames.append((start, end))

    min_samples = int(round(sr * (min_duration_ms / 1000.0)))
    normalised_segments: list[np.ndarray] = []
    for start_frame, end_frame in merged_frames:
        start_sample = start_frame * hop_size
        end_sample = min(len(wave), end_frame * hop_size + frame_size)
        if end_sample <= start_sample:
            continue
        segment = wave[start_sample:end_sample]
        if segment.size < min_samples:
            continue

        if segment.size > tone_samples:
            max_idx = int(np.argmax(np.square(segment)))
            start = max(0, max_idx - tone_samples // 2)
            end = start + tone_samples
            if end > segment.size:
                start = max(0, segment.size - tone_samples)
                end = segment.size
            normalised_segments.append(segment[start:end])
        else:
            normalised_segments.append(_resize_segment(segment, tone_samples))

    if not normalised_segments:
        normalised_segments = [wave if wave.size else np.zeros(tone_samples, dtype=wave.dtype)]

    return normalised_segments


def decode_wave_to_symbols(
    wave: np.ndarray,
    sr: int = SR,
    tone_dur: float = TONE_DUR,
    gap_dur: float = GAP_DUR,
    segmentation: Literal["hard", "energy"] = "hard",
) -> List[str]:
    """Decode an encoded waveform back into its symbol sequence.

    Parameters
    ----------
    wave:
        Input waveform (float array) produced by ``encode_symbols_to_wave``.
    sr:
        Sampling rate used during encoding.
    tone_dur:
        Duration of each tone in seconds.
    gap_dur:
        Duration of the silent gap between tones in seconds.
    segmentation:
        Strategy to segment tones. ``"hard"`` uses fixed strides, ``"energy"``
        detects high-energy segments.

    Returns
    -------
    list[str]
        Decoded symbol sequence.
    """

    if wave.size == 0:
        return []

    tone_samples = int(round(sr * tone_dur))
    if segmentation == "hard":
        gap_samples = int(round(sr * gap_dur))
        stride = tone_samples + gap_samples
        num_symbols = (
            (len(wave) + gap_samples) // stride if stride > 0 else len(wave) // tone_samples
        )
        num_symbols = max(num_symbols, 1)
        segments = []
        for idx in range(num_symbols):
            start = idx * stride
            end = start + tone_samples
            segment = wave[start:end]
            if segment.size < tone_samples:
                segment = np.pad(segment, (0, tone_samples - segment.size))
            segments.append(segment)
    elif segmentation == "energy":
        segments = _segment_by_energy(wave, sr, tone_samples)
    else:
        raise ValueError(f"Unsupported segmentation mode: {segmentation}")

    freqs = np.fft.rfftfreq(tone_samples, d=1.0 / sr)
    decoded: list[str] = []
    for segment in segments:
        rms = float(np.sqrt(np.mean(segment ** 2))) if segment.size else 0.0
        if rms < 1e-6:
            continue
        if segment.size != tone_samples:
            segment = _resize_segment(segment, tone_samples)
        spectrum = np.fft.rfft(segment)
        magnitude = np.abs(spectrum)
        if magnitude.size > 0:
            magnitude[0] = 0.0
        dominant_idx = int(np.argmax(magnitude))
        dominant_freq = freqs[dominant_idx]
        decoded.append(_nearest_symbol(dominant_freq))

    if not decoded and segmentation == "energy":
        return decode_wave_to_symbols(
            wave,
            sr=sr,
            tone_dur=tone_dur,
            gap_dur=gap_dur,
            segmentation="hard",
        )

    return decoded


def exact_match(pred_symbols: Iterable[str], gold_symbols: Iterable[str]) -> float:
    """Return ``1.0`` when predictions exactly match the gold sequence, else ``0.0``."""
    return 1.0 if list(pred_symbols) == list(gold_symbols) else 0.0


__all__ = [
    "decode_wave_to_symbols",
    "exact_match",
]

