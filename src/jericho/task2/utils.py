"""Task2 bracket validity utilities."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from ..symbols import encode_symbols_to_wave, GAP_DUR, SR, TONE_DUR

# Task2 symbols
BRACKET_OPEN = "("
BRACKET_CLOSE = ")"
VALID_SYMBOL = "V"
INVALID_SYMBOL = "X"

TASK2_INPUT_SYMBOLS = [BRACKET_OPEN, BRACKET_CLOSE]
TASK2_OUTPUT_SYMBOLS = [VALID_SYMBOL, INVALID_SYMBOL]


def is_balanced(symbols: Sequence[str]) -> bool:
    """Check if a bracket sequence is balanced.

    Parameters
    ----------
    symbols:
        Sequence of bracket symbols (only '(' and ')' are considered).

    Returns
    -------
    bool
        True if balanced, False otherwise.
    """
    depth = 0
    for s in symbols:
        if s == BRACKET_OPEN:
            depth += 1
        elif s == BRACKET_CLOSE:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def target_symbol_for_task2(symbols: Sequence[str]) -> str:
    """Return the target symbol ('V' or 'X') for a bracket sequence.

    Parameters
    ----------
    symbols:
        Input bracket sequence.

    Returns
    -------
    str
        'V' if balanced, 'X' if unbalanced.
    """
    return VALID_SYMBOL if is_balanced(symbols) else INVALID_SYMBOL


def generate_balanced_brackets(
    length: int,
    rng: random.Random | None = None,
) -> list[str]:
    """Generate a random balanced bracket sequence of given length.

    Parameters
    ----------
    length:
        Total number of brackets. Must be even and >= 2.
    rng:
        Optional random generator for reproducibility.

    Returns
    -------
    list[str]
        A balanced bracket sequence.

    Raises
    ------
    ValueError
        If length is odd or < 2.
    """
    if length < 2 or length % 2 != 0:
        raise ValueError(f"Length must be even and >= 2, got {length}")

    if rng is None:
        rng = random.Random()

    # Generate using Catalan number method: random shuffle of n '(' and n ')'
    # then fix any prefix violations
    n = length // 2
    brackets: list[str] = [BRACKET_OPEN] * n + [BRACKET_CLOSE] * n
    rng.shuffle(brackets)

    # Fix to ensure balance using cyclic rotation approach
    # Find the first position where we can start to get a valid sequence
    result = _fix_to_balanced(brackets)
    return result


def _fix_to_balanced(brackets: list[str]) -> list[str]:
    """Fix a shuffled bracket sequence to be balanced via cyclic rotation."""
    n = len(brackets)
    if n == 0:
        return []

    # Find the rotation that makes it balanced
    # A sequence is balanced if at every prefix, opens >= closes
    best_start = 0
    min_depth = 0
    depth = 0

    for i, b in enumerate(brackets):
        if b == BRACKET_OPEN:
            depth += 1
        else:
            depth -= 1
        if depth < min_depth:
            min_depth = depth
            best_start = i + 1

    # Rotate
    return brackets[best_start:] + brackets[:best_start]


def generate_unbalanced_brackets(
    length: int,
    rng: random.Random | None = None,
) -> list[str]:
    """Generate a random unbalanced bracket sequence of given length.

    Parameters
    ----------
    length:
        Total number of brackets. Must be >= 1.
    rng:
        Optional random generator for reproducibility.

    Returns
    -------
    list[str]
        An unbalanced bracket sequence.
    """
    if length < 1:
        raise ValueError(f"Length must be >= 1, got {length}")

    if rng is None:
        rng = random.Random()

    # Strategy: generate random brackets until we get an unbalanced one
    max_attempts = 100
    for _ in range(max_attempts):
        brackets = [rng.choice(TASK2_INPUT_SYMBOLS) for _ in range(length)]
        if not is_balanced(brackets):
            return brackets

    # Fallback: force unbalanced by making all opens or all closes
    if length == 1:
        return [BRACKET_OPEN]  # Single open is unbalanced
    # For longer sequences, use mismatched counts
    return [BRACKET_OPEN] * (length // 2 + 1) + [BRACKET_CLOSE] * (length - length // 2 - 1)


def synthesise_task2_target_wave(
    symbols: Sequence[str],
    *,
    target_length_samples: int | None = None,
    sr: int = SR,
    tone_dur: float = TONE_DUR,
    gap_dur: float = GAP_DUR,
    rng: np.random.Generator | None = None,
    fixed_phase: float | None = None,
) -> np.ndarray:
    """Convert Task2 target symbol (V or X) into audio, optionally padded.
    
    Parameters
    ----------
    symbols:
        Input bracket sequence.
    target_length_samples:
        If provided, pad or truncate to this length.
    sr, tone_dur, gap_dur:
        Audio synthesis parameters.
    rng:
        Random generator for phase variation.
    fixed_phase:
        If provided, use fixed phase for reproducibility.
    
    Returns
    -------
    np.ndarray
        Audio waveform (float32).
    """
    target_symbol = target_symbol_for_task2(symbols)
    wave = encode_symbols_to_wave(
        [target_symbol],
        sr=sr,
        tone_dur=tone_dur,
        gap_dur=gap_dur,
        rng=rng,
        fixed_phase=fixed_phase,
    ).astype(np.float32, copy=False)
    
    if target_length_samples is None:
        return wave
    if target_length_samples <= wave.size:
        return wave[:target_length_samples]
    pad = np.zeros(target_length_samples - wave.size, dtype=np.float32)
    return np.concatenate([wave, pad])


__all__ = [
    "BRACKET_OPEN",
    "BRACKET_CLOSE",
    "VALID_SYMBOL",
    "INVALID_SYMBOL",
    "TASK2_INPUT_SYMBOLS",
    "TASK2_OUTPUT_SYMBOLS",
    "is_balanced",
    "target_symbol_for_task2",
    "generate_balanced_brackets",
    "generate_unbalanced_brackets",
    "synthesise_task2_target_wave",
]

