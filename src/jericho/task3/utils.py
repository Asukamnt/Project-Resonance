"""Utility functions for Task3 (Arithmetic Mod) data generation."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..symbols import GAP_DUR, SR, TONE_DUR, encode_symbols_to_wave

MOD_OPERATOR = "%"


class Task3ParseError(ValueError):
    """Raised when Task3 symbol sequences cannot be parsed."""


def digits_to_int(digits: Sequence[str]) -> int:
    """Convert a sequence of digit strings into an integer."""
    if not digits:
        raise Task3ParseError("digit sequence empty")
    if any(d not in "0123456789" for d in digits):
        raise Task3ParseError(f"Non-digit token in digits: {digits}")
    return int("".join(digits))


def int_to_digits(value: int) -> List[str]:
    """Convert a non-negative integer to digit tokens."""
    if value < 0:
        raise Task3ParseError("Negative values not supported for Task3")
    return list(str(value))


def parse_mod_expression(symbols: Sequence[str]) -> Tuple[int, int]:
    """Parse tokens representing A % B into integer operands."""
    if MOD_OPERATOR not in symbols:
        raise Task3ParseError("Modulo operator '%' not found")
    if symbols.count(MOD_OPERATOR) != 1:
        raise Task3ParseError("Only single-step modulo expressions supported")
    op_index = symbols.index(MOD_OPERATOR)
    left_digits = symbols[:op_index]
    right_digits = symbols[op_index + 1 :]
    if not left_digits or not right_digits:
        raise Task3ParseError("Incomplete modulo expression")
    left = digits_to_int(left_digits)
    right = digits_to_int(right_digits)
    if right == 0:
        raise Task3ParseError("Modulo divisor cannot be zero")
    return left, right


def target_symbols_for_task3(symbols: Sequence[str]) -> List[str]:
    """Return digit tokens representing the remainder for an expression."""
    dividend, divisor = parse_mod_expression(symbols)
    remainder = dividend % divisor
    return int_to_digits(remainder)


def synthesise_task3_target_wave(
    symbols: Sequence[str],
    *,
    target_length_samples: int | None = None,
    sr: int = SR,
    tone_dur: float = TONE_DUR,
    gap_dur: float = GAP_DUR,
    rng: np.random.Generator | None = None,
    fixed_phase: float | None = None,
) -> np.ndarray:
    """Convert Task3 remainder tokens into audio, optionally padded to length.

    ``fixed_phase`` can be used to stabilise targets by removing phase randomness.
    """
    target_tokens = target_symbols_for_task3(symbols)
    wave = encode_symbols_to_wave(
        target_tokens,
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
    "MOD_OPERATOR",
    "Task3ParseError",
    "digits_to_int",
    "int_to_digits",
    "parse_mod_expression",
    "target_symbols_for_task3",
    "synthesise_task3_target_wave",
]

