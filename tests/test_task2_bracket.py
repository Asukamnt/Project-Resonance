"""Tests for Task2 bracket validity utilities."""

from __future__ import annotations

import random

import pytest

from jericho.task2 import (
    BRACKET_OPEN,
    BRACKET_CLOSE,
    VALID_SYMBOL,
    INVALID_SYMBOL,
    is_balanced,
    target_symbol_for_task2,
    generate_balanced_brackets,
    generate_unbalanced_brackets,
)


class TestIsBalanced:
    """Tests for is_balanced function."""

    def test_empty_is_balanced(self):
        assert is_balanced([]) is True

    def test_simple_balanced(self):
        assert is_balanced(["(", ")"]) is True

    def test_nested_balanced(self):
        assert is_balanced(["(", "(", ")", ")"]) is True

    def test_sequential_balanced(self):
        assert is_balanced(["(", ")", "(", ")"]) is True

    def test_complex_balanced(self):
        assert is_balanced(["(", "(", ")", "(", ")", ")"]) is True

    def test_single_open_unbalanced(self):
        assert is_balanced(["("]) is False

    def test_single_close_unbalanced(self):
        assert is_balanced([")"]) is False

    def test_wrong_order_unbalanced(self):
        assert is_balanced([")", "("]) is False

    def test_extra_open_unbalanced(self):
        assert is_balanced(["(", "(", ")"]) is False

    def test_extra_close_unbalanced(self):
        assert is_balanced(["(", ")", ")"]) is False

    def test_prefix_violation(self):
        assert is_balanced([")", "(", "(", ")"]) is False


class TestTargetSymbol:
    """Tests for target_symbol_for_task2 function."""

    def test_balanced_returns_valid(self):
        assert target_symbol_for_task2(["(", ")"]) == VALID_SYMBOL

    def test_unbalanced_returns_invalid(self):
        assert target_symbol_for_task2(["("]) == INVALID_SYMBOL

    def test_empty_returns_valid(self):
        assert target_symbol_for_task2([]) == VALID_SYMBOL


class TestGenerateBalanced:
    """Tests for generate_balanced_brackets function."""

    def test_length_2(self):
        brackets = generate_balanced_brackets(2, random.Random(42))
        assert len(brackets) == 2
        assert is_balanced(brackets) is True

    def test_length_4(self):
        brackets = generate_balanced_brackets(4, random.Random(42))
        assert len(brackets) == 4
        assert is_balanced(brackets) is True

    def test_length_10(self):
        brackets = generate_balanced_brackets(10, random.Random(42))
        assert len(brackets) == 10
        assert is_balanced(brackets) is True

    def test_reproducible(self):
        b1 = generate_balanced_brackets(8, random.Random(123))
        b2 = generate_balanced_brackets(8, random.Random(123))
        assert b1 == b2

    def test_odd_length_raises(self):
        with pytest.raises(ValueError):
            generate_balanced_brackets(3)

    def test_length_1_raises(self):
        with pytest.raises(ValueError):
            generate_balanced_brackets(1)


class TestGenerateUnbalanced:
    """Tests for generate_unbalanced_brackets function."""

    def test_length_1(self):
        brackets = generate_unbalanced_brackets(1, random.Random(42))
        assert len(brackets) == 1
        assert is_balanced(brackets) is False

    def test_length_3(self):
        brackets = generate_unbalanced_brackets(3, random.Random(42))
        assert len(brackets) == 3
        assert is_balanced(brackets) is False

    def test_length_5(self):
        brackets = generate_unbalanced_brackets(5, random.Random(42))
        assert len(brackets) == 5
        assert is_balanced(brackets) is False

    def test_reproducible(self):
        b1 = generate_unbalanced_brackets(5, random.Random(456))
        b2 = generate_unbalanced_brackets(5, random.Random(456))
        assert b1 == b2

    def test_length_0_raises(self):
        with pytest.raises(ValueError):
            generate_unbalanced_brackets(0)

