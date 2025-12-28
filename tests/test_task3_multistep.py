"""Tests for Task3 multi-step mod expressions."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jericho.task3 import (
    evaluate_mod_expression,
    count_mod_steps,
    target_symbols_for_task3,
    Task3ParseError,
)
from jericho.data.make_task3_manifest import (
    _multi_step_expression_tokens,
    build_task3_manifest,
)


class TestMultiStepParser:
    """Test multi-step mod expression parsing."""
    
    def test_single_step(self):
        """Single step: A % B"""
        tokens = ["1", "2", "%", "5"]
        result = evaluate_mod_expression(tokens)
        assert result == 12 % 5 == 2
    
    def test_two_step_left_associative(self):
        """Two steps: A % B % C = (A % B) % C"""
        # 17 % 5 % 3 = 2 % 3 = 2
        tokens = ["1", "7", "%", "5", "%", "3"]
        result = evaluate_mod_expression(tokens)
        assert result == (17 % 5) % 3 == 2
    
    def test_three_step(self):
        """Three steps: A % B % C % D"""
        # 100 % 7 % 5 % 3 = 2 % 5 % 3 = 2 % 3 = 2
        tokens = ["1", "0", "0", "%", "7", "%", "5", "%", "3"]
        result = evaluate_mod_expression(tokens)
        assert result == ((100 % 7) % 5) % 3 == 2
    
    def test_large_numbers(self):
        """Large numbers: 999 % 17 % 5"""
        tokens = ["9", "9", "9", "%", "1", "7", "%", "5"]
        result = evaluate_mod_expression(tokens)
        # 999 % 17 = 13, 13 % 5 = 3
        assert result == (999 % 17) % 5 == 3
    
    def test_count_mod_steps(self):
        """Count modulo operations."""
        assert count_mod_steps(["1", "%", "2"]) == 1
        assert count_mod_steps(["1", "%", "2", "%", "3"]) == 2
        assert count_mod_steps(["1", "%", "2", "%", "3", "%", "4"]) == 3
        assert count_mod_steps(["1", "2", "3"]) == 0
    
    def test_target_symbols_multistep(self):
        """target_symbols_for_task3 should work with multi-step."""
        # 17 % 5 % 3 = 2
        tokens = ["1", "7", "%", "5", "%", "3"]
        target = target_symbols_for_task3(tokens)
        assert target == ["2"]
    
    def test_zero_divisor_error(self):
        """Should raise error for zero divisor."""
        tokens = ["1", "0", "%", "0"]
        with pytest.raises(Task3ParseError, match="zero"):
            evaluate_mod_expression(tokens)
    
    def test_empty_operand_error(self):
        """Should raise error for empty operand."""
        tokens = ["%", "5"]
        with pytest.raises(Task3ParseError, match="Empty"):
            evaluate_mod_expression(tokens)
        
        tokens = ["5", "%"]
        with pytest.raises(Task3ParseError, match="Empty"):
            evaluate_mod_expression(tokens)


class TestMultiStepDataGeneration:
    """Test multi-step expression data generation."""
    
    def test_multi_step_expression_tokens(self):
        """Generate multi-step expression tokens."""
        tokens = _multi_step_expression_tokens([17, 5, 3])
        assert tokens == ["1", "7", "%", "5", "%", "3"]
        
        tokens = _multi_step_expression_tokens([100, 7])
        assert tokens == ["1", "0", "0", "%", "7"]
    
    def test_build_manifest_with_compose(self):
        """Build manifest including ood_compose split."""
        entries = build_task3_manifest(
            seed=42,
            split_sizes={
                "train": 10,
                "val": 5,
                "iid_test": 5,
                "ood_digits": 5,
                "ood_compose": 10,
                "ood_length": 5,
            },
            preset="tiny",
        )
        
        # Check splits exist
        splits = {e.split for e in entries}
        assert "ood_compose" in splits
        assert "ood_length" in splits
        
        # Check ood_compose has multi-step expressions
        compose_entries = [e for e in entries if e.split == "ood_compose"]
        assert len(compose_entries) == 10
        
        for entry in compose_entries:
            # Should have 2 '%' operators
            assert entry.symbols.count("%") == 2
            # Should be parseable
            result = evaluate_mod_expression(entry.symbols)
            assert 0 <= result < 100  # Result should be in valid range


class TestOracleBaseline:
    """Test oracle baseline for multi-step expressions."""
    
    def test_oracle_accuracy_single_step(self):
        """Oracle should achieve 100% on single-step."""
        test_cases = [
            (["1", "2", "%", "5"], ["2"]),
            (["9", "9", "%", "1", "0"], ["9"]),
            (["5", "%", "3"], ["2"]),
        ]
        
        for symbols, expected in test_cases:
            result = target_symbols_for_task3(symbols)
            assert result == expected, f"{symbols} -> {result} != {expected}"
    
    def test_oracle_accuracy_multi_step(self):
        """Oracle should achieve 100% on multi-step."""
        test_cases = [
            # 17 % 5 % 3 = 2 % 3 = 2
            (["1", "7", "%", "5", "%", "3"], ["2"]),
            # 100 % 17 % 5 = 15 % 5 = 0
            (["1", "0", "0", "%", "1", "7", "%", "5"], ["0"]),
            # 99 % 10 % 7 = 9 % 7 = 2
            (["9", "9", "%", "1", "0", "%", "7"], ["2"]),
        ]
        
        for symbols, expected in test_cases:
            result = target_symbols_for_task3(symbols)
            assert result == expected, f"{symbols} -> {result} != {expected}"

