"""Tests for RemainderHead and compute_remainder_logits with hidden states."""

from __future__ import annotations

import pytest
import torch

from jericho.pipelines.task3_mod_audio import (
    RemainderHead,
    compute_remainder_logits,
)


class TestRemainderHead:
    """Tests for the new attention-based RemainderHead."""

    def test_remainder_head_forward_shape(self):
        """RemainderHead should output correct shape."""
        d_model = 64
        num_digits = 10
        batch_size = 4
        seq_len = 20

        head = RemainderHead(
            d_model=d_model,
            num_digits=num_digits,
            hidden_dim=32,
            num_attn_heads=2,
            dropout=0.0,
        )

        hidden_states = torch.randn(batch_size, seq_len, d_model)
        expression_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = head(hidden_states, expression_mask)

        assert output.shape == (batch_size, num_digits)

    def test_remainder_head_with_partial_mask(self):
        """RemainderHead should work with partial expression masks."""
        d_model = 64
        num_digits = 10
        batch_size = 2
        seq_len = 20

        head = RemainderHead(d_model=d_model, num_digits=num_digits)

        hidden_states = torch.randn(batch_size, seq_len, d_model)
        expression_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        # Only first 10 frames are expression
        expression_mask[:, :10] = True

        output = head(hidden_states, expression_mask)

        assert output.shape == (batch_size, num_digits)
        # Output should be finite
        assert torch.isfinite(output).all()

    def test_remainder_head_gradient_flow(self):
        """Verify gradients flow through RemainderHead."""
        d_model = 64
        num_digits = 10
        batch_size = 2
        seq_len = 20

        head = RemainderHead(d_model=d_model, num_digits=num_digits)

        hidden_states = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        expression_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = head(hidden_states, expression_mask)
        loss = output.sum()
        loss.backward()

        # Gradient should flow to hidden_states
        assert hidden_states.grad is not None
        assert hidden_states.grad.abs().sum() > 0


class TestComputeRemainderLogitsAttnHidden:
    """Tests for compute_remainder_logits with attn_hidden head."""

    def test_attn_hidden_uses_hidden_states(self):
        """attn_hidden head should use hidden_states and remainder_head_module."""
        d_model = 64
        num_digits = 10
        batch_size = 2
        seq_len = 20
        vocab_size = 12

        remainder_head_module = RemainderHead(
            d_model=d_model,
            num_digits=num_digits,
            hidden_dim=32,
        )

        symbol_logits = torch.randn(batch_size, seq_len, vocab_size)
        expression_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        output = compute_remainder_logits(
            symbol_logits,
            expression_mask,
            digit_ids=list(range(1, 11)),
            percent_id=11,
            blank_id=0,
            head="attn_hidden",
            remainder_gru=None,
            remainder_linear=None,
            hidden_states=hidden_states,
            remainder_head_module=remainder_head_module,
        )

        assert output.shape == (batch_size, num_digits)

    def test_attn_hidden_fallback_to_pooled(self):
        """attn_hidden should fallback to pooled if module not provided."""
        batch_size = 2
        seq_len = 20
        vocab_size = 12
        digit_ids = list(range(1, 11))

        symbol_logits = torch.randn(batch_size, seq_len, vocab_size)
        expression_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # No hidden_states or remainder_head_module provided
        output = compute_remainder_logits(
            symbol_logits,
            expression_mask,
            digit_ids=digit_ids,
            percent_id=11,
            blank_id=0,
            head="attn_hidden",
            remainder_gru=None,
            remainder_linear=None,
            hidden_states=None,
            remainder_head_module=None,
        )

        # Should fallback to pooled, output shape is num_digits
        assert output.shape == (batch_size, len(digit_ids))


class TestComputeRemainderLogitsGradients:
    """Tests for gradient flow in compute_remainder_logits (no detach)."""

    def test_gru_token_gradient_flow(self):
        """Verify gradients flow through gru_token head (no detach)."""
        batch_size = 2
        seq_len = 20
        vocab_size = 12
        num_digits = 10

        from torch.nn.utils.rnn import pack_padded_sequence

        digit_ids = list(range(1, 11))
        subset_dim = len(set(digit_ids + [11, 0]))

        remainder_gru = torch.nn.GRU(
            input_size=subset_dim,
            hidden_size=32,
            batch_first=True,
        )
        remainder_linear = torch.nn.Linear(32, num_digits)

        symbol_logits = torch.randn(batch_size, seq_len, vocab_size)
        expression_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        token_probs = torch.randn(batch_size, 5, vocab_size, requires_grad=True)
        token_mask = torch.ones(batch_size, 5, dtype=torch.bool)

        output = compute_remainder_logits(
            symbol_logits,
            expression_mask,
            digit_ids=digit_ids,
            percent_id=11,
            blank_id=0,
            head="gru_token",
            remainder_gru=remainder_gru,
            remainder_linear=remainder_linear,
            token_probs=token_probs,
            token_mask=token_mask,
        )

        loss = output.sum()
        loss.backward()

        # Gradient should flow to token_probs (no detach)
        assert token_probs.grad is not None
        assert token_probs.grad.abs().sum() > 0

    def test_pooled_gradient_flow(self):
        """Verify gradients flow through pooled head (no detach)."""
        batch_size = 2
        seq_len = 20
        vocab_size = 12
        digit_ids = list(range(1, 11))

        symbol_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        expression_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = compute_remainder_logits(
            symbol_logits,
            expression_mask,
            digit_ids=digit_ids,
            percent_id=11,
            blank_id=0,
            head="pooled",
            remainder_gru=None,
            remainder_linear=None,
        )

        loss = output.sum()
        loss.backward()

        # Gradient should flow to symbol_logits (no detach)
        assert symbol_logits.grad is not None
        assert symbol_logits.grad.abs().sum() > 0

