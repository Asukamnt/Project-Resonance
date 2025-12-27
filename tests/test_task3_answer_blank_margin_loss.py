import torch

from jericho.pipelines.task3_mod_audio import compute_answer_blank_margin_loss


def test_answer_blank_margin_loss_backward_and_effect():
    batch, frames, vocab = 2, 3, 5
    blank_id = 0
    digit_ids = [1, 2, 3]
    logits = torch.zeros(batch, frames, vocab, requires_grad=True)
    with torch.no_grad():
        logits[..., blank_id] = 2.0
        logits[..., digit_ids] = 0.0
    tone_mask = torch.ones(batch, frames, dtype=torch.bool)

    loss = compute_answer_blank_margin_loss(
        logits,
        tone_mask,
        digit_ids=digit_ids,
        blank_id=blank_id,
        margin=1.0,
    )
    assert loss.item() > 0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    # Increasing weight should scale total loss upward
    loss_weighted = 2.0 * loss
    assert loss_weighted.item() > loss.item()

