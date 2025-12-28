from __future__ import annotations

from pathlib import Path


def test_next_step_doc_exists() -> None:
    """Docs are part of the deliverable: ensure next_step planning doc exists."""
    path = Path("docs/next_step.md")
    assert path.exists(), "Expected docs/next_step.md to exist"

    text = path.read_text(encoding="utf-8")
    for keyword in (
        "Week5",
        "Week6",
        "Week7",
        "Week8",
        "最小证据任务",
        "MET",
        "负对照",
        "消融",
        "一键",
    ):
        assert keyword in text


