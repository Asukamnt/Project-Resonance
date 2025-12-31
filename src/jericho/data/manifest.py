"""Manifest utilities for Jericho datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import json


@dataclass
class ManifestEntry:
    """Container for a single dataset sample description."""

    split: str
    symbols: List[str]
    length: int
    difficulty_tag: str
    example_id: str
    seed: int
    sequence_seed: int
    # Optional target symbols (used by Phase2 optical manifests and some tasks)
    target_symbols: Optional[List[str]] = None
    # Optional task name (used by Phase2 optical manifests)
    task: Optional[str] = None

    def to_json(self) -> str:
        """Serialise the entry as a JSON string."""
        payload = asdict(self)
        # Keep manifest format stable: omit optional fields when not present.
        payload = {k: v for k, v in payload.items() if v is not None}
        return json.dumps(payload, ensure_ascii=False)


def write_manifest(entries: Iterable[ManifestEntry], path: str | Path) -> None:
    """Write manifest entries to disk as JSON Lines."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_json())
            f.write("\n")


def read_manifest(path: str | Path, split: Optional[str] = None) -> List[ManifestEntry]:
    """Load manifest entries from disk.

    Parameters
    ----------
    path:
        JSON Lines manifest path.
    split:
        Optional split filter. When provided, only entries matching the split are returned.
    """

    entries: list[ManifestEntry] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            entry = ManifestEntry(**payload)
            if split is None or entry.split == split:
                entries.append(entry)
    return entries


__all__ = ["ManifestEntry", "read_manifest", "write_manifest"]

