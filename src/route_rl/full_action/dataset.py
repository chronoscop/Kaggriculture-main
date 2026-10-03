# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: data.dataset; see docs/action_bc_sources.md.
"""Replay batches shared by behavior cloning and data preparation."""

from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
from route_rl.full_action.features import FEATURE_DIM

IGNORE_LABEL = -100
REPLAY_UNITS = 20
REPLAY_MARKET_SLOTS = 10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ReplayDataset:
    def __init__(self, cache: Path, split: str, processes: int, rank: int) -> None:
        index = json.loads((cache / "index.json").read_text())
        rows = [row for row in index["episodes"] if row["split"] == split]
        if not rows:
            raise ValueError(f"empty replay split: {split}")
        padded_size = ((len(rows) + processes - 1) // processes) * processes
        rows = [rows[index % len(rows)] for index in range(padded_size)][rank::processes]
        self.features, self.labels = [], []
        for row in rows:
            with np.load(cache / f"{row['key']}.npz", allow_pickle=False) as archive:
                self.features.append(archive["features"])
                self.labels.append(archive["labels"])
        self.offsets = np.cumsum([0, *[len(values) for values in self.features]])
        self.size = int(self.offsets[-1])
        if not self.size:
            raise ValueError(f"empty replay split: {split}")

    def batch(self, indices: np.ndarray, size: int) -> dict[str, np.ndarray]:
        if not len(indices):
            raise ValueError("empty replay batch")
        selected = np.resize(indices, size)
        batch = {
            "features": np.zeros((size, 264, FEATURE_DIM), np.float32),
            "unit_action": np.full((size, REPLAY_UNITS), IGNORE_LABEL, np.int32),
            "market_action": np.full((size, REPLAY_MARKET_SLOTS), IGNORE_LABEL, np.int32),
        }
        episode = np.searchsorted(self.offsets[1:], selected, side="right")
        for game in np.unique(episode):
            positions = np.flatnonzero(episode == game)
            turns = selected[positions] - self.offsets[game]
            batch["features"][positions] = self.features[game][turns]
            batch["unit_action"][positions] = self.labels[game][turns, :REPLAY_UNITS]
            batch["market_action"][positions] = self.labels[game][turns, REPLAY_UNITS:]
        return batch
