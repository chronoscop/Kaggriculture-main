# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: data.dataset; see docs/action_bc_sources.md.
"""Replay batches shared by behavior cloning and data preparation."""

from __future__ import annotations
import json
import hashlib
from collections import OrderedDict
from pathlib import Path
import zipfile
import numpy as np
from route_rl.full_action.features import FEATURE_DIM

IGNORE_LABEL = -100
REPLAY_UNITS = 20
REPLAY_MARKET_SLOTS = 10
DATA_LOADING_CONTRACT = "streaming-npz-v1"
SHUFFLE_CONTRACT = "trajectory-window-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ReplayDataset:
    def __init__(self, cache: Path, split: str, processes: int, rank: int, window: int = 16) -> None:
        if window < 1:
            raise ValueError("shuffle window must be positive")
        self.cache, self.window = cache, window
        index = json.loads((cache / "index.json").read_text())
        rows = [row for row in index["episodes"] if row["split"] == split]
        if not rows:
            raise ValueError(f"empty replay split: {split}")
        padded_size = ((len(rows) + processes - 1) // processes) * processes
        rows = [rows[index % len(rows)] for index in range(padded_size)][rank::processes]
        self.rows = rows
        self._resident = OrderedDict()
        lengths = []
        self.decoded_bytes = 0
        for row in rows:
            # Read the small NPY headers inside the ZIP, without inflating arrays.
            headers = []
            with zipfile.ZipFile(cache / f"{row['key']}.npz") as archive:
                for name in ("features", "labels"):
                    with archive.open(f"{name}.npy") as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise ValueError(f"unsupported replay array header: {version}")
                        headers.append((shape, dtype))
                        self.decoded_bytes += int(np.prod(shape)) * dtype.itemsize
            features, labels = headers
            if len(features[0]) != 3 or features[0][1:] != (264, FEATURE_DIM):
                raise ValueError("wrong replay feature shape")
            if labels[0] != (features[0][0], REPLAY_UNITS + REPLAY_MARKET_SLOTS):
                raise ValueError("wrong replay label shape")
            if features[0][0] < 1 or row.get("samples", features[0][0]) != features[0][0]:
                raise ValueError("replay sample count differs from its index")
            lengths.append(features[0][0])
        self.offsets = np.cumsum([0, *lengths])
        self.size = int(self.offsets[-1])
        if not self.size:
            raise ValueError(f"empty replay split: {split}")

    @property
    def resident_trajectories(self) -> int:
        return len(self._resident)

    def _trajectory(self, episode: int) -> tuple[np.ndarray, np.ndarray]:
        if episode not in self._resident:
            with np.load(self.cache / f"{self.rows[episode]['key']}.npz", allow_pickle=False) as archive:
                self._resident[episode] = archive["features"], archive["labels"]
            while len(self._resident) > 2 * self.window:
                self._resident.popitem(last=False)
        self._resident.move_to_end(episode)
        return self._resident[episode]

    def _retain(self, episodes: set[int]) -> None:
        for episode in list(self._resident):
            if episode not in episodes:
                del self._resident[episode]

    def batch(self, indices: np.ndarray, size: int) -> dict[str, np.ndarray]:
        if not len(indices) or size < 1:
            raise ValueError("empty replay batch")
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError("replay row index out of bounds")
        selected = np.resize(indices, size)
        batch = {
            "features": np.zeros((size, 264, FEATURE_DIM), np.float32),
            "unit_action": np.full((size, REPLAY_UNITS), IGNORE_LABEL, np.int32),
            "market_action": np.full((size, REPLAY_MARKET_SLOTS), IGNORE_LABEL, np.int32),
        }
        episode = np.searchsorted(self.offsets[1:], selected, side="right")
        for game in np.unique(episode):
            features, labels = self._trajectory(int(game))
            positions = np.flatnonzero(episode == game)
            turns = selected[positions] - self.offsets[game]
            batch["features"][positions] = features[turns]
            batch["unit_action"][positions] = labels[turns, :REPLAY_UNITS]
            batch["market_action"][positions] = labels[turns, REPLAY_UNITS:]
        return batch

    def batches(self, size: int, rng: np.random.Generator | None = None):
        """Visit every row once; carry partial batches across trajectory windows."""
        if size < 1:
            raise ValueError("batch size must be positive")
        episodes = np.arange(len(self.rows))
        if rng is not None:
            rng.shuffle(episodes)
        carry = np.empty(0, np.int64)
        try:
            for start in range(0, len(episodes), self.window):
                current = episodes[start:start + self.window]
                indices = np.concatenate([np.arange(self.offsets[ep], self.offsets[ep + 1]) for ep in current])
                if rng is not None:
                    rng.shuffle(indices)
                if len(carry):
                    indices = np.concatenate((carry, indices))
                complete = len(indices) // size * size
                for offset in range(0, complete, size):
                    yield self.batch(indices[offset:offset + size], size), size
                    if offset == 0:
                        # The previous window's carry has now been copied/consumed.
                        self._retain(set(map(int, current)))
                carry = indices[complete:]
                self._retain(set(map(int, np.searchsorted(self.offsets[1:], carry, side="right"))))
            if len(carry):
                yield self.batch(carry, size), len(carry)
        finally:
            self._resident.clear()
