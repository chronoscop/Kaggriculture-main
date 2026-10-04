"""Owned batched Rust simulation; Python remains an explicit differential backend."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..full_action.inference import dummy_fixed_batch

BACKENDS = ("official-python", "rust-batch")
_MODULE = None
_LOADED_IDENTITY = None


def _source_hash(crate: Path) -> str:
    paths = [crate / "Cargo.toml", crate / "Cargo.lock", crate / "pyproject.toml", *sorted((crate / "src").glob("*.rs"))]
    entries = {str(path.relative_to(crate)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def native_module():
    global _MODULE, _LOADED_IDENTITY
    directory = Path(__file__).with_name("_native")
    receipt_path = directory / "build.json"
    if not receipt_path.exists():
        raise RuntimeError("Rust batch backend is not built; run python tools/build_action_native.py --jobs 2")
    receipt = json.loads(receipt_path.read_text())
    if Path(receipt["binary"]).name != receipt["binary"]:
        raise RuntimeError("Rust batch receipt must name a local binary")
    binary = directory / receipt["binary"]
    if not binary.is_file() or hashlib.sha256(binary.read_bytes()).hexdigest() != receipt["binary_sha256"]:
        raise RuntimeError("Rust batch backend binary does not match its build receipt")
    crate = Path(__file__).resolve().parents[3] / "native/action_engine"
    if (crate / "Cargo.toml").is_file() and _source_hash(crate) != receipt["source_sha256"]:
        raise RuntimeError("Rust batch source changed after building; rebuild with tools/build_action_native.py")
    identity = (receipt["source_sha256"], receipt["binary_sha256"])
    if _MODULE is not None:
        if identity != _LOADED_IDENTITY:
            raise RuntimeError("Rust batch binary changed while loaded; restart the process")
        return _MODULE
    spec = importlib.util.spec_from_file_location("route_rl_action_engine", binary)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load owned Rust action engine")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.OWNED_SOURCE_SHA256 != receipt["source_sha256"]:
        raise RuntimeError("Rust batch backend source identity does not match its build receipt")
    _MODULE = module
    _LOADED_IDENTITY = identity
    return module


def backend_identity(name: str) -> dict[str, Any]:
    if name not in BACKENDS:
        raise ValueError(f"unsupported collection backend: {name}")
    if name == "official-python":
        from ..replay_rules import require_pinned_rules
        require_pinned_rules()
        return {"backend": name, "contract": "owned-official-python-collection-v1", "resolver_version": "1.32.7"}
    module = native_module()
    receipt = json.loads(Path(__file__).with_name("_native").joinpath("build.json").read_text())
    return {"backend": name, "contract": "owned-rust-batch-collection-v1", "resolver_version": "1.32.7",
            "module": "route_rl_action_engine", "execution_base": module.OWNED_EXECUTION_CONTRACT,
            "source_sha256": receipt["source_sha256"], "binary_sha256": receipt["binary_sha256"]}


class NativePrefixResolver:
    """The same resolver interface consumed by the shared economic sampler."""
    def __init__(self, native: Any) -> None:
        self.native = native
        self.unit_count, self.player = int(native.unit_count), int(native.player)
        self._state = None

    @property
    def state(self):
        if self._state is None:
            self._state = self.native.observation()
        return self._state

    @property
    def farm(self):
        return self.state["farms"][self.player]

    @property
    def farms(self):
        return self.state["farms"]

    @property
    def private(self):
        return self.state["private"]

    @property
    def market(self):
        return self.state["market"]

    @property
    def day(self):
        return int(self.state["day"])

    def unit_support(self, index):
        return np.asarray(self.native.unit_support(index), np.bool_)

    def apply_unit(self, index, identity):
        self.native.apply_unit(index, int(identity))
        self._state = None

    def market_support(self):
        return np.asarray(self.native.market_support(), np.bool_)

    def apply_market(self, identity):
        self.native.apply_market(int(identity))
        if identity:
            self._state = None


class RustBatchBackend:
    def __init__(self, seeds: list[int]) -> None:
        self.identity = backend_identity("rust-batch")
        self.environment = native_module().RustBatchEnv(seeds)
        self.seeds = list(seeds)
        self.buffers = {name: np.repeat(array, len(seeds)*2, axis=0) for name, array in dummy_fixed_batch().items()}
        self.step = 0

    def observe(self) -> list[list[dict]]:
        return self.environment.observe()

    def encode(self) -> dict[str, np.ndarray]:
        b = self.buffers
        self.environment.write_owned_features(b["features"], b["memory_features"], b["coordinates"],
            b["spatial_mask"], b["rope_groups"], b["token_mask"], b["unit_indices"], b["market_indices"])
        return b

    def resolver(self, game: int, player: int) -> NativePrefixResolver:
        return NativePrefixResolver(self.environment.owned_prefix(game, player))

    def advance(self, selected: list[Any], *, external: bool = False) -> None:
        if external:
            done = self.environment.step_owned_actions([action.action for action in selected])
        else:
            done = self.environment.step_owned_ids(np.stack([action.unit_action for action in selected]),
                                                   np.stack([action.market_action for action in selected]))
        expected = self.step == 718
        if len(done) != len(self.seeds) or any(bool(value) != expected for value in done):
            raise ValueError("Rust rollout horizon does not match the complete official game")
        self.step += 1

    def rewards(self) -> list[list[float]]:
        if self.step != 719:
            raise ValueError("terminal rewards require a complete Rust rollout")
        return self.environment.rewards()
