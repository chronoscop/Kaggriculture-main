"""Load the OFFICIAL Kaggriculture engine -- and prove it is the pinned one.

The Rust port is certified against ``kaggle-environments==1.32.7``. Earlier
releases ship a different ``kaggriculture.py`` (1.32.6 and older price
CARROT/TOMATO/EGG scarcity on the pre-hinge curves), and a stale copy
anywhere on ``sys.path`` would silently answer a different question. So every
consumer goes through :func:`engine_module`, which checks the SHA-256 of

* the imported ``envs/kaggriculture/kaggriculture.py`` and its
  ``kaggriculture.json`` spec, and
* the runner files that decide how agents are called (``core.py``,
  ``agent.py``, ``utils.py``),

against the pins below, refuses an engine whose RNG has been patched (see
:mod:`kaggsim.forced`), and raises :class:`EngineMismatch` otherwise.

To point at a specific copy (e.g. an unpacked wheel) without installing it,
set ``KAGGSIM_OFFICIAL_PATH`` to the directory that CONTAINS the
``kaggle_environments`` package; it is put first on ``sys.path`` before the
import.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import os
import random
import sys

PINNED_VERSION = "1.32.7"
PINNED_ENGINE_SHA256 = (
    "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e")
PINNED_SPEC_SHA256 = (
    "a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867")
#: Runner files (relative to the kaggle_environments package).
PINNED_RUNNER_SHA256 = {
    "core.py":
        "0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e",
    "agent.py":
        "9b7682ce9921c8f34080a8be0f7b41598cc12ac7eb14d24e4b707883f25213b6",
    "utils.py":
        "537b627b11784d424147ef57ebb0369b039bf83c9f891e81f10486b1f552334b",
}
ENV_VAR = "KAGGSIM_OFFICIAL_PATH"

# Resolved configuration the official runner hands a 2-argument agent
# (kaggriculture.json defaults, key order as observed; `seed` is cleared).
DEFAULT_CONFIGURATION = {
    "seed": None, "episodeSteps": 720, "actTimeout": 1, "runTimeout": 1200,
    "boardSize": 10,
    "startingMoney": 3000, "maxMarketOrdersPerTurn": 10, "turnsPerDay": 24,
    "shedCapacity": 100, "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3, "townShopSellInterval": 4,
    "townCenterSellInterval": 24, "farmHandCostMult": 1, "marketParams": {},
}


class EngineMismatch(RuntimeError):
    """The importable kaggriculture engine is not the pinned release."""


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _prepend_override():
    path = os.environ.get(ENV_VAR)
    if not path:
        return None
    path = os.path.abspath(path)
    if not os.path.isdir(os.path.join(path, "kaggle_environments")):
        raise EngineMismatch(
            f"{ENV_VAR}={path} does not contain a kaggle_environments package")
    loaded = sys.modules.get("kaggle_environments")
    if loaded is not None:
        where = os.path.abspath(getattr(loaded, "__file__", "") or "")
        if not where.startswith(path):
            raise EngineMismatch(
                f"kaggle_environments was already imported from {where} "
                f"before {ENV_VAR} could take effect")
    if not sys.path or os.path.abspath(sys.path[0]) != path:
        sys.path.insert(0, path)
    return path


def _import_engine():
    """Import the interpreter module WITHOUT any checks (internal)."""
    _prepend_override()
    # Importing kaggle_environments prints load failures for unrelated
    # optional environments; keep them off stdout.
    with contextlib.redirect_stdout(io.StringIO()):
        importlib.import_module("kaggle_environments")
        from kaggle_environments.envs.kaggriculture import kaggriculture as mod
    return mod


def verify(mod=None, _allow_forced: bool = False) -> str:
    """Raise EngineMismatch unless the engine and runner files equal the
    pins and the engine's RNG is not patched. Returns the engine hash."""
    if mod is None:
        mod = _import_engine()
    got = sha256_file(mod.__file__)
    if got != PINNED_ENGINE_SHA256:
        raise EngineMismatch(
            f"official engine at {mod.__file__} has sha256 {got}; "
            f"expected kaggle-environments=={PINNED_VERSION} "
            f"({PINNED_ENGINE_SHA256}). Install the pinned release or set "
            f"{ENV_VAR}.")
    spec = os.path.join(os.path.dirname(mod.__file__), "kaggriculture.json")
    if os.path.exists(spec) and sha256_file(spec) != PINNED_SPEC_SHA256:
        raise EngineMismatch(f"environment spec {spec} differs from the pin")
    pkg = sys.modules.get("kaggle_environments")
    if pkg is not None and getattr(pkg, "__file__", None):
        root = os.path.dirname(pkg.__file__)
        for name, want in PINNED_RUNNER_SHA256.items():
            p = os.path.join(root, name)
            if not os.path.exists(p) or sha256_file(p) != want:
                raise EngineMismatch(
                    f"runner file {p} differs from kaggle-environments=="
                    f"{PINNED_VERSION}")
    if getattr(mod, "random", random) is not random and not _allow_forced:
        raise EngineMismatch(
            "the official engine's RNG is patched (kaggsim.forced is active); "
            "fidelity tools refuse to run on a modified game")
    return got


def engine_module():
    """Import the official ``kaggriculture`` module and verify it."""
    mod = _import_engine()
    verify(mod)
    return mod


def _make(seed: int, _allow_forced: bool = False, **configuration):
    mod = _import_engine()
    verify(mod, _allow_forced=_allow_forced)
    with contextlib.redirect_stdout(io.StringIO()):
        from kaggle_environments import make as _kmake
    cfg = {"seed": int(seed)}
    cfg.update(configuration)
    return _kmake("kaggriculture", configuration=cfg, info={"seed": int(seed)},
                  debug=False)


def make(seed: int, **configuration):
    """``kaggle_environments.make('kaggriculture', ...)`` on the pinned,
    unpatched engine."""
    return _make(seed, **configuration)


def _banks(env):
    last = env.steps[-1]
    return tuple(None if s.get("reward") is None else float(s["reward"])
                 for s in last)


def _run(agent0, agent1, seed, quiet, _allow_forced):
    env = _make(seed, _allow_forced=_allow_forced, actTimeout=60,
                runTimeout=100000)
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet \
        else contextlib.nullcontext()
    with ctx:
        env.run([agent0, agent1])
    return _banks(env), env


def run_agents(agent0, agent1, seed: int, quiet: bool = True):
    """Play two agent callables on the official runner (``env.run``).

    Returns ``(final banks, env)``. A seat whose agent errored has bank
    ``None`` (see ``env.steps[-1][seat]["status"]``). Agent stdout is
    suppressed when ``quiet``. The per-turn timeout is relaxed (60 s) so a
    slow development machine does not turn into agent errors.
    """
    return _run(agent0, agent1, seed, quiet, _allow_forced=False)
