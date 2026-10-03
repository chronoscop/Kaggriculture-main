"""Explicit replay compatibility with the pinned Kaggriculture label resolver."""
from __future__ import annotations

import hashlib
from pathlib import Path

RESOLVER_VERSION = "1.32.7"
COMPATIBLE_REPLAY_VERSIONS = ("1.32.7", "1.33.0")
# Both official wheels were compared byte for byte on 2026-10-03. All six
# Kaggriculture files match, including these rules and configuration files.
RULE_SHA256 = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
SPEC_SHA256 = "a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867"
LABEL_CONTRACT = "owned-full-action-replay-labels-v3"


def compatibility_receipt() -> dict:
    return {"label_contract": LABEL_CONTRACT, "resolver_version": RESOLVER_VERSION,
            "compatible_replay_versions": list(COMPATIBLE_REPLAY_VERSIONS),
            "rule_sha256": RULE_SHA256, "spec_sha256": SPEC_SHA256,
            "time_encoding": "explicit step or actual day * 24 + hour"}


def require_pinned_rules() -> None:
    from .full_action.legality import engine, require_expected_engine

    require_expected_engine()
    path = Path(engine.__file__)
    for file, expected in ((path, RULE_SHA256), (path.with_suffix(".json"), SPEC_SHA256)):
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"installed Kaggriculture rules differ from verified resolver: {file}")


def replay_seed(replay: dict) -> int | None:
    configuration = replay.get("configuration", {}).get("seed")
    recorded = replay.get("info", {}).get("seed")
    if configuration is not None and recorded is not None and int(configuration) != int(recorded):
        raise ValueError("replay configuration and recorded runtime seed disagree")
    value = configuration if configuration is not None else recorded
    return None if value is None else int(value)
