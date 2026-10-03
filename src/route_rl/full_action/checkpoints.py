"""Project checkpoint format shared by full-action BC and candidate inference."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pickle

import jax
import numpy as np

POLICY_CONTRACT = "route-rl-full-action-policy-v1"


def policy_hash(params: dict) -> str:
    digest = hashlib.sha256()
    leaves, tree = jax.tree_util.tree_flatten(jax.device_get(params))
    digest.update(str(tree).encode())
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def atomic_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as destination:
        pickle.dump(payload, destination, protocol=pickle.HIGHEST_PROTOCOL)
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)


def save_params_payload(path: Path, params: dict, model_config: dict, completed_updates: int) -> None:
    atomic_pickle(path, {
        "contract": POLICY_CONTRACT,
        "params": jax.tree.map(lambda value: np.asarray(jax.device_get(value)), params),
        "model_config": model_config,
        "training_backend": "jax",
        "rl_completed_updates": completed_updates,
        "policy_sha256": policy_hash(params),
    })


def load_training_source(path: Path) -> dict:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("contract") != POLICY_CONTRACT:
        raise ValueError("checkpoint is not a project full-action policy; initialize this branch separately")
    params = payload["params"]
    if payload.get("policy_sha256") != policy_hash(params):
        raise ValueError("checkpoint parameter checksum mismatch")
    return {**payload, "state": {"params": params}}
