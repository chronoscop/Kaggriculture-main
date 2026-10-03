# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: training/sharding.py:put_replicated; see docs/action_bc_sources.md.
"""Explicit parameter replication for the BC updater."""
from collections.abc import Sequence
from typing import Any
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

def put_replicated(tree: Any, devices: Sequence[jax.Device]) -> Any:
    selected_devices = tuple(devices)
    device_count = len(selected_devices)
    if device_count <= 0:
        raise ValueError("at least one data-parallel device is required")
    axis_name = "data_replicas"
    mesh = Mesh(np.asarray(selected_devices, dtype=object), (axis_name,))
    sharding = NamedSharding(mesh, PartitionSpec(axis_name))

    def replicate(value: Any) -> jax.Array:
        host = np.asarray(jax.device_get(value))
        replicas = np.broadcast_to(host, (device_count, *host.shape)).copy()
        return jax.device_put(replicas, sharding)

    return jax.tree.map(replicate, tree)


