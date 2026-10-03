# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: training.global_update; see docs/action_bc_sources.md.
"""Keep PPO parameters/Adam arrays global between microsteps, avoiding pmap rewrapping."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


class GlobalUpdate:
    def __init__(self, step: Callable, devices: Sequence[jax.Device]) -> None:
        self.mesh = Mesh(np.asarray(tuple(devices), dtype=object), ("data",))
        self.functions = {emit: self._compile(step, emit) for emit in (False, True)}

    def _compile(self, step: Callable, emit: bool) -> Any:
        def per_device(*trees: Any) -> Any:
            local = jax.tree.map(lambda value: value[0], trees)
            result = step(*local, emit)
            return jax.tree.map(lambda value: value[None], result)

        return jax.jit(
            jax.shard_map(per_device, mesh=self.mesh, in_specs=P("data"), out_specs=P("data"), check_vma=False)
        )

    def __call__(self, *inputs: Any) -> Any:
        return self.functions[inputs[-1]](*inputs[:-1])

    def lower(self, *inputs: Any) -> Any:
        return self.functions[inputs[-1]].lower(*inputs[:-1])

    def to_global(self, tree: Any) -> Any:
        return multihost_utils.host_local_array_to_global_array(tree, self.mesh, P("data"))

    def to_local(self, tree: Any) -> Any:
        return multihost_utils.global_array_to_host_local_array(tree, self.mesh, P("data"))

    @staticmethod
    def first_local_replica(tree: Any) -> Any:
        def first(value: jax.Array) -> jax.Array:
            shard = value.addressable_shards[0].data
            if shard.shape[0] != 1:
                raise ValueError("expected one leading replica per device")
            return shard.reshape(shard.shape[1:])

        return jax.tree.map(first, tree)
