# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: training.metrics; see docs/action_bc_sources.md.
"""Host-side reduction of training metrics."""

import jax
import numpy as np


def aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    weights = np.asarray([float(np.asarray(r["sample_count"]).reshape(-1)[0]) for r in rows])
    total = max(float(weights.sum()), 1.0)
    return {
        name: float(np.sum([float(np.asarray(r[name]).reshape(-1)[0]) * w for r, w in zip(rows, weights)]) / total)
        for name in rows[0]
        if name != "sample_count"
    } | {"sample_count": total}


def first_local_replica(tree):
    return jax.tree.map(lambda value: np.asarray(value.addressable_shards[0].data[0]), tree)
