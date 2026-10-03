# Adapted from msdsm/kaggriculture-solution, commit 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d.
# Source: training/bc.py (project entry and supported execution scope adapted); see docs/action_bc_sources.md.
"""Continue the initial actor on public and selected search-selfplay labels only."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from route_rl.full_action.dataset import IGNORE_LABEL, ReplayDataset, file_sha256
from .checkpoints import load_training_source as host_checkpoint
from route_rl.full_action.metrics import aggregate, first_local_replica
from route_rl.full_action.bc_objective import make_steps
from route_rl.full_action.checkpoints import atomic_pickle, policy_hash, save_params_payload
from route_rl.full_action.global_update import GlobalUpdate
from route_rl.full_action.model import JaxModelConfig
from route_rl.full_action.sharding import put_replicated


from types import SimpleNamespace

def run_training(initial: Path, cache: Path, output: Path, config: dict) -> None:
    args = SimpleNamespace(initial=initial, cache=cache, output=output, **config)
    devices = tuple(jax.local_devices())
    if len(devices) != 1:
        raise RuntimeError("BC uses one local device per process")
    from jax.experimental import multihost_utils

    processes, rank = jax.process_count(), jax.process_index()
    if processes != 1:
        raise RuntimeError("this project BC entry supports one process; distributed launch is not implemented")
    batch_size = args.batch_per_gpu
    if batch_size < 2 or batch_size % 2:
        raise ValueError("even batch >=2 required")
    args.output.mkdir(parents=True, exist_ok=True)
    initial = host_checkpoint(args.initial)
    model = JaxModelConfig(**initial["model_config"])
    model.validate()
    if model.rope_correction_backend != "partitioned":
        raise ValueError("feature-only BC cache requires the partitioned model config")
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(args.learning_rate, eps=1e-5))
    params, state = initial["state"]["params"], optimizer.init(initial["state"]["params"])
    epoch = 0
    latest = args.output / "latest_bc_state.pkl"
    initial_sha = file_sha256(args.initial)
    cache_sha = file_sha256(args.cache / "index.json")
    coefficients = [args.unit_entropy, args.market_entropy, args.teacher_kl]
    if any(not np.isfinite(x) or x < 0 for x in coefficients):
        raise ValueError("regularization coefficients must be finite and nonnegative")
    if latest.exists():
        with latest.open("rb") as source:
            saved = pickle.load(source)
        if saved["initial_sha256"] != initial_sha or saved["cache_sha256"] != cache_sha:
            raise ValueError("BC resume identity mismatch")
        if saved.get("regularization") != coefficients:
            raise ValueError("BC resume regularization mismatch")
        params, state, epoch = saved["params"], saved["optimizer_state"], saved["epoch"]
    train_step, validation_step = make_steps(
        model, jnp.bfloat16 if args.compute_dtype == "bfloat16" else jnp.float32, optimizer, *coefficients
    )
    update, validate = GlobalUpdate(train_step, jax.devices()), GlobalUpdate(validation_step, jax.devices())

    def distributed(tree):
        local = put_replicated(tree, devices)
        return update.to_global(local) if processes > 1 else local

    params, state = distributed(params), distributed(state)
    teacher = distributed(initial["state"]["params"])
    training = ReplayDataset(args.cache, "train", processes, rank)
    validation = ReplayDataset(args.cache, "validation", processes, rank)
    print(
        json.dumps(
            {
                "event": "data",
                "train_rows": training.size,
                "validation_rows": validation.size,
                "initial_sha256": initial_sha,
                "batch_per_gpu": batch_size,
                "completed_epochs": epoch,
                "regularization": coefficients,
            }
        ),
        flush=True,
    )
    first_epoch = epoch + 1
    for epoch in range(first_epoch, args.epochs + 1):
        started = time.monotonic()
        summary = {}
        for name, dataset, learning in (("train", training, True), ("validation", validation, False)):
            indices = np.arange(dataset.size)
            if learning:
                np.random.default_rng(np.random.SeedSequence([args.seed, epoch, rank])).shuffle(indices)
            metrics = []
            maximum_rows = int(np.max(multihost_utils.process_allgather(np.asarray(len(indices)))))
            for offset in range(0, maximum_rows, batch_size):
                selected = indices[offset : offset + batch_size]
                batch = dataset.batch(selected if len(selected) else indices[:1], batch_size)
                for key in ("unit_action", "market_action"):
                    batch[key][len(selected) :] = IGNORE_LABEL
                batch["sample_mask"] = (np.arange(batch_size) < len(selected)).astype(np.float32)
                batch = jax.tree.map(lambda value: value[None], batch)
                if processes > 1:
                    batch = update.to_global(batch)
                if learning:
                    params, state, row = update(params, state, teacher, batch, True)
                else:
                    row = validate(params, teacher, batch, True)
                row = jax.device_get(update.to_local(row) if processes > 1 else row)
                if not np.isfinite(np.asarray(row["loss"])).all():
                    raise RuntimeError("nonfinite BC loss")
                metrics.append(row)
                if offset % (100 * batch_size) == 0:
                    print(
                        json.dumps(
                            {
                                "event": "batch",
                                "epoch": epoch,
                                "split": name,
                                "rows": offset,
                                "loss": float(np.asarray(row["loss"]).item()),
                            }
                        ),
                        flush=True,
                    )
            summary[name] = aggregate(metrics)
        host_params = first_local_replica(params)
        boundary = {
            "params": host_params,
            "optimizer_state": first_local_replica(state),
            "epoch": epoch,
            "initial_sha256": initial_sha,
            "cache_sha256": cache_sha,
            "shuffle_seed": [args.seed, epoch],
            "model_config": model.to_dict(),
            "regularization": coefficients,
        }
        if rank == 0:
            atomic_pickle(latest, boundary)
            if args.save_epoch_policies:
                save_params_payload(args.output / f"epoch-{epoch}-policy.pkl", host_params, model.to_dict(), 0)
            record = {"epoch": epoch, "seconds": time.monotonic() - started, **summary}
            with (args.output / "metrics.jsonl").open("a") as output:
                output.write(json.dumps(record) + "\n")
            print(json.dumps({"event": "epoch", **record}), flush=True)
        multihost_utils.sync_global_devices(f"replay-bc-epoch-{epoch}")
    if rank != 0:
        multihost_utils.sync_global_devices("replay-bc-complete")
        jax.distributed.shutdown()
        return
    host_params = first_local_replica(params)
    save_params_payload(args.output / "final_student_jax.pkl", host_params, model.to_dict(), 0)
    (args.output / "receipt.json").write_text(
        json.dumps(
            {
                "epochs": args.epochs,
                "initial_sha256": initial_sha,
                "policy_sha256": policy_hash(host_params),
                "cache_sha256": cache_sha,
                "value_training": False,
                "entropy_coefficients": coefficients[:2],
                "teacher_kl_coefficient": args.teacher_kl,
                "teacher_kl_direction": "initial_teacher || student",
                "teacher_policy_sha256": policy_hash(first_local_replica(teacher)),
            },
            indent=2,
        )
        + "\n"
    )
    multihost_utils.sync_global_devices("replay-bc-complete")
    if processes > 1:
        jax.distributed.shutdown()

