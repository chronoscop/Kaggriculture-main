"""Project-owned full-action replay BC, with explicit data and checkpoint contracts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess

from .replay_download import load_index, read_replay, write_json
from .replay_rules import compatibility_receipt, replay_seed

CONTRACT = "public-full-action-bc-v3"
PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PACKAGE_ROOT / "full_action/configs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity() -> dict:
    files = [PACKAGE_ROOT / name for name in ("action_bc.py", "replay_prepare.py", "replay_rules.py", "replay_download.py")]
    files.extend((PACKAGE_ROOT / "full_action").rglob("*.py"))
    files.extend(CONFIG_ROOT.glob("*.json"))
    hashes = {str(path.relative_to(PACKAGE_ROOT)): sha256(path) for path in sorted(files)}
    return {"implementation": "route_rl.full_action", "reference_checkout_required": False,
            "files_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def run_training(initial: Path, cache: Path, output: Path, config: dict) -> None:
    from .full_action.trainer import run_training as run

    run(initial.resolve(), cache.resolve(), output.resolve(), config)


def initialize(model_name: str, output: Path, seed: int) -> None:
    import jax
    import jax.numpy as jnp
    from .full_action.model import JaxModelConfig, initialize_params, add_zero_value_head, parameter_count
    from .full_action.checkpoints import save_params_payload

    model = JaxModelConfig(**json.loads((CONFIG_ROOT / f"{model_name}.json").read_text()))
    params = add_zero_value_head(initialize_params(jax.random.PRNGKey(seed), model), model)
    params["value"]["linear_cost"] = jnp.zeros(2, jnp.float32)
    save_params_payload(output, params, model.to_dict(), 0)
    print(json.dumps({"parameters": parameter_count(params), "output": str(output), "contract": CONTRACT}))


def validation_episode(episode: int) -> bool:
    digest = hashlib.sha256(f"50:{episode}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10 == 0


def inspect_index(index: Path) -> dict:
    rows = load_index(index)
    if not rows:
        raise ValueError("empty teacher index")
    games, sources, seeds, seed_splits = {}, Counter(), set(), {}
    for row in rows.values():
        episode, submission, seat = int(row["episode_id"]), int(row["submission_id"]), int(row["seat"])
        if episode <= 0 or submission <= 0 or seat not in (0, 1):
            raise ValueError("episode/submission IDs must be positive and seat must be 0 or 1")
        path = (index.parent / row["path"]).resolve()
        digest = sha256(path)
        if row.get("replay_sha256", digest) != digest:
            raise ValueError(f"replay checksum changed: {episode}")
        if episode in games:
            if games[episode][0] != digest:
                raise ValueError(f"different replay files claim episode {episode}")
        else:
            replay = read_replay(path)
            seed = replay_seed(replay)
            games[episode] = digest, seed
            if seed is not None:
                seed = int(seed)
                split = "validation" if validation_episode(episode) else "train"
                if seed in seed_splits and seed_splits[seed] != split:
                    raise ValueError(f"known demonstration seed {seed} crosses train/validation episodes; "
                                     "exclude the conflicting games from the teacher index")
                seed_splits[seed] = split
                seeds.add(seed)
        sources[submission] += 1
    splits = Counter("validation" if validation_episode(episode) else "train" for episode, _ in rows)
    if set(splits) != {"train", "validation"}:
        raise ValueError("episode-hash split needs both train and validation games; download more games")
    return {"teacher_seats": len(rows), "unique_games": len(games), "splits": dict(splits),
            "by_submission": dict(sources), "known_game_seeds": sorted(seeds),
            "games_without_seed": sum(seed is None for _, seed in games.values()),
            "index_sha256": sha256(index),
            "replays_sha256": hashlib.sha256(json.dumps(games, sort_keys=True).encode()).hexdigest()}


def prepare(index: Path, output: Path, workers: int) -> None:
    from .replay_prepare import prepare_index

    inventory = inspect_index(index)
    identity = {"contract": CONTRACT, "source": source_identity(), "input": inventory,
                "replay_compatibility": compatibility_receipt()}
    receipt = output / "preparation.json"
    if output.exists() and any(output.iterdir()):
        if not receipt.exists() or json.loads(receipt.read_text()) != identity:
            raise ValueError("preparation input or source changed; use a new output directory")
    write_json(receipt, identity)
    cache = output / "cache"
    prepare_index(index, output, workers)
    write_json(cache / "pipeline.json", identity)
    print(json.dumps({"cache": str(cache), **inventory}), flush=True)


def audit_cache(cache: Path) -> dict:
    import numpy as np

    index = json.loads((cache / "index.json").read_text())
    seen, splits, hashes, decoded_bytes = {}, Counter(), {}, 0
    valid_labels = Counter()
    for row in index["episodes"]:
        identity = int(row["episode_id"]), int(row["seat"])
        if identity in seen:
            raise ValueError(f"duplicate episode/seat in cache: {identity}")
        seen[identity] = row["split"]
        if row["split"] not in ("train", "validation"):
            raise ValueError("unknown cache split")
        expected = "validation" if validation_episode(identity[0]) else "train"
        if row["split"] != expected:
            raise ValueError("public episode-hash split differs; both seats must stay together")
        if Path(row["key"]).name != row["key"]:
            raise ValueError("cache key must be a filename")
        path = cache / f"{row['key']}.npz"
        with np.load(path, allow_pickle=False) as archive:
            features, labels = archive["features"], archive["labels"]
            if features.shape != (719, 264, 124) or labels.shape != (719, 30):
                raise ValueError(f"wrong full-trajectory tensor shapes: {path}")
            if not np.issubdtype(labels.dtype, np.integer) or not np.isfinite(features).all():
                raise ValueError(f"invalid tensor values: {path}")
            for name, values, maximum in (("unit", labels[:, :20], 500), ("market", labels[:, 20:], 1903)):
                if not np.all((values == -100) | ((values >= 0) & (values < maximum))):
                    raise ValueError(f"invalid {name} action IDs: {path}")
                valid_labels[name] += int(np.sum(values >= 0))
            decoded_bytes += features.nbytes + labels.nbytes
        hashes[row["key"]] = sha256(path)
        splits[row["split"]] += 719
    if set(splits) != {"train", "validation"} or any(count == 0 for count in valid_labels.values()):
        raise ValueError("cache needs both splits and valid unit/market labels")
    return {"index_sha256": sha256(cache / "index.json"),
            "arrays_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "samples": dict(splits), "valid_labels": dict(valid_labels), "decoded_bytes": decoded_bytes}


def train(args) -> None:
    preparation = json.loads((args.cache / "pipeline.json").read_text())
    source = source_identity()
    if preparation["source"] != source or preparation["contract"] != CONTRACT:
        raise ValueError("cache encoding source changed; rebuild in a new preparation directory")
    config = json.loads((CONFIG_ROOT / "bc.json").read_text())
    config.update(epochs=args.epochs, batch_per_gpu=args.batch_size, compute_dtype=args.compute_dtype)
    cache = audit_cache(args.cache)
    print(json.dumps({"event": "cache_audit", **cache}), flush=True)
    memory_info = Path("/proc/meminfo")
    if memory_info.exists():
        available = next((int(line.split()[1]) * 1024 for line in memory_info.read_text().splitlines()
                          if line.startswith("MemAvailable:")), None)
        if available is not None and cache["decoded_bytes"] > available * 0.8:
            raise ValueError(f"BC trainer loads all tensors into host RAM ({cache['decoded_bytes'] / 2**30:.1f} GiB); "
                             "use a smaller teacher subset or a host with more RAM")
    identity = {"contract": CONTRACT, "source": source, "initial_sha256": sha256(args.initial),
                "cache": cache, "settings": {key: value for key, value in config.items() if key != "epochs"},
                "preparation": preparation, "deployment": "candidate_only"}
    receipt = args.out / "integration.json"
    if args.out.exists() and any(args.out.iterdir()):
        if not receipt.exists() or json.loads(receipt.read_text()) != identity:
            raise ValueError("BC resume source/data/settings mismatch; use a new run directory")
        previous_config = args.out / "bc_config.json"
        if previous_config.exists() and args.epochs < json.loads(previous_config.read_text())["epochs"]:
            raise ValueError("BC epochs are cumulative; a resumed run cannot decrease its epoch target")
    write_json(receipt, identity)
    write_json(args.out / "bc_config.json", config)
    run_training(args.initial, args.cache, args.out, config)


def evaluate(args) -> None:
    identity = json.loads((args.run / "integration.json").read_text())
    if identity["contract"] != CONTRACT or identity["source"] != source_identity():
        raise ValueError("source changed since BC; restore the source before evaluating")
    evaluation_seeds = set(range(args.seed, args.seed + args.games // 2))
    known = set(identity["preparation"]["input"]["known_game_seeds"])
    if known & evaluation_seeds:
        raise ValueError("evaluation seeds overlap recorded demonstration games")
    if args.out.exists() or args.out.with_suffix(".games.json").exists():
        raise ValueError("evaluation output already exists; use a new output file")
    policy = args.policy or args.run / "final_student_jax.pkl"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from .full_action.evaluation import evaluate_games

    raw = args.out.with_suffix(".games.json")
    records = evaluate_games(policy, args.opponent, args.seed, args.games, raw)
    if len(records) != args.games or any(row["statuses"] != ["DONE", "DONE"] or row["turns"] != 720 for row in records):
        raise ValueError("evaluation contains failed/incomplete games; inspect the raw game report")
    scores = []
    for row in records:
        own, other = row["rewards"][row["a_seat"]], row["rewards"][1 - row["a_seat"]]
        scores.append(1.0 if own > other else 0.5 if own == other else 0.0)
    write_json(args.out, {"contract": CONTRACT, "policy_sha256": sha256(policy),
                         "opponent_sha256": sha256(args.opponent), "games": records,
                         "score_rate": sum(scores) / len(scores), "deployment": "candidate_only",
                         "demonstration_games_without_seed": identity["preparation"]["input"]["games_without_seed"],
                         "inference": "route_rl.full_action.inference.GreedyPolicy"})
    print(json.dumps({"games": len(scores), "score_rate": sum(scores) / len(scores), "report": str(args.out)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check source and dependency availability without training")
    prep = commands.add_parser("prepare", help="index and encode full public teacher trajectories")
    prep.add_argument("--index", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--workers", type=int, default=4)
    init = commands.add_parser("init", help="initialize this project's full-action model")
    init.add_argument("--model", choices=("bootstrap", "10m", "smoke"), default="bootstrap")
    init.add_argument("--out", type=Path, required=True)
    init.add_argument("--seed", type=int, default=0)
    learn = commands.add_parser("train", help="train the full-action BC candidate")
    learn.add_argument("--initial", type=Path, required=True)
    learn.add_argument("--cache", type=Path, required=True)
    learn.add_argument("--out", type=Path, required=True)
    learn.add_argument("--epochs", type=int, default=2)
    learn.add_argument("--batch-size", type=int, default=320)
    learn.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    check = commands.add_parser("audit-cache", help="validate labels, shapes, splits and checksums")
    check.add_argument("--cache", type=Path, required=True)
    match = commands.add_parser("evaluate", help="paired fresh-seed candidate evaluation")
    match.add_argument("--run", type=Path, required=True)
    match.add_argument("--policy", type=Path)
    match.add_argument("--opponent", type=Path, required=True)
    match.add_argument("--seed", type=int, required=True)
    match.add_argument("--games", type=int, default=16)
    match.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            packages = {}
            for package in ("numpy", "jax", "optax", "kaggle-environments", "kaggle"):
                try:
                    packages[package] = metadata.version(package)
                except metadata.PackageNotFoundError:
                    packages[package] = None
            print(json.dumps({"source": source_identity(), "dependencies": packages,
                              "required_versions": {"numpy": "2.5.3", "jax": "0.11.1", "optax": "0.2.8",
                                                    "kaggle-environments": "1.32.7", "kaggle": "2.2.4"}}))
        elif args.command == "prepare":
            if args.workers < 1:
                raise ValueError("workers must be positive")
            prepare(args.index, args.out, args.workers)
        elif args.command == "init":
            if args.out.exists():
                raise ValueError("initial policy already exists; choose a new path")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            initialize(args.model, args.out, args.seed)
        elif args.command == "audit-cache":
            print(json.dumps(audit_cache(args.cache)))
        elif args.command == "train":
            if args.epochs < 1 or args.batch_size < 2 or args.batch_size % 2:
                raise ValueError("epochs must be positive and batch-size even and >=2")
            train(args)
        else:
            if args.games < 2 or args.games % 2:
                raise ValueError("games must be even and >=2")
            evaluate(args)
    except (ValueError, RuntimeError, OSError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
