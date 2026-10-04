"""Generate owned search-teacher demonstrations; never starts BC training."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path

TEACHER_CONTRACT = "owned-search-teacher-actual-actions-v1"
COMBINE_CONTRACT = "owned-public-search-teacher-combined-index-v1"
SUBMISSION_ID = 9_900_001


def teacher_episode(seed: int, validation: bool) -> int:
    from .action_bc import validation_episode
    base = 1_000_000_000_000 + seed * 100
    for nonce in range(100):
        identity = base + nonce
        if validation_episode(identity) == validation:
            return identity
    raise ValueError("could not assign a BC-compatible whole-seed split")


def select_seeds(seeds, sources, seed_start, games, excluded):
    from .replay_download import read_replay
    from .replay_rules import replay_seed
    values = list(seeds)
    provenance = []
    for path in sources:
        replay = read_replay(path)
        seed = replay_seed(replay)
        if seed is None:
            raise ValueError(f"failed replay has no verified seed: {path}")
        values.append(seed)
        provenance.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "seed": seed, "source_configuration": replay.get("configuration", {}),
                           "role": "diagnostic seed selection; generated under pinned default rules; no original action labels reused"})
    if seed_start is not None:
        if games is None or games < 1:
            raise ValueError("--seed-start requires --games >= 1")
        values.extend(range(seed_start, seed_start + games))
    elif games is not None:
        raise ValueError("--games requires --seed-start")
    values = sorted(set(values))
    if not values or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 < seed < 2**31 for seed in values):
        raise ValueError("teacher requires nonzero signed-32-bit explicit seeds or verified source replays")
    if set(values) & set(excluded):
        raise ValueError("teacher seeds overlap reserved held-out/evaluation seeds")
    return values, provenance


def run_exclusions(paths):
    """Use explicit existing run evidence without loading arbitrary checkpoints."""
    seeds, ranges, records = set(), [], []
    names = {"known_game_seeds", "training_seeds", "validation_seeds", "heldout_seeds",
             "critic_training_seeds", "critic_validation_seeds", "seeds"}
    def visit(value):
        if isinstance(value, dict):
            for key, member in value.items():
                if key in names and isinstance(member, list) and all(isinstance(seed, int) for seed in member):
                    seeds.update(member)
                elif key == "reserved_training_ranges":
                    ranges.extend((int(start), int(stop)) for start, stop in member)
                else:
                    visit(member)
        elif isinstance(value, list):
            for member in value:
                visit(member)
    for supplied in paths:
        path = supplied / "integration.json" if supplied.is_dir() else supplied
        if not path.exists():
            raise FileNotFoundError(f"exclusion run identity missing: {path}")
        for document in [path, *[file for file in (path.parent / "receipt.json",) if file.exists()]]:
            visit(json.loads(document.read_text()))
            records.append({"path": str(document.resolve()), "sha256": hashlib.sha256(document.read_bytes()).hexdigest()})
    return seeds, sorted(set(ranges)), records


def generate_game(seed, seconds, uncertainty_scale):
    from kaggle_environments import make
    from .ppo.controllers import validate_action
    from .replay_download import validate_replay
    from .replay_rules import require_pinned_rules
    from .season_search.search_policy import SearchPolicy

    require_pinned_rules()
    environment = make("kaggriculture", configuration={"seed": seed}, debug=False)
    planners = [SearchPolicy(seconds=seconds), SearchPolicy(seconds=seconds)]
    for planner in planners:
        planner.config["ponder"] = False
    for step in range(719):
        if environment.done:
            raise ValueError("teacher game ended early")
        actions = []
        for player in (0, 1):
            obs = deepcopy(dict(environment.state[player].observation))
            obs["_terminal_uncertainty"] = uncertainty_scale
            action = planners[player](obs)
            validate_action(action, obs)
            actions.append(action)
        environment.step(actions)
    replay = environment.toJSON()
    if isinstance(replay, str):
        replay = json.loads(replay)
    replay["module_version"] = "1.32.7"
    replay["name"] = "kaggriculture"
    validate_replay(replay)
    return replay


def generate(args):
    from .ppo.controllers import DEFAULTS, OBJECTIVE, contract_identity
    from .replay_download import write_json
    exclusions = set()
    if args.exclude_seeds:
        value = json.loads(args.exclude_seeds.read_text())
        exclusions = set(value if isinstance(value, list) else value["seeds"])
    run_seeds, ranges, exclusion_sources = run_exclusions(args.exclude_run)
    exclusions.update(run_seeds)
    seeds, sources = select_seeds(args.seed, args.source_replay, args.seed_start, args.games, exclusions)
    if any(start <= seed < stop for seed in seeds for start, stop in ranges):
        raise ValueError("teacher seeds overlap excluded run reserved training/validation ranges")
    if not 0 <= args.holdout_games < len(seeds):
        raise ValueError("holdout-games must be >= 0 and smaller than total games; use >= 1 for BC-ready data")
    if args.seconds <= 0 or args.seconds > 6 or not math.isfinite(args.seconds) or args.uncertainty_scale < 1 or not math.isfinite(args.uncertainty_scale):
        raise ValueError("teacher search requires 0 < seconds <= 6 and uncertainty scale >= 1")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("teacher output must be new; immutable generated demonstrations are not overwritten")
    ordered = sorted(seeds, key=lambda seed: hashlib.sha256(f"{args.split_seed}:{seed}".encode()).digest())
    heldout = set(ordered[:args.holdout_games])
    identity = contract_identity({**DEFAULTS, "search_seconds": args.seconds,
                                  "uncertainty_scale": args.uncertainty_scale})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "replays").mkdir()
    rows, records = [], []
    receipt = {"contract": TEACHER_CONTRACT, "objective": OBJECTIVE, "planner": identity,
               "search_seconds_per_dawn": args.seconds, "uncertainty_scale": args.uncertainty_scale,
               "seeds": seeds, "heldout_seeds": sorted(heldout), "training_seeds": sorted(set(seeds)-heldout),
               "source_replays": sources, "split": "seed-hash-selected; BC-compatible episode-id assignment",
               "excluded_run_sources": exclusion_sources, "excluded_seed_ranges": ranges,
               "split_seed": args.split_seed, "training_started": False, "complete": False}
    write_json(args.out / "teacher_receipt.json", receipt)
    for seed in seeds:
        replay = generate_game(seed, args.seconds, args.uncertainty_scale)
        episode = teacher_episode(seed, seed in heldout)
        relative = f"replays/{episode}.json.gz"
        path = args.out / relative
        data = json.dumps(replay, separators=(",", ":")).encode()
        path.write_bytes(gzip.compress(data, mtime=0))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for player in (0, 1):
            rows.append({"episode_id": episode, "submission_id": SUBMISSION_ID, "seat": player,
                         "seed": seed, "path": relative, "replay_sha256": digest,
                         "teacher_contract": TEACHER_CONTRACT,
                         "split": "validation" if seed in heldout else "train"})
        index = args.out / "teacher-seats.jsonl"
        index.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        final = replay["steps"][-1]
        record = {"seed": seed, "episode_id": episode, "turns": len(replay["steps"]),
                  "rewards": [state["reward"] for state in final], "statuses": [state["status"] for state in final],
                  "replay_sha256": digest, "split": "validation" if seed in heldout else "train"}
        records.append(record)
        write_json(args.out / "games.json", {"games": records})
        print(json.dumps({"event": "teacher_game", **record}), flush=True)
    receipt.update(complete=True, games=len(records), teacher_seats=len(rows),
                   index_sha256=hashlib.sha256((args.out / "teacher-seats.jsonl").read_bytes()).hexdigest())
    write_json(args.out / "teacher_receipt.json", receipt)
    if heldout:
        from .action_bc import inspect_index
        receipt["bc_index_audit"] = inspect_index(args.out / "teacher-seats.jsonl")
        write_json(args.out / "teacher_receipt.json", receipt)
    return receipt


def combine(args):
    """Merge verified source indexes with absolute, checksum-pinned replay paths."""
    from .action_bc import inspect_index, validation_episode
    from .replay_download import load_index, read_replay, write_json
    from .replay_rules import replay_seed
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("combined output must be new; indexes are immutable")
    receipt_path = args.teacher_index.parent / "teacher_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("contract") != TEACHER_CONTRACT or receipt.get("complete") is not True:
        raise ValueError("teacher generation must finish before combining")
    if hashlib.sha256(args.teacher_index.read_bytes()).hexdigest() != receipt.get("index_sha256"):
        raise ValueError("teacher index bytes changed since generation")
    inventory = inspect_index(args.public_index)
    seen, games, seed_splits, replay_splits, public_holdout, replay_cache = {}, {}, {}, {}, set(), {}
    for kind, index in (("public", args.public_index), ("teacher", args.teacher_index)):
        rows = load_index(index)
        if kind == "teacher" and len(rows) != receipt.get("teacher_seats"):
            raise ValueError("teacher receipt/index count mismatch")
        for key, source_row in rows.items():
            row = dict(source_row)
            path = (index.parent / row["path"]).resolve()
            if path not in replay_cache:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                replay = read_replay(path)
                replay_cache[path] = digest, replay_seed(replay)
            digest, seed = replay_cache[path]
            if row.get("replay_sha256", digest) != digest:
                raise ValueError(f"{kind} replay bytes changed: {path}")
            if kind == "teacher" and row.get("replay_sha256") != digest:
                raise ValueError("generated teacher rows require explicit replay checksums")
            if seed is None or seed == 0:
                if kind == "teacher":
                    raise ValueError("generated teachers require explicit nonzero replay seeds")
            if "seed" in row and row["seed"] != seed:
                raise ValueError("index seed differs from actual replay seed")
            episode = int(row["episode_id"])
            split = "validation" if validation_episode(episode) else "train"
            if row.get("split", split) != split:
                raise ValueError("index split differs from unchanged BC episode-hash contract")
            if kind == "teacher":
                expected = "validation" if seed in receipt["heldout_seeds"] else "train"
                if seed not in receipt["seeds"] or split != expected or row.get("teacher_contract") != TEACHER_CONTRACT:
                    raise ValueError("teacher seed/split contract differs from its receipt")
            if episode in games and games[episode] != (digest, seed):
                raise ValueError("different actual games claim the same episode ID")
            games[episode] = digest, seed
            if digest in replay_splits and replay_splits[digest] != split:
                raise ValueError("same actual replay crosses train/validation; public holdout games must stay protected")
            replay_splits[digest] = split
            if seed is not None:
                if seed in seed_splits and seed_splits[seed] != split:
                    raise ValueError("same seed crosses train/validation; public holdout seeds must stay protected")
                seed_splits[seed] = split
                if kind == "public" and split == "validation":
                    public_holdout.add(seed)
            row.update(path=str(path), replay_sha256=digest, seed=seed, split=split)
            if key in seen:
                previous = seen[key]
                essential = ("submission_id", "replay_sha256", "seed", "split")
                if any(previous.get(field) != row.get(field) for field in essential):
                    raise ValueError("conflicting duplicate teacher seat identity")
                continue
            seen[key] = row
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "teacher-seats.jsonl"
    output.write_text("".join(json.dumps(seen[key], sort_keys=True) + "\n" for key in sorted(seen)))
    combined = inspect_index(output)
    result = {"contract": COMBINE_CONTRACT, "source_indexes": {
        kind: {"path": str(index.resolve()), "sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
        for kind, index in (("public", args.public_index), ("teacher", args.teacher_index))},
        "teacher_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "public_holdout_seeds": sorted(public_holdout), "public_inventory": inventory,
        "combined_inventory": combined, "absolute_replay_paths": True, "training_started": False}
    write_json(args.out / "combine_receipt.json", result)
    print(json.dumps({"event": "teacher_indexes_combined", "index": str(output),
                      "teacher_seats": combined["teacher_seats"], "splits": combined["splits"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("generate", help="full official search self-play; saves actual requests for both seats")
    create.add_argument("--out", type=Path, required=True)
    create.add_argument("--seed", type=int, action="append", default=[])
    create.add_argument("--source-replay", type=Path, action="append", default=[])
    create.add_argument("--seed-start", type=int)
    create.add_argument("--games", type=int)
    create.add_argument("--holdout-games", type=int, default=1)
    create.add_argument("--split-seed", type=int, default=51)
    create.add_argument("--exclude-seeds", type=Path)
    create.add_argument("--exclude-run", type=Path, action="append", default=[])
    create.add_argument("--seconds", type=float, default=0.25)
    create.add_argument("--uncertainty-scale", type=float, default=2000.)
    merge = commands.add_parser("combine", help="verified immutable public+generated teacher index for existing BC prepare")
    merge.add_argument("--public-index", type=Path, required=True)
    merge.add_argument("--teacher-index", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True, help="new directory; writes teacher-seats.jsonl and combine_receipt.json")
    args = parser.parse_args()
    try:
        generate(args) if args.command == "generate" else combine(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
