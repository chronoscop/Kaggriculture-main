"""Prepare verified public replays using this project's full-action BC components."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from .replay_download import load_index, read_replay, write_json
from .replay_rules import LABEL_CONTRACT, compatibility_receipt, require_pinned_rules


def prepare_trajectory(task: tuple[dict, str, str]) -> dict:
    row, replay_path, cache_directory = task
    import numpy as np
    from .full_action.legality import action_legality
    from .full_action.labels import label_actions
    from .full_action.features import FEATURE_DIM, encode_observation
    from .full_action.inventory_tracker import OpponentInventoryTracker
    from .action_bc import validation_episode

    require_pinned_rules()
    source = Path(replay_path)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    if row.get("replay_sha256", checksum) != checksum:
        raise ValueError(f"replay checksum changed: {row['episode_id']}")
    episode, seat, submission = int(row["episode_id"]), int(row["seat"]), int(row["submission_id"])
    key = f"{submission}-{episode}-{seat}"
    cache = Path(cache_directory)
    receipt, archive = cache / f"{key}.json", cache / f"{key}.npz"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved.get("replay_sha256") != checksum or saved.get("label_contract") != LABEL_CONTRACT:
            raise ValueError("cached replay or label contract changed; use a new preparation directory")
        if archive.is_file():
            return saved
    replay = read_replay(source)
    steps = replay["steps"]
    features = np.zeros((719, 264, FEATURE_DIM), np.float16)
    labels = np.full((719, 30), -100, np.int16)
    tracker = OpponentInventoryTracker(observer_player=seat)
    for turn in range(719):
        observation = steps[turn][seat]["observation"]
        if turn:
            tracker.update(steps[turn - 1][seat]["observation"], observation, steps[turn][seat].get("action", {}))
        actions = [state.get("action", {}) for state in steps[turn + 1]]
        encoded = encode_observation(observation, tracker.estimate())
        if len(encoded.features) > 264 or 1 + len(observation["farms"][seat].get("hands", [])) > 20:
            raise ValueError(f"episode {episode}, turn {turn}: units exceed the project model capacity")
        features[turn, :len(encoded.features)] = encoded.features
        labels[turn] = label_actions(observation, actions[seat] or {},
                                    action_legality([state["observation"] for state in steps[turn]], actions, seat))
    temporary = archive.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, features=features, labels=labels)
    temporary.replace(archive)
    result = {"key": key, "episode_id": episode, "submission_id": submission, "seat": seat,
              "split": "validation" if validation_episode(episode) else "train", "samples": 719,
              "valid_labels": int(np.sum(labels != -100)), "ignored_labels": int(np.sum(labels == -100)),
              "replay_sha256": checksum, "replay_module_version": replay["module_version"],
              **compatibility_receipt()}
    write_json(receipt, result)
    return result


def prepare_index(index: Path, output: Path, workers: int) -> None:
    rows = list(load_index(index).values())
    cache, manifests = output / "cache", output / "manifests"
    cache.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    tasks, sources = [], {}
    for row in rows:
        path = (index.parent / row["path"]).resolve()
        sources.setdefault(row["submission_id"], []).append({**row, "path": str(path)})
        tasks.append((row, str(path), str(cache.resolve())))
    for submission, episodes in sources.items():
        write_json(manifests / str(submission) / "manifest.json",
                   {"source": {"submission_id": submission}, "episodes": episodes,
                    "source_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                    **compatibility_receipt()})
    prepared = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(prepare_trajectory, tasks):
            prepared.append(row)
            print(json.dumps({"event": "prepared", "teacher_seats": len(prepared),
                              "episode_id": row["episode_id"], "seat": row["seat"]}), flush=True)
    write_json(cache / "index.json", {"schema_version": 1, "seed": 50, "episodes": prepared,
                                      **compatibility_receipt()})
