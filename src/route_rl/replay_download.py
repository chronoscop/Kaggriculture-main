"""Download explicit public teachers with the official Kaggle API, for action BC."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import time

from .replay_rules import COMPATIBLE_REPLAY_VERSIONS, compatibility_receipt, replay_seed

SEED_ZERO_SKIP_REASON = "excluded_seed_zero"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_replay(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        replay = json.load(stream)
    validate_replay(replay)
    return replay


def validate_replay(replay: dict) -> None:
    if replay.get("module_version") not in COMPATIBLE_REPLAY_VERSIONS:
        raise ValueError(f"unverified replay version {replay.get('module_version')}; "
                         f"supported versions: {COMPATIBLE_REPLAY_VERSIONS}")
    if replay.get("name", "kaggriculture") != "kaggriculture":
        raise ValueError("replay is not Kaggriculture")
    steps = replay.get("steps", [])
    if len(steps) != 720:
        raise ValueError(f"incomplete replay: {len(steps)} states, expected 720")
    for turn, states in enumerate(steps):
        if len(states) != 2:
            raise ValueError(f"turn {turn}: expected two seats")
        for seat, state in enumerate(states):
            observation = state.get("observation", {})
            step = observation.get("step")
            if step is None and "day" in observation and "hour" in observation:
                step = int(observation["day"]) * 24 + int(observation["hour"])
            if observation.get("player") != seat or step != turn:
                raise ValueError(f"turn {turn}, seat {seat}: observation alignment mismatch")
            if ("day" in observation and int(observation["day"]) != turn // 24) or (
                "hour" in observation and int(observation["hour"]) != turn % 24
            ):
                raise ValueError(f"turn {turn}, seat {seat}: day/hour alignment mismatch")
            if not all(key in observation for key in ("farms", "private", "market", "town")):
                raise ValueError(f"turn {turn}, seat {seat}: incomplete observation")
            if turn and not isinstance(state.get("action"), dict):
                raise ValueError(f"turn {turn}, seat {seat}: missing action[t+1] labels")
    if any(state.get("status") != "DONE" for state in steps[-1]):
        raise ValueError("both seats must finish normally; error/timeout games are excluded")
    replay_seed(replay)


def metadata_rejection(episode, submission: int) -> str | None:
    if episode.state.name != "COMPLETED":
        return "episode_not_completed"
    if not episode.agents:
        return "missing_agent_metadata"
    # Public API responses can omit per-agent state, leaving the SDK default.
    # Completion is verified from the full replay, not inferred from this default.
    allowed = {"EPISODE_AGENT_STATE_COMPLETE", "EPISODE_AGENT_STATE_UNSPECIFIED"}
    if any(agent.state.name not in allowed for agent in episode.agents):
        return "agent_pending_or_failed"
    if not any(agent.submission_id == submission for agent in episode.agents):
        return "teacher_not_in_episode"
    return None


def teacher_seats(episode, submission: int) -> list[int]:
    # Use agent.index, never the ordering of agents or the winning seat.
    if metadata_rejection(episode, submission) is not None:
        return []
    seats = [int(agent.index) for agent in episode.agents if agent.submission_id == submission]
    if len(set(seats)) != len(seats) or any(seat not in (0, 1) for seat in seats):
        raise ValueError(f"episode {episode.id}: invalid teacher-seat metadata")
    return seats


def call_api(function, *args, **kwargs):
    from requests import RequestException

    for attempt in range(3):
        try:
            return function(*args, **kwargs)
        except RequestException as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            if status in (401, 403):
                raise RuntimeError("Kaggle access denied. Run `kaggle auth login` and check competition access.") from None
            if attempt == 2 or (status is not None and status != 429 and status < 500):
                raise RuntimeError(f"Kaggle request failed ({status or type(error).__name__}); rerun to resume.") from None
            time.sleep(2 ** attempt)


def kaggle_api():
    try:
        with redirect_stdout(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise RuntimeError("Install the downloader: python -m pip install 'kaggle==2.2.4'") from None
    api = KaggleApi()
    for method in ("competition_team_submissions", "competition_list_episodes", "competition_episode_replay"):
        if not hasattr(api, method):
            raise RuntimeError("Kaggle client is too old; install kaggle==2.2.4")
    try:
        with redirect_stdout(io.StringIO()):
            api.authenticate()
    except (Exception, SystemExit):
        raise RuntimeError("Kaggle is not authenticated. Run `kaggle auth login` in this environment first.") from None
    return api


def discover(api, competition: str, count: int, output: Path) -> None:
    if output.exists():
        raise ValueError("teacher snapshot already exists; reuse it or choose a new output file")
    board = call_api(api.competition_leaderboard_view, competition, page_size=200)
    teachers, seen = [], set()
    for row in board or []:
        if row.team_id in seen:
            continue
        seen.add(row.team_id)
        submissions = call_api(api.competition_team_submissions, row.team_id)
        scored = []
        for submission in submissions or []:
            try:
                scored.append((Decimal(submission.public_score), submission))
            except (InvalidOperation, TypeError):
                continue
        scored = [(score, s) for score, s in scored if score.is_finite()]
        if not scored:
            continue
        score, submission = max(scored, key=lambda pair: (pair[0], pair[1].id))
        teachers.append({"submission_id": submission.id, "team_id": row.team_id,
                         "team_name": row.team_name, "public_score": str(score)})
        if len(teachers) == count:
            break
    if len(teachers) != count:
        raise ValueError(f"found only {len(teachers)} eligible teachers in the first leaderboard page; requested {count}")
    write_json(output, {"schema": "public-bc-teachers-v1", "competition": competition,
                        "captured_at": datetime.now(timezone.utc).isoformat(), "teachers": teachers})
    print(json.dumps({"teachers": len(teachers), "snapshot": str(output)}), flush=True)


def load_index(path: Path) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            identity = int(row["episode_id"]), int(row["seat"])
            if identity in rows and rows[identity] != row:
                raise ValueError(f"conflicting teacher identity: {identity}")
            rows[identity] = row
    return rows


def filter_zero_seed_index(output: Path) -> dict:
    """Remove complete zero-seed episodes in place; retain a recoverable backup."""
    index = output / "teacher-seats.jsonl"
    if not index.exists():
        return {"index": str(index), "removed_games": 0, "removed_teacher_seats": 0,
                "teacher_seats": 0, "unique_games": 0}
    original = index.read_bytes()
    rows = load_index(index)
    seeds = {}
    for row in rows.values():
        episode = int(row["episode_id"])
        if "seed" in row:
            seed = None if row["seed"] is None else int(row["seed"])
        elif episode in seeds:
            seed = seeds[episode]
        else:
            seed = replay_seed(read_replay(output / row["path"]))
        if episode in seeds and seeds[episode] != seed:
            raise ValueError(f"different seats record different seeds for episode {episode}")
        seeds[episode] = seed
    excluded = {episode for episode, seed in seeds.items() if seed == 0}
    retained = {key: row for key, row in rows.items() if key[0] not in excluded}
    result = {"index": str(index), "removed_games": len(excluded),
              "removed_teacher_seats": len(rows) - len(retained),
              "teacher_seats": len(retained), "unique_games": len({key[0] for key in retained})}
    if excluded:
        checksum = hashlib.sha256(original).hexdigest()
        backup = index.with_name(f"{index.stem}.before-zero-seed-{checksum[:12]}{index.suffix}")
        if backup.exists() and backup.read_bytes() != original:
            raise ValueError("zero-seed index backup differs; preserve it and choose a new download directory")
        if index.read_bytes() != original:
            raise ValueError("teacher index changed during filtering; stop concurrent downloads and rerun")
        if not backup.exists():
            backup.write_bytes(original)
        temporary = index.with_suffix(index.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in retained.values()))
        temporary.replace(index)
        with (output / "skipped.jsonl").open("a") as stream:
            for episode in sorted(excluded):
                stream.write(json.dumps({"episode_id": episode, "seed": 0,
                                         "reason": SEED_ZERO_SKIP_REASON}) + "\n")
        write_json(output / "seed-zero-filter.json",
                   {"schema": "public-bc-seed-zero-filter-v1", **result,
                    "source_index_sha256": checksum, "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                    "backup": backup.name, "excluded_episode_ids": sorted(excluded)})
    download_receipt = output / "download_receipt.json"
    if download_receipt.exists():
        receipt = json.loads(download_receipt.read_text())
        receipt.update(schema="public-bc-download-v3", excluded_seed_values=[0],
                       teacher_seats=len(retained), unique_games=len({key[0] for key in retained}),
                       index_sha256=hashlib.sha256(index.read_bytes()).hexdigest())
        write_json(download_receipt, receipt)
    return result


def zero_seed_episode_ids(output: Path) -> set[int]:
    path = output / "skipped.jsonl"
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {int(row["episode_id"]) for row in rows if row.get("reason") == SEED_ZERO_SKIP_REASON}


def download(api, teachers_file: Path, output: Path, limit: int, delay: float) -> None:
    snapshot = json.loads(teachers_file.read_text())
    if snapshot.get("competition") != "kaggriculture":
        raise ValueError("teacher snapshot must specify competition=kaggriculture")
    teachers = snapshot["teachers"]
    ids = [int(row["submission_id"]) for row in teachers]
    if not ids or len(ids) != len(set(ids)) or any(submission <= 0 for submission in ids):
        raise ValueError("teacher submission IDs must be positive and unique")
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "teachers.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != snapshot:
        raise ValueError("teacher snapshot changed; use a new download directory")
    write_json(snapshot_path, snapshot)
    replays = output / "replays"
    replays.mkdir(exist_ok=True)
    index_path = output / "teacher-seats.jsonl"
    filtered = filter_zero_seed_index(output)
    if filtered["removed_games"]:
        print(json.dumps({"event": "filtered_seed_zero", **filtered}), flush=True)
    rows = load_index(index_path)
    excluded_zero_seed = zero_seed_episode_ids(output)
    for submission in ids:
        episodes = call_api(api.competition_list_episodes, submission)
        # Stable outcome-independent sampling, retaining losses and draws too.
        episodes = sorted(episodes, key=lambda e: hashlib.sha256(f"51:{submission}:{e.id}".encode()).digest())
        accepted, skipped = 0, 0
        metadata_filtered = Counter()
        seen_episodes = set()
        for episode in episodes:
            if accepted == limit:
                break
            if episode.id in seen_episodes:
                continue
            seen_episodes.add(episode.id)
            rejection = metadata_rejection(episode, submission)
            if rejection is not None:
                metadata_filtered[rejection] += 1
                continue
            if int(episode.id) in excluded_zero_seed:
                skipped += 1
                continue
            seats = teacher_seats(episode, submission)
            destination = replays / f"{episode.id}.json.gz"
            fetched = False
            try:
                if destination.exists():
                    replay = read_replay(destination)
                else:
                    with tempfile.TemporaryDirectory(dir=replays) as temporary:
                        call_api(api.competition_episode_replay, episode.id, path=temporary, quiet=True)
                        fetched = True
                        replay = read_replay(Path(temporary) / f"episode-{episode.id}-replay.json")
                if replay_seed(replay) == 0:
                    excluded_zero_seed.add(int(episode.id))
                    raise ValueError(SEED_ZERO_SKIP_REASON)
                if fetched:
                    compressed = gzip.compress(json.dumps(replay, separators=(",", ":")).encode(), mtime=0)
                    temporary_path = destination.with_suffix(".tmp")
                    temporary_path.write_bytes(compressed)
                    temporary_path.replace(destination)
            except ValueError as error:
                skipped += 1
                with (output / "skipped.jsonl").open("a") as stream:
                    stream.write(json.dumps({"episode_id": episode.id, "reason": str(error)}) + "\n")
                continue
            finally:
                if fetched:
                    time.sleep(delay)
            checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
            for seat in seats:
                row = {"path": str(destination.relative_to(output)), "episode_id": int(episode.id),
                       "seat": seat, "submission_id": submission, "replay_sha256": checksum,
                       "seed": replay_seed(replay), "replay_module_version": replay["module_version"]}
                identity = episode.id, seat
                if identity in rows:
                    if rows[identity] != row:
                        raise ValueError(f"existing teacher/replay identity changed: {identity}")
                else:
                    with index_path.open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    rows[identity] = row
            accepted += 1
            print(json.dumps({"submission_id": submission, "accepted_games": accepted,
                              "episode_id": episode.id}), flush=True)
        print(json.dumps({"submission_id": submission, "accepted_games": accepted, "skipped_games": skipped,
                          "requested_games": limit, "listed_games": len(episodes),
                          "metadata_filtered_games": dict(metadata_filtered)}), flush=True)
    if not rows:
        raise ValueError("no compatible full replays downloaded; see skipped.jsonl and teacher IDs")
    write_json(output / "download_receipt.json", {"schema": "public-bc-download-v3", **compatibility_receipt(),
               "teacher_seats": len(rows), "unique_games": len({key[0] for key in rows}),
               "limit_per_teacher": limit, "selection": "outcome-independent episode hash, seed 51",
               "excluded_seed_values": [0],
               "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    find = commands.add_parser("discover", help="snapshot top public teachers, one active submission per team")
    find.add_argument("--competition", default="kaggriculture", choices=["kaggriculture"])
    find.add_argument("--top-teams", type=int, default=20)
    find.add_argument("--out", type=Path, required=True)
    fetch = commands.add_parser("download", help="download full replays and write BC teacher-seat JSONL")
    selection = fetch.add_mutually_exclusive_group(required=True)
    selection.add_argument("--teachers", type=Path, help="saved public teacher snapshot")
    selection.add_argument("--submission", type=int, action="append", help="explicit teacher submission ID; repeatable")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--limit-per-teacher", type=int, default=100)
    fetch.add_argument("--delay", type=float, default=1.0)
    clean = commands.add_parser("filter-zero-seed", help="exclude seed=0 episodes from the existing index in place")
    clean.add_argument("--out", type=Path, required=True, help="download directory containing teacher-seats.jsonl")
    args = parser.parse_args()
    if args.command == "discover" and not 1 <= args.top_teams <= 200:
        parser.error("top-teams must be between 1 and 200")
    if args.command == "download" and (args.limit_per_teacher < 1 or args.delay < 0 or not math.isfinite(args.delay)):
        parser.error("limit-per-teacher must be positive and delay finite and nonnegative")
    try:
        if args.command == "filter-zero-seed":
            print(json.dumps({"event": "filtered_seed_zero", **filter_zero_seed_index(args.out)}), flush=True)
            return
        api = kaggle_api()
        if args.command == "discover":
            discover(api, args.competition, args.top_teams, args.out)
        else:
            teachers = args.teachers
            if args.submission:
                snapshot = {"schema": "public-bc-explicit-teachers-v1", "competition": "kaggriculture",
                            "teachers": [{"submission_id": submission} for submission in args.submission]}
                teachers = args.out / "requested-teachers.json"
                if teachers.exists() and json.loads(teachers.read_text()) != snapshot:
                    raise ValueError("explicit teacher IDs changed; use a new output directory")
                write_json(teachers, snapshot)
            download(api, teachers, args.out, args.limit_per_teacher, args.delay)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
