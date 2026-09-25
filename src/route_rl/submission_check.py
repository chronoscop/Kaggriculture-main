"""Run an exported archive in a relocated, one-core, no-site-packages process."""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import select
import subprocess
import sys
import tarfile
import tempfile

from .submission import MAX_ARCHIVE, MAX_DISK
from .paths import BASELINE, add_kaggsim

MAX_RAM = int(6.5 * 1024 ** 3)

def unpack(archive, root):
    archive, root = Path(archive), Path(root)
    if archive.stat().st_size > MAX_ARCHIVE:
        raise ValueError("archive exceeds 100 MiB")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        total = sum(m.size for m in members)
        names = [m.name for m in members]
        if total > MAX_DISK or len(names) != len(set(names)):
            raise ValueError("oversized archive or duplicate entries")
        if "main.py" not in names:
            raise ValueError("main.py must be at the archive root")
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("archive must contain only safe relative regular files")
            target = root.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    return total

class Probe:
    def __init__(self, main_path, cwd):
        script = Path(__file__).with_name("submission_probe.py")
        self.process = subprocess.Popen([sys.executable, "-I", "-S", "-u",
            str(script), str(main_path)], cwd=cwd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        try:
            self.startup = self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        ready, _, _ = select.select([self.process.stdout], [], [], 10)
        if not ready:
            raise TimeoutError("submission process did not respond within 10 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("submission process exited: " + self.process.stderr.read()[-4000:])
        return json.loads(line)

    def act(self, observation, stress=False):
        self.process.stdin.write(json.dumps({"observation": observation, "stress": stress}) + "\n")
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def check(archive, seeds=(9103,), max_action_seconds=1.0):
    add_kaggsim()
    from kaggsim.serve import Serve, obs_for, load_agent, call_agent
    archive = Path(archive).resolve()
    timings, games, startups = [], [], []
    peak, cutoffs = 0, 0
    with tempfile.TemporaryDirectory(prefix="route-submission-check-") as tmp:
        # Mimic the mounted layout, and launch from a DIFFERENT cwd.
        root = Path(tmp) / "kaggle_simulations" / "agent"
        root.mkdir(parents=True)
        unpacked = unpack(archive, root)
        with Serve() as server:
            for seed in seeds:
                for seat in (0, 1):
                    opponent = load_agent(str(BASELINE))
                    state = server.reset(seed)
                    with Probe(root / "main.py", tmp) as probe:
                        startups.append(probe.startup)
                        peak = max(peak, probe.startup["rss_bytes"])
                        while state["step"] < 719:
                            result = probe.act(obs_for(state, seat))
                            timings.append(result["seconds"])
                            peak = max(peak, result["rss_bytes"])
                            cutoffs += result["budget_cutoffs"]
                            own = result["action"]
                            other = call_agent(opponent, obs_for(state, 1 - seat))
                            state = server.step2(own, other) if seat == 0 else server.step2(other, own)
                        games.append(dict(seed=seed, seat=seat,
                            own_cash=state["farms"][seat]["money"],
                            opponent_cash=state["farms"][1-seat]["money"]))
            state = server.reset(seeds[0])
            observation = obs_for(state, 0)
            farm = observation["farms"][0]
            farm["money"] = 1000000
            farm["tiles"] = [[None for _ in range(10)] for _ in range(10)]
            farm["unlocked_quadrants"] = ["NW", "NE", "SW", "SE"]
            farm["hands"] = [[4, 4] for _ in range(24)]
            observation["private"]["inventories"] = [{} for _ in range(25)]
            observation["private"]["seeds"] = {c: 20 for c in
                ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")}
            with Probe(root / "main.py", tmp) as probe:
                stress = probe.act(observation, stress=True)
                peak = max(peak, stress["rss_bytes"])
                stress = {k: v for k, v in stress.items() if k != "action"}
                stress["description"] = "25 workers, 100 owned empty plots, forced legal purchases after full policy scoring"
    ordered = sorted(timings)
    report = dict(archive=str(archive), compressed_bytes=archive.stat().st_size,
        unpacked_bytes=unpacked, peak_agent_rss_bytes=peak,
        max_action_seconds=max(timings),
        p99_action_seconds=ordered[min(len(ordered)-1, int(len(ordered)*0.99))],
        planning_budget_cutoffs=cutoffs, startups=startups, games=games, synthetic_stress=stress,
        limits=dict(archive_bytes=MAX_ARCHIVE, disk_bytes=MAX_DISK,
                    ram_bytes=MAX_RAM, action_seconds=max_action_seconds),
        note="Local single-core measurement; target-platform smoke test still required.")
    report["passed"] = (peak < MAX_RAM and max(timings) < max_action_seconds and
                        stress["seconds"] < max_action_seconds and
                        all(not s["imported_heavy_libraries"] for s in startups))
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9103])
    parser.add_argument("--max-action-seconds", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = check(args.archive, args.seeds, args.max_action_seconds)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
