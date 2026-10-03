"""Paired candidate matches using the pinned official rules and own policy worker."""
from contextlib import ExitStack
import json
from pathlib import Path
import select
import subprocess
import sys

from ..replay_rules import require_pinned_rules


class AgentProcess:
    def __init__(self, kind: str, path: Path):
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-m", "route_rl.full_action.agent_worker", kind, str(path.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        try:
            if not select.select([self.process.stdout], [], [], 120)[0]:
                raise TimeoutError("agent initialization did not finish")
            line = self.process.stdout.readline()
            if not line or json.loads(line) != {"ready": True}:
                raise RuntimeError("agent initialization failed; see stderr")
        except Exception:
            self.close()
            raise

    def __call__(self, observation, configuration):
        self.process.stdin.write(json.dumps({"observation": observation, "configuration": configuration}) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 120)[0]:
            raise TimeoutError("agent process did not respond")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"agent process exited with status {self.process.poll()}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["action"]

    def close(self):
        self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


def evaluate_games(policy: Path, opponent: Path, seed_start: int, games: int, output: Path) -> list[dict]:
    require_pinned_rules()
    from kaggle_environments import make

    records = []
    for seed in range(seed_start, seed_start + games // 2):
        for seat in (0, 1):
            with ExitStack() as stack:
                workers = [AgentProcess("policy", policy), AgentProcess("entry", opponent)]
                for worker in workers:
                    stack.callback(worker.close)
                if seat:
                    workers.reverse()

                def first(observation, configuration):
                    return workers[0](observation, configuration)

                def second(observation, configuration):
                    return workers[1](observation, configuration)

                environment = make("kaggriculture", configuration={"seed": seed}, debug=False)
                environment.run([first, second])
                last = environment.steps[-1]
                records.append({"seed": seed, "a_seat": seat, "turns": len(environment.steps),
                                "rewards": [state.reward for state in last],
                                "statuses": [state.status for state in last]})
                output.write_text(json.dumps(records, indent=2) + "\n")
                print(json.dumps({"event": "evaluation_game", **records[-1]}), flush=True)
    return records
