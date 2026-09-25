"""Run a small tournament: your agent file vs a panel, across worlds.

    python scripts/examples/tournament_example.py [my_agent.py]

Without an argument a built-in fixture policy stands in for "your agent".
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "src-python"))

from kaggsim.tournament import run_tournament  # noqa: E402

candidate = ({"name": "mine", "type": "python",
              "path": os.path.abspath(sys.argv[1])} if len(sys.argv) > 1
             else {"name": "mine", "type": "pypolicy", "kind": "scripted",
                   "seed": 1})
summary = run_tournament({
    "name": "example",
    "candidate": candidate,
    "panel": [
        {"name": "random", "type": "builtin", "kind": "random", "seed": 2},
        {"name": "chaos", "type": "builtin", "kind": "chaos", "seed": 3},
        {"name": "farmer", "type": "pypolicy", "kind": "scripted", "seed": 9},
    ],
    "schedule": "gauntlet",
    "seats": "both",
    "worlds": {"strategy": "stratified", "pool": [0, 600], "per_world": 1,
               "key_depth": 1},
    "world_weighting": "uniform",
    "workers": int(os.environ.get("WORKERS", "2")),
    "output": {"dir": os.environ.get("OUT", "tournaments"), "resume": False},
})
focus = summary["focus"]
print(json.dumps({"games": summary["games"],
                  "games_per_sec": summary["games_per_sec"],
                  "score": focus["overall"]["score"],
                  "ci95": focus["overall"]["ci95"],
                  "world_weighted": focus["world_weighted_score"],
                  "by_opponent": {k: v["score"] for k, v in
                                  focus["by_opponent"].items()},
                  "out_dir": summary["out_dir"]}, indent=1))
