"""Generate labelled self-play data across worlds, with a custom feature
extractor plugged in through a command sink.

    python scripts/examples/selfplay_example.py

Writes ``selfplay/example/samples.jsonl`` (Rust features + labels, via the
default sink) and ``selfplay/example/custom.jsonl`` (the same samples after
the custom Python extractor in ``custom_features.py``).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src-python"))
os.environ["PYTHONPATH"] = os.pathsep.join(
    [HERE] + [p for p in [os.environ.get("PYTHONPATH")] if p])

from kaggsim.selfplay import run_selfplay  # noqa: E402

out_dir = os.environ.get("OUT", "selfplay")
summary = run_selfplay({
    "name": "example",
    "agent": {"name": "me", "type": "builtin", "kind": "random",
              "seed": "per_game"},
    "opponents": {"mode": "pool", "p_mirror": 0.5, "rng_seed": 1,
                  "pool": [{"name": "chaos", "type": "builtin",
                            "kind": "chaos", "seed": "per_game"}]},
    "worlds": {"strategy": "stratified", "pool": [0, 1000], "per_world": 2,
               "key_depth": 2},
    "seats": "both",
    "workers": int(os.environ.get("WORKERS", "2")),
    "samples": {"features": ["time", "money", "market", "shed", "tiles"],
                "include_obs": True, "stride": 6,
                "labels": ["outcome", "margin", "return_to_go"],
                "gamma": 0.995, "seats": "all"},
    "sinks": [
        {"type": "jsonl", "path": os.path.join(out_dir, "example",
                                               "samples.jsonl"),
         "records": ["game", "sample"]},
        {"type": "command", "records": ["sample"],
         "argv": [sys.executable, "-m", "kaggsim.processor",
                  "--features", "custom_features:extract",
                  "--keep", "game_id,seed,step,seat,world,features,labels",
                  "--out", os.path.join(out_dir, "example",
                                        "custom.jsonl")]},
    ],
    "output": {"dir": out_dir, "resume": False},
})
print(json.dumps(summary, indent=1))
