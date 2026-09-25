"""Self-play generation across worlds (front end for ``kagg selfplay``).

    from kaggsim.selfplay import run_selfplay
    run_selfplay({
        "name": "sp1",
        "agent": {"name": "me", "type": "python", "path": "main.py"},
        "opponents": {"mode": "pool", "p_mirror": 0.5,
                      "pool": [{"name": "rnd", "type": "builtin",
                                "kind": "random", "seed": "per_game"}]},
        "worlds": {"strategy": "stratified", "pool": [0, 5000],
                   "per_world": 25, "key_depth": 2},
        "workers": 8,
        "samples": {"features": ["time", "money", "market", "shed"],
                    "labels": ["outcome", "return_to_go"], "gamma": 0.99},
        "sinks": [{"type": "command", "records": ["sample"],
                   "argv": ["python", "-m", "kaggsim.processor",
                            "--features", "mypkg.feats:extract",
                            "--out", "data/sp1.jsonl"]}]})

    python -m kaggsim.selfplay template > sp.json
    python -m kaggsim.selfplay run sp.json --set workers=8
"""
from __future__ import annotations

from .tournament import main as _main
from .tournament import run_selfplay, template

__all__ = ["run_selfplay", "template", "main"]


def main(argv=None):
    return _main(argv, kind="selfplay")


if __name__ == "__main__":
    raise SystemExit(main())
