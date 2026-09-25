"""Paired 719-action Rust-engine evaluation against the frozen baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .paths import BASELINE
from .training import Network, Serve, episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "random", "checkpoint"),
                        default="baseline")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9001, 9002])
    parser.add_argument("--takeover", type=int, default=144)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    model, torch = None, None
    if args.mode == "checkpoint":
        if args.checkpoint is None:
            parser.error("checkpoint mode requires --checkpoint")
        import torch
        model = Network(torch)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("engine") != "kaggle-environments==1.32.7":
            raise ValueError("checkpoint engine pin does not match")
        baseline_hash = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
        if checkpoint.get("baseline_sha256") != baseline_hash:
            raise ValueError("checkpoint was trained against a different baseline")
        model.module.load_state_dict(checkpoint["network"])
        model.module.eval()
    results = []
    with Serve() as srv:
        for seed in args.seeds:
            for seat in (0, 1):
                _, banks, stats = episode(srv, model, torch, seed, seat, "cpu",
                    training=args.mode == "random", takeover=args.takeover)
                results.append({"seed": seed, "seat": seat, "own_cash": banks[0],
                                "baseline_cash": banks[1],
                                "margin": banks[0] - banks[1], "routes": stats})
                print(json.dumps(results[-1]), flush=True)
    report = {"engine": "kaggle-environments==1.32.7", "mode": args.mode,
              "baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
              "games": results,
              "mean_margin": sum(r["margin"] for r in results) / len(results),
              "wins": sum(r["margin"] > 0 for r in results)}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"mean_margin": report["mean_margin"],
                      "wins": report["wins"], "games": len(results)}), flush=True)


if __name__ == "__main__":
    main()
