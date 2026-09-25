"""Paired 719-action Rust-engine evaluation against the frozen baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

from .paths import BASELINE
from .training import Network, Serve, episode
from .features import SCHEMA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "random", "checkpoint"),
                        default="baseline")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9001, 9002])
    parser.add_argument("--takeover", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--replan-interval", type=int, default=None)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    model, torch = None, None
    if args.mode == "checkpoint":
        if args.checkpoint is None:
            parser.error("checkpoint mode requires --checkpoint")
        import torch
        torch.set_num_threads(1)
        model = Network(torch)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != SCHEMA:
            raise ValueError("legacy route-template checkpoint; retrain for dynamic-routes-v2")
        for key in ("takeover", "horizon", "replan_interval"):
            if getattr(args, key) is None:
                setattr(args, key, checkpoint[key])
        if checkpoint.get("engine") != "kaggle-environments==1.32.7":
            raise ValueError("checkpoint engine pin does not match")
        baseline_hash = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
        if checkpoint.get("baseline_sha256") != baseline_hash:
            raise ValueError("checkpoint was trained against a different baseline")
        model.module.load_state_dict(checkpoint["network"])
        model.module.eval()
    args.takeover = 0 if args.takeover is None else args.takeover
    args.horizon = 24 if args.horizon is None else args.horizon
    args.replan_interval = 6 if args.replan_interval is None else args.replan_interval
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with Serve() as srv:
        for seed in args.seeds:
            for seat in (0, 1):
                output = (args.trace_dir / f"{args.mode}_{seed}_seat{seat}.jsonl").open(
                    "w", encoding="utf-8") if args.trace_dir else nullcontext(None)
                with output as trace_file:
                    trace = (lambda row: trace_file.write(json.dumps(row) + "\n")) if trace_file else None
                    _, banks, stats = episode(srv, model, torch, seed, seat, "cpu",
                        training=args.mode == "random", takeover=args.takeover,
                        horizon=args.horizon, replan_interval=args.replan_interval, trace=trace)
                results.append({"seed": seed, "seat": seat, "own_cash": banks[0],
                                "baseline_cash": banks[1],
                                "margin": banks[0] - banks[1], "routes": stats})
                print(json.dumps(results[-1]), flush=True)
    report = {"schema": SCHEMA, "takeover": args.takeover, "horizon": args.horizon,
              "replan_interval": args.replan_interval, "engine": "kaggle-environments==1.32.7", "mode": args.mode,
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
