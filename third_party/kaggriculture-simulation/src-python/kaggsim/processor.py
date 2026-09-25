"""Plug-in processors for the ``kagg`` record stream.

``kagg tournament`` / ``kagg selfplay`` emit JSON lines (``"record":
"game"`` rows and ``"record": "sample"`` training samples) to their sinks.
A ``command`` sink pipes them into any program; this module is a ready-made
Python end of that pipe with pluggable feature extractors, label functions
and processors:

    {"type": "command", "records": ["sample"],
     "argv": ["python", "-m", "kaggsim.processor",
              "--features", "mypkg.feats:extract",   # fn(sample) -> dict
              "--labels", "mypkg.feats:labels",      # fn(sample) -> dict
              "--keep", "game_id,step,seat,world,labels",
              "--out", "data/train.jsonl"]}

or a whole custom processor class::

    "argv": ["python", "-m", "kaggsim.processor",
             "--processor", "mypkg.sink:NpzWriter", "--opt", "path=x.npz"]

A feature/label function receives the sample record (a dict with ``obs``
when the run sets ``samples.include_obs``, plus ``features``, ``labels``,
``action_line``, metadata) and returns a dict merged into
``features`` / ``labels``. :func:`basic_features` re-computes the Rust
feature set from an observation, for custom extractors that extend it.
"""
from __future__ import annotations

import argparse
import json
import sys

from ._util import resolve_ref as resolve
from .constants import PRODUCTS, SHOPS_SORTED


def basic_features(obs: dict) -> dict:
    """Python mirror of the Rust feature groups, from a per-seat observation
    (features are relative to ``obs["player"]``, whose private block the
    observation carries)."""
    me = obs["player"]
    opp = 1 - me
    farms = obs["farms"]
    f = {"step": obs["step"], "day": obs["day"], "hour": obs["hour"],
         "money_me": farms[me]["money"], "money_opp": farms[opp]["money"],
         "hands_me": len(farms[me]["hands"]),
         "hands_opp": len(farms[opp]["hands"]),
         "quadrants_me": len(farms[me]["unlocked_quadrants"]),
         "quadrants_opp": len(farms[opp]["unlocked_quadrants"])}
    for p in PRODUCTS:
        f[f"price_{p}"] = obs["market"]["prices"].get(p, 0)
        f[f"minv_{p}"] = obs["market"]["inventory"].get(p, 0)
        f[f"shed_{p}"] = obs["private"]["shed"].get(p, 0)
    for k, v in obs["private"]["seeds"].items():
        f[f"seeds_{k}"] = v
    f["carried_total"] = sum(sum(inv.values())
                             for inv in obs["private"]["inventories"])
    names = ["empty", "locked", "weed", "plant", "structure", "animal",
             "ripe"]
    for who, i in (("me", me), ("opp", opp)):
        c = dict.fromkeys(names, 0)
        for row in farms[i]["tiles"]:
            for t in row:
                if t is None:
                    c["empty"] += 1
                elif t == "LOCKED":
                    c["locked"] += 1
                elif t.get("kind") == "WEED":
                    c["weed"] += 1
                elif t.get("kind") == "PLANT":
                    c["plant"] += 1
                    c["ripe"] += t.get("yield_units", 0) > 0
                elif "animal" in t:
                    c["animal"] += 1
                else:
                    c["structure"] += 1
        for n in names:
            f[f"tiles_{n}_{who}"] = c[n]
    for s in SHOPS_SORTED:
        f[f"shop_{s}"] = obs["town"]["unlocked_shops"].count(s)
    return f


class Processor:
    """Base class: override any of the hooks."""

    def __init__(self, **options):
        self.options = options

    def on_game(self, record: dict):
        pass

    def on_sample(self, record: dict):
        pass

    def close(self):
        pass


class JsonlWriter(Processor):
    """Apply feature / label functions, keep chosen fields, write JSONL."""

    def __init__(self, out=None, features=None, labels=None, keep=None,
                 records="sample", drop_obs=True, **options):
        super().__init__(**options)
        self.fh = open(out, "w", encoding="utf-8") if out else sys.stdout
        self.features = resolve(features) if isinstance(features, str) \
            else features
        self.labels = resolve(labels) if isinstance(labels, str) else labels
        self.keep = [k for k in (keep.split(",") if isinstance(keep, str)
                                 else keep or []) if k]
        self.records = set(records.split(",")) if isinstance(records, str) \
            else set(records)
        self.drop_obs = drop_obs
        self.count = 0

    def _emit(self, rec):
        if self.keep:
            rec = {k: rec[k] for k in self.keep if k in rec}
        self.fh.write(json.dumps(rec) + "\n")
        self.count += 1

    def on_game(self, record):
        if "game" in self.records:
            self._emit(record)

    def on_sample(self, record):
        if "sample" not in self.records:
            return
        if self.features is not None:
            record.setdefault("features", {})
            if record["features"] is None:
                record["features"] = {}
            record["features"].update(self.features(record) or {})
        if self.labels is not None:
            record.setdefault("labels", {}).update(self.labels(record) or {})
        if self.drop_obs:
            record.pop("obs", None)
            record.pop("state", None)
        self._emit(record)

    def close(self):
        if self.fh is not sys.stdout:
            self.fh.close()
        else:
            self.fh.flush()


def run(processor: Processor, stream) -> dict:
    """Feed a record stream (iterable of JSON lines) to a processor."""
    games = samples = bad = 0
    try:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if rec.get("record") == "game":
                games += 1
                processor.on_game(rec)
            elif rec.get("record") == "sample":
                samples += 1
                processor.on_sample(rec)
    finally:
        processor.close()
    return {"games": games, "samples": samples, "bad_lines": bad}


def main(argv=None, stdin=None):
    ap = argparse.ArgumentParser(prog="python -m kaggsim.processor",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--processor", default=None,
                    help="module:Class (a Processor subclass)")
    ap.add_argument("--opt", action="append", default=[],
                    help="key=value option for the processor")
    ap.add_argument("--features", default=None, help="module:fn(sample)")
    ap.add_argument("--labels", default=None, help="module:fn(sample)")
    ap.add_argument("--keep", default=None, help="comma-separated fields")
    ap.add_argument("--records", default="sample", help="sample,game")
    ap.add_argument("--keep-obs", action="store_true")
    ap.add_argument("--out", default=None, help="output file (default stdout)")
    args = ap.parse_args(argv)
    bad = [o for o in args.opt if "=" not in o]
    if bad:
        ap.error(f"--opt needs key=value, got {bad}")
    opts = dict(o.split("=", 1) for o in args.opt)
    if args.processor:
        proc = resolve(args.processor)(**opts)
    else:
        if opts:
            ap.error("--opt is only for a custom --processor")
        proc = JsonlWriter(args.out, args.features, args.labels, args.keep,
                           args.records, not args.keep_obs)
    stats = run(proc, stdin or sys.stdin)
    print(json.dumps(stats), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
