"""Python agent host for the Rust runner (``kagg tournament`` / ``selfplay``).

``kagg`` starts ``python -m kaggsim.host`` per worker thread for the Python
agents it needs, and talks one JSON object per line over stdin/stdout. A
host holds up to two agents in *slots* (one per seat), so a Python-vs-Python
game costs one round trip per step:

    -> {"cmd": "ping"}                          <- {"ok": true, "pid": 123}
    -> {"cmd": "load", "slot": 0, "spec": {...}, "game_seed": 17,
        "echo": false}                          <- {"ok": true}
    -> {"cmd": "act", "slot": 0, "obs": {...}}  <- {"line": "...", "ms": 0.8}
    -> {"cmd": "act2", "obs": [{...}, {...}]}   <- {"lines": [..], "ms": [..]}
    -> {"cmd": "quit"}

``load`` builds a FRESH agent for the next game (a submission file is
re-executed in a new namespace, so no module state leaks between games).
Agents are called exactly as the official runner calls them: a freshly
built attribute-accessible observation (the message is decoded straight
into ``Struct`` objects, so each call gets its own copy) and the resolved
default ``configuration`` for 2-argument agents. Replies carry the action
as a positional tape line (``kaggsim.tape.action_to_line``); with ``echo``
the action dict too. Agent output on stdout is redirected to stderr, so it
cannot corrupt the protocol. Errors: ``{"error": "...", "slot": n}``.

Spec types handled here: ``python`` (``path``), ``factory`` (``ref``,
``kwargs``), ``pypolicy`` (``kind``, ``seed``).
"""
from __future__ import annotations

import contextlib
import inspect
import json
import os
import sys
import time
import traceback

from ._util import resolve_ref
from .policies import make_policy
from .serve import Struct, call_agent, load_agent
from .tape import action_to_line


def _struct_hook(d):
    """json object_hook building official-style Structs in one pass."""
    s = Struct.__new__(Struct)
    entries = {k: v for k, v in d.items() if k != "items"}
    dict.update(s, entries)
    s.__dict__.update(entries)
    return s


def loads_struct(text: str):
    """Parse JSON straight into Struct / list objects (a fresh copy)."""
    return json.loads(text, object_hook=_struct_hook)


def build_agent(spec: dict, game_seed: int):
    t = spec.get("type")
    if t == "python":
        return load_agent(spec["path"])
    if t == "pypolicy":
        seed = spec.get("seed", 0)
        if seed == "per_game":
            seed = int(game_seed) * 7919 + 13
        return make_policy(spec["kind"], int(seed))
    if t == "factory":
        fn = resolve_ref(spec["ref"])
        kwargs = dict(spec.get("kwargs") or {})
        try:
            if "game_seed" in inspect.signature(fn).parameters:
                kwargs["game_seed"] = game_seed
        except (TypeError, ValueError):
            pass
        agent = fn(**kwargs)
        if not callable(agent):
            raise TypeError(f"factory {spec['ref']} did not return a callable")
        return agent
    raise ValueError(f"host cannot run agent type {t!r}")


def _jsonable(action):
    return json.loads(json.dumps(action, default=str))


class Host:
    def __init__(self):
        self.agents = {}
        self.echo = {}

    def load(self, msg):
        slot = int(msg.get("slot", 0))
        self.agents.pop(slot, None)
        with contextlib.redirect_stdout(sys.stderr):
            self.agents[slot] = build_agent(msg["spec"],
                                            msg.get("game_seed", 0))
        self.echo[slot] = bool(msg.get("echo"))
        return {"ok": True}

    def _act(self, slot, obs):
        agent = self.agents.get(slot)
        if agent is None:
            raise RuntimeError(f"no agent loaded in slot {slot}")
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            action = call_agent(agent, obs)
        ms = (time.perf_counter() - t0) * 1000.0
        return action, round(ms, 4)

    def handle(self, msg):
        cmd = msg.get("cmd")
        if cmd == "ping":
            return {"ok": True, "pid": os.getpid()}
        if cmd == "load":
            return self.load(msg)
        if cmd == "act":
            slot = int(msg.get("slot", 0))
            try:
                action, ms = self._act(slot, msg["obs"])
            except (KeyboardInterrupt, GeneratorExit):
                raise
            except BaseException as exc:  # noqa: BLE001  incl. SystemExit
                return {"error": _err(exc), "slot": slot}
            out = {"line": action_to_line(action), "ms": ms}
            if self.echo.get(slot):
                out["action"] = _jsonable(action)
            return out
        if cmd == "act2":
            lines, mss, acts = [], [], []
            for slot, obs in enumerate(msg["obs"]):
                try:
                    action, ms = self._act(slot, obs)
                except (KeyboardInterrupt, GeneratorExit):
                    raise
                except BaseException as exc:  # noqa: BLE001  incl. SystemExit
                    return {"error": _err(exc), "slot": slot}
                lines.append(action_to_line(action))
                mss.append(ms)
                acts.append(_jsonable(action) if self.echo.get(slot)
                            else None)
            out = {"lines": lines, "ms": mss}
            if any(self.echo.get(s) for s in (0, 1)):
                out["actions"] = acts
            return out
        return {"error": f"unknown cmd {cmd!r}"}


def _err(exc):
    tb = traceback.format_exc(limit=3)
    return f"{type(exc).__name__}: {exc} | {tb[-300:]}"


def serve(stdin=None, stdout=None):
    stdin = stdin or sys.stdin
    out = stdout or sys.stdout
    host = Host()
    for raw in stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            # Observations become attribute-accessible Structs in one pass;
            # other messages (e.g. load specs with kwargs) stay plain dicts.
            msg = json.loads(raw)
            if msg.get("cmd") in ("act", "act2"):
                msg = loads_struct(raw)
        except (ValueError, AttributeError) as exc:
            reply = {"error": f"bad request: {exc}"}
        else:
            if msg.get("cmd") == "quit":
                break
            try:
                reply = host.handle(msg)
            except (KeyboardInterrupt, GeneratorExit):
                raise
            except BaseException as exc:  # noqa: BLE001
                reply = {"error": _err(exc)}
        out.write(json.dumps(reply) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
