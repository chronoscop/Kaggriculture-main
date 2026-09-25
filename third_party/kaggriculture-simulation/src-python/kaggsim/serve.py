"""Client for ``kagg serve`` and a match runner with official-runner fidelity.

The engine steps in Rust; agents run in this process as ordinary Python, and
are invoked exactly the way the official ``kaggle_environments`` runner
invokes them:

* **Arity and configuration.** An agent is called with ``(obs, configuration)`` truncated
  to its ``co_argcount``; 2-argument agents get a fresh attribute-accessible
  ``configuration`` holding the resolved defaults.
* **Per-seat isolation.** Each seat gets its own freshly built,
  deep-copied, attribute-accessible observation carrying only its OWN
  ``private`` block. Mutating it cannot leak into the other seat or the engine.
* **Episode length.** Actions are solicited for steps 0..718 only; the
  final banks are read from the step-719 state. (``kagg serve`` itself also
  refuses to step past 719.)

See ``docs/fidelity.md``.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

from .binary import find_kagg
from .constants import FINAL_STEP
from .official import DEFAULT_CONFIGURATION
from .tape import action_to_line


class Struct(dict):
    """dict with attribute access -- mirrors kaggle_environments' Struct."""

    def __init__(self, **entries):
        entries = {k: v for k, v in entries.items() if k != "items"}
        dict.__init__(self, entries)
        self.__dict__.update(entries)

    def __setattr__(self, attr, value):
        self.__dict__[attr] = value
        self[attr] = value


def structify(o):
    """Recursively rebuild dicts as :class:`Struct` (a deep copy)."""
    if isinstance(o, list):
        return [structify(v) for v in o]
    if isinstance(o, dict):
        return Struct(**{k: structify(v) for k, v in o.items()})
    return o


#: What the official runner reports to an agent that used no overage time.
REMAINING_OVERAGE_TIME = 60


def seat_view(state: dict, seat: int) -> dict:
    """The per-seat observation dict an agent receives (plain dicts)."""
    return copy.deepcopy({
        "remainingOverageTime": REMAINING_OVERAGE_TIME,
        "step": state["step"], "day": state["day"], "hour": state["hour"],
        "player": seat, "farms": state["farms"], "market": state["market"],
        "town": state["town"], "private": state["private"][seat]})


def obs_for(state: dict, seat: int):
    """Fresh, isolated, attribute-accessible observation for one seat."""
    return structify(seat_view(state, seat))


def call_agent(agent, obs):
    """Invoke like the official runner: args truncated to ``co_argcount``."""
    args = [obs, structify(DEFAULT_CONFIGURATION)]
    code = getattr(agent, "__code__", None)
    if code is not None and hasattr(code, "co_argcount"):
        args = args[: code.co_argcount]
    return agent(*args)


_CODE_CACHE: dict = {}


def _compiled(path: str):
    """Compiled code of an agent file, cached per process and invalidated
    when the file's size or modification time changes."""
    path = os.path.abspath(path)
    st = os.stat(path)
    key = (st.st_mtime_ns, st.st_size)
    hit = _CODE_CACHE.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    with open(path, encoding="utf-8") as fh:
        code = compile(fh.read(), path, "exec")
    _CODE_CACHE[path] = (key, code)
    return code


def load_agent(path: str):
    """Load a single-file submission's entry point with FRESH module state.

    The entry point is chosen exactly as the official runner chooses it
    (``kaggle_environments.agent.get_last_callable``): the source is executed
    in an empty namespace with the file's directory on ``sys.path``, and the
    LAST callable in that namespace, by insertion order, is the agent. That
    is not necessarily the function named ``agent``: re-binding an existing
    name keeps its original position, so a file that defines ``agent``,
    then helpers, then ``agent`` again is run by the ladder through its last
    helper.

    The source is read and compiled once per process (cached); every call
    executes it in a brand-new namespace, so no module-level state carries
    over between games.
    """
    ns: dict = {}
    exec_dir = os.path.dirname(os.path.abspath(path))
    sys.path.append(exec_dir)
    try:
        exec(_compiled(path), ns)
    finally:
        sys.path.remove(exec_dir)
    entry = [v for v in ns.values() if callable(v)]
    if not entry:
        raise ValueError(f"{path} defines no callable agent")
    return entry[-1]


class ServeError(RuntimeError):
    pass


class Serve:
    """One persistent ``kagg serve`` process; many episodes over its life."""

    def __init__(self, kagg: str | None = None):
        self.exe = find_kagg(kagg)
        self.proc = None

    def _ensure(self):
        if self.proc is not None and self.proc.poll() is not None:
            self._release()          # a dead process: close its pipes
        if self.proc is None:
            self.proc = subprocess.Popen(
                [self.exe, "serve"], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, text=True, encoding="utf-8",
                bufsize=1)

    def cmd(self, line: str) -> dict:
        """Send one request line; return the parsed JSON response."""
        self._ensure()
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
            resp = self.proc.stdout.readline()
        except OSError as exc:
            raise ServeError(f"kagg serve pipe failed: {exc}") from exc
        if not resp:
            raise ServeError("kagg serve closed the pipe")
        out = json.loads(resp)
        if isinstance(out, dict) and "error" in out:
            raise ServeError(out["error"])
        return out

    # -- protocol helpers --------------------------------------------------
    def reset(self, seed: int, opp_tape: str | None = None,
              opp_seat: int = 1) -> dict:
        if opp_tape:
            return self.cmd(f"RESET {int(seed)} OPP {int(opp_seat)} "
                            f"{os.path.abspath(opp_tape)}")
        return self.cmd(f"RESET {int(seed)}")

    def step(self, action) -> dict:
        return self.cmd("STEP " + action_to_line(action))

    def step2(self, action0, action1) -> dict:
        return self.cmd("STEP2 " + action_to_line(action0) + "\x1e"
                        + action_to_line(action1))

    def load_state(self, state: dict, seed: int | None = None) -> dict:
        st = dict(state)
        if seed is not None:
            st["seed"] = int(seed)
        return self.cmd("LOADSTATE " + json.dumps(st))

    def rollout(self, state: dict, horizon: int, lines0, lines1) -> dict:
        return self.cmd(f"ROLLOUT {int(horizon)} " + json.dumps(state)
                        + "\x1e" + "\x1f".join(_lines(lines0)) + "\x1e"
                        + "\x1f".join(_lines(lines1)))

    def gengame(self, seed: int, lines0, lines1) -> dict:
        """Whole game in one call: {"days": [...], "final": obs}."""
        return self.cmd(f"GENGAME {int(seed)}\x1e"
                        + "\x1f".join(_lines(lines0)) + "\x1e"
                        + "\x1f".join(_lines(lines1)))

    def _release(self):
        for fh in (self.proc.stdin, self.proc.stdout):
            try:
                fh.close()
            except OSError:
                pass
        self.proc = None

    def close(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        self._release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _lines(seq):
    return [a if isinstance(a, str) else action_to_line(a) for a in seq]


def run_match(agent0, agent1, seed: int, srv: Serve | None = None,
              record: bool = False, on_step=None):
    """Play two agent callables on the Rust engine.

    Returns ``(bank0, bank1)``, or ``(banks, trace)`` with ``record=True``
    where ``trace`` is a list of per-step dicts ``{step, actions, state}``
    (``state`` is the full post-step JSON state). ``on_step(pre_state,
    actions, post_state)``, if given, is called after every step with the
    full pre-step state, both actions and the post-step state.
    """
    own = srv is None
    srv = srv or Serve()
    trace = []
    try:
        js = srv.reset(seed)
        while js["step"] < FINAL_STEP:
            a0 = call_agent(agent0, obs_for(js, 0))
            a1 = call_agent(agent1, obs_for(js, 1))
            pre = js
            js = srv.step2(a0, a1)
            if on_step is not None:
                on_step(pre, (a0, a1), js)
            if record:
                trace.append({"step": js["step"], "actions": [
                    json.loads(json.dumps(a0)), json.loads(json.dumps(a1))],
                    "state": js})
        banks = (float(js["farms"][0]["money"]),
                 float(js["farms"][1]["money"]))
        return (banks, trace) if record else banks
    finally:
        if own:
            srv.close()


def run_files(path0: str, path1: str, seed: int, srv: Serve | None = None):
    """:func:`run_match` on two agent files, each freshly loaded."""
    return run_match(load_agent(path0), load_agent(path1), seed, srv)
