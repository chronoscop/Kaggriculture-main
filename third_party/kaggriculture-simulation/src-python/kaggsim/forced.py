"""DEVELOPMENT USE ONLY: decouple the weed and shop RNG streams in the
official engine.

The interpreter seeds ONE ``random.Random((seed * 1_000_003) ^ day)`` per day
in its end-of-day step and uses it both for weed spawns (``.random()``, once
per EMPTY tile -- a count the players control) and for the town shop draw
(``.choice()``). So in an A/B where one build plants or digs differently from
the other, "same seed" does not mean "same world": the shop sequence moves
with the actions, and that noise can swamp small effects.

:func:`activate` patches the imported official module so that, inside its
end-of-day function, ``.random()`` stays on the engine's own stream (weeds
unchanged) while ``.choice()`` draws from an independent stream keyed
``key + 1``. ``forced_shops`` optionally pins the whole unlock sequence.

THIS CHANGES THE GAME. The fidelity tools (``kaggsim.official.make``,
``run_agents``, ``kaggsim.fidelity``, ``kaggsim.economy``) refuse to run
while the patch is active. Use :func:`run_agents` / :func:`make` from this
module for patched games. Activation is explicit and per process;
:func:`activate` returns an ``undo()`` and :func:`patched` is a context
manager.

    from kaggsim.forced import patched, run_agents
    with patched(forced_shops=["SHOP_A", "SHOP_B"]):
        banks, env = run_agents(agent_a, agent_b, seed=5)

    python -m kaggsim.forced --selftest
"""
from __future__ import annotations

import argparse
import contextlib
import random

from . import official


class _SplitRandom:
    def __init__(self, key, forced_shops, drawn):
        self._weeds = random.Random(key)
        self._shops = random.Random(key + 1)
        self._forced = forced_shops
        self._drawn = drawn

    def random(self):
        return self._weeds.random()

    def choice(self, seq):
        if self._forced is not None:
            i = self._drawn[0]
            self._drawn[0] += 1
            if i < len(self._forced) and self._forced[i] in seq:
                return self._forced[i]
        return self._shops.choice(seq)

    def __getattr__(self, name):
        return getattr(self._weeds, name)


def activate(forced_shops=None):
    """Patch the official module in THIS process. Returns ``undo()``."""
    mod = official._import_engine()
    official.verify(mod)            # only ever patch the pinned engine
    real_random = mod.random
    real_eod = mod._end_of_day
    drawn = [0]
    in_eod = [False]

    class _Proxy:
        def Random(self, key=None):  # noqa: N802 (mirrors random.Random)
            # Only the RNG created inside the end-of-day step is split.
            if in_eod[0]:
                return _SplitRandom(key, forced_shops, drawn)
            return real_random.Random(key)

        def __getattr__(self, name):
            return getattr(real_random, name)

    def _end_of_day(*args, **kwargs):
        in_eod[0] = True
        try:
            return real_eod(*args, **kwargs)
        finally:
            in_eod[0] = False

    mod.random = _Proxy()
    mod._end_of_day = _end_of_day

    def undo():
        mod.random = real_random
        mod._end_of_day = real_eod

    return undo


@contextlib.contextmanager
def patched(forced_shops=None):
    undo = activate(forced_shops)
    try:
        yield
    finally:
        undo()


def make(seed: int, **configuration):
    """``official.make`` that is allowed to run on the patched engine."""
    return official._make(seed, _allow_forced=True, **configuration)


def run_agents(agent0, agent1, seed: int, quiet: bool = True):
    """``official.run_agents`` that is allowed to run on the patched engine."""
    return official._run(agent0, agent1, seed, quiet, _allow_forced=True)


def _shops_after(busy: bool, patch: bool, forced=None, seed=99, steps=240):
    def agent(obs, cfg=None):
        planting = busy and obs["step"] % 3 == 0
        return {"farmer": ["PLANT", "WHEAT"] if planting else ["PASS"],
                "hands": [],
                "market": [["BUY_SEED", "WHEAT", 1]]
                if busy and obs["step"] < 40 else []}

    ctx = patched(forced) if patch else contextlib.nullcontext()
    with ctx:
        env = make(seed, episodeSteps=steps, actTimeout=60, runTimeout=100000)
        env.run([agent, agent])
        return list(env.state[0].observation.town["unlocked_shops"])


def selftest() -> bool:
    base = _shops_after(False, False)
    moved = _shops_after(True, False)
    split_idle = _shops_after(False, True)
    split_busy = _shops_after(True, True)
    pin = ["BAKERY", "YARN_STORE", "PET_CAFE"]
    pinned = _shops_after(True, True, forced=pin)
    print(f"unpatched idle  : {base}\nunpatched busy  : {moved}")
    print(f"split idle      : {split_idle}\nsplit busy      : {split_busy}")
    print(f"pinned {pin}: {pinned}")
    ok = split_idle == split_busy and pinned[:len(pin)] == pin[:len(pinned)]
    print(f"coupling visible unpatched: {base != moved}; split stable: "
          f"{split_idle == split_busy}; pin honoured: "
          f"{pinned[:len(pin)] == pin[:len(pinned)]}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kaggsim.forced",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return 0 if selftest() else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
