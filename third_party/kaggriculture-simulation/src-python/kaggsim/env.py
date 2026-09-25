"""A minimal gym-style environment over ``kagg serve``.

    from kaggsim.env import Env
    with Env() as env:
        obs0, obs1 = env.reset(seed=3)
        done = False
        while not done:
            (obs0, obs1), (r0, r1), done, info = env.step(a0, a1)

Observations are the per-seat views an agent would receive (plain dicts,
each seat sees only its own ``private``). The reward is the change in the
seat's bank since the previous step, so the undiscounted return equals
``final bank - 3000``. ``done`` becomes true at the official terminal step
(719): exactly 719 actions per episode. ``info`` carries the full state and
both banks.
"""
from __future__ import annotations

from .constants import FINAL_STEP
from .serve import Serve, seat_view


class Env:
    def __init__(self, kagg: str | None = None):
        self._srv = Serve(kagg)
        self._state = None

    def reset(self, seed: int):
        self._state = self._srv.reset(seed)
        return seat_view(self._state, 0), seat_view(self._state, 1)

    def step(self, action0, action1):
        if self._state is None:
            raise RuntimeError("call reset() first")
        prev = [f["money"] for f in self._state["farms"]]
        if self._state["step"] < FINAL_STEP:
            self._state = self._srv.step2(action0, action1)
        st = self._state
        banks = [float(f["money"]) for f in st["farms"]]
        rewards = (banks[0] - prev[0], banks[1] - prev[1])
        done = st["step"] >= FINAL_STEP
        info = {"state": st, "banks": tuple(banks), "step": st["step"]}
        return (seat_view(st, 0), seat_view(st, 1)), rewards, done, info

    @property
    def state(self):
        return self._state

    def close(self):
        self._srv.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
