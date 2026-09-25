"""Execute all workers' learned routes; release dependencies after receipts."""
from __future__ import annotations
import copy
from dataclasses import asdict
from .routes import PlanningState, command_toward, farm_of, positions, legal, project, tile_at

def receipt_matches(before, after, job):
    expected = copy.deepcopy(before)
    project(expected, job)
    op = job.op[0]
    ap, ep = after["private"], expected["private"]
    if op == "HIRE":
        return len(farm_of(after)["hands"]) == len(farm_of(before)["hands"]) + 1
    if op == "BUY_LAND":
        return len(farm_of(after)["unlocked_quadrants"]) == len(farm_of(before)["unlocked_quadrants"]) + 1
    if op == "BUY_SEED":
        return ap["seeds"].get(job.op[1], 0) == ep["seeds"].get(job.op[1], 0)
    if op in ("BUY_ANIMAL", "BUY_PRODUCT", "SELL"):
        item = job.op[1]
        stock_ok = ap["shed"].get(item, 0) == ep["shed"].get(item, 0)
        money = farm_of(after)["money"] - farm_of(before)["money"]
        return stock_ok and (money > 0 if op == "SELL" else money < 0)
    if op in ("PICKUP", "DROP", "HARVEST", "COLLECT_FERTILIZER"):
        return {k: v for k, v in ap["inventories"][job.actor].items() if v} == {
            k: v for k, v in ep["inventories"][job.actor].items() if v}
    if op == "WAIT":
        return True
    tile, want = tile_at(after, job.pos), tile_at(expected, job.pos)
    if op == "DIG":
        return tile is None
    if not isinstance(tile, dict) or not isinstance(want, dict):
        return False
    keys = {"PLANT": ("crop", "planted_day"), "PLACE": ("animal", "placed_day"),
            "WATER": ("watered_today",), "FEED": ("fed_today",),
            "CARE": ("cared_today",), "FERTILIZE": ("fertilized_until_day",)}.get(op, ("kind",))
    return all(tile.get(k) == want.get(k) for k in keys)

class RouteController:
    def __init__(self, baseline=None, choose=None, takeover=0, horizon=24,
                 replan_interval=6, max_jobs_per_worker=24, trace=None):
        if not 1 <= horizon <= 24 or replan_interval < 1 or max_jobs_per_worker < 1:
            raise ValueError("horizon must be 1..24; planning limits must be positive")
        self.trace = trace
        self.baseline, self.choose = baseline, choose or (lambda plan, jobs: 0)
        self.takeover, self.horizon = takeover, horizon
        self.replan_interval, self.max_jobs_per_worker = replan_interval, max_jobs_per_worker
        self.plan, self.done, self.pending = None, set(), []
        self.last_plan_step, self.day = -1000, -1
        self.last_decisions = []
        self.stats = dict(decisions=0, route_starts=0, tasks=0, aborts=0,
                          purchases=0, sales=0, replans=0, waits=0)

    def _emit(self, obs, event, **data):
        if self.trace is not None:
            self.trace(dict(step=obs["step"], day=obs["day"], event=event, **data))

    def _build(self, obs):
        self.plan, self.done = PlanningState(obs, self.horizon), set()
        self.last_plan_step = obs["step"]
        self.stats["replans"] += 1
        n = len(positions(obs))
        # Round-robin construction permits cross-worker funding and handoffs.
        for _ in range(self.max_jobs_per_worker):
            for actor in range(n):
                if actor in self.plan.ended:
                    continue
                jobs = self.plan.candidates(actor)
                idx = int(self.choose(self.plan, jobs))
                if not 0 <= idx < len(jobs):
                    raise ValueError("policy chose invalid job index")
                self.last_decisions.append(jobs[idx])
                self.stats["decisions"] += 1
                self.plan.commit(jobs[idx])
            if len(self.plan.ended) == n:
                break
        self.stats["route_starts"] += len({j.actor for j in self.plan.jobs})
        self._emit(obs, "plan", cash=farm_of(obs)["money"],
                   estimated_sale_credit=self.plan.sale_credit,
                   jobs=[asdict(j) for j in self.plan.jobs], ended=sorted(self.plan.ended))

    def act(self, obs, configuration=None):
        self.last_decisions = []
        if obs["step"] < self.takeover:
            if self.baseline is None:
                raise ValueError("warm-up requires a baseline")
            return self.baseline(obs, configuration)
        changed_day = self.day != obs["day"]
        self.day = obs["day"]
        invalid = False
        for before, job in self.pending:
            if changed_day:
                continue  # Daily reset unloads bags and removes hired hands.
            if receipt_matches(before, obs, job):
                self.done.add(job.id)
                self.stats["tasks"] += 1
                self._emit(obs, "receipt", job=asdict(job), success=True,
                           actual_cash=farm_of(obs)["money"])
                if job.op[0] in ("HIRE", "BUY_LAND"):
                    invalid = True
            else:
                invalid = True
                self.stats["aborts"] += 1
                self._emit(obs, "receipt", job=asdict(job), success=False,
                           actual_cash=farm_of(obs)["money"])
        self.pending = []
        if (self.plan is None or changed_day or invalid or
                obs["step"] - self.last_plan_step >= self.replan_interval or
                (self.plan.jobs and len(self.done) == len(self.plan.jobs))):
            self._build(obs)
        action = {"farmer": ["PASS"], "hands": [["PASS"] for _ in farm_of(obs)["hands"]], "market": []}
        heads = {}
        for job in self.plan.jobs:
            if job.id not in self.done:
                heads.setdefault(job.actor, job)
        for actor, job in heads.items():
            if actor >= len(positions(obs)):
                self.plan = None
                continue
            move = None if job.market else command_toward(positions(obs)[actor], job.pos)
            if move is not None:
                if actor == 0:
                    action["farmer"] = move
                else:
                    action["hands"][actor - 1] = move
                continue
            if not set(job.deps) <= self.done:
                self.stats["waits"] += 1
                continue
            if not legal(obs, job):
                self.stats["aborts"] += 1
                self._emit(obs, "invalid", job=asdict(job), actual_cash=farm_of(obs)["money"])
                self.plan = None
                break
            if job.market:
                action["market"].append(list(job.op))
                self.stats["sales" if job.op[0] == "SELL" else "purchases"] += 1
            elif actor == 0:
                action["farmer"] = ["PASS"] if job.op[0] == "WAIT" else list(job.op)
            else:
                action["hands"][actor - 1] = ["PASS"] if job.op[0] == "WAIT" else list(job.op)
            self.pending.append((copy.deepcopy(obs), job))
        self._emit(obs, "action", action=action)
        return action
