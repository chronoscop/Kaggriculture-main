"""Optional, versioned final-day control with continuous public observer memory."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from ..full_action.catalog import UNIT_ACTION_TO_ID
from ..full_action.features import observation_step
from ..replay_rules import require_pinned_rules

CONTROLLER_CONTRACT = "owned-final-day-search-expected-score-v1"
OBJECTIVE = "public-margin-logistic-expected-terminal-score-v1"
ASSET_ROOT = Path(__file__).resolve().parents[1] / "season_search"
DEFAULTS = {"start_day": 29, "search_seconds": 0.25, "min_overage": 10.0,
            "reserve_seconds": 3.0, "uncertainty_scale": 2000.0,
            "objective": OBJECTIVE, "fallback": "neural"}


def validate_config(config):
    if config is None:
        return None
    if not isinstance(config, dict) or config.keys() - DEFAULTS.keys():
        raise ValueError("unknown season controller configuration")
    result = {**DEFAULTS, **config}
    if result["start_day"] != 29 or result["objective"] != OBJECTIVE or result["fallback"] != "neural":
        raise ValueError("season controller requires final-day expected-score search and neural fallback")
    for name in ("search_seconds", "min_overage", "reserve_seconds", "uncertainty_scale"):
        number = result[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
            raise ValueError(f"invalid season controller {name}")
    if result["search_seconds"] > 6 or result["uncertainty_scale"] < 1:
        raise ValueError("season search budget <= 6 seconds and uncertainty scale >= 1 are required")
    return result


def contract_identity(config):
    config = validate_config(config)
    if config is None:
        return None
    binary = ASSET_ROOT / "terminal_search.so"
    if not binary.exists():
        raise FileNotFoundError("build owned season search first: python tools/build_season_search.py")
    from ..season_search.search_policy import native_binary_identity
    _, binary_digest = native_binary_identity()
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(ASSET_ROOT.iterdir())
               if path.suffix in (".py", ".cpp", ".inc", ".json")}
    own = Path(__file__)
    sources["controllers.py"] = hashlib.sha256(own.read_bytes()).hexdigest()
    return {"contract": CONTROLLER_CONTRACT, "objective": OBJECTIVE, "config": config,
            "binary_sha256": binary_digest,
            "source_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}


def validate_action(action, observation):
    """Reject unsupported requests before handoff; never rewrite planner actions."""
    from .sampling import MARKET_ACTIONS
    count = 1 + len(observation["farms"][observation["player"]]["hands"])
    units = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    if len(units) != count or any(tuple(unit) not in UNIT_ACTION_TO_ID for unit in units):
        raise ValueError("planner unit request is outside the owned policy catalog")
    if len(action.get("market", [])) > 10 or any(tuple(order) not in MARKET_ACTIONS for order in action.get("market", [])):
        raise ValueError("planner market request is outside the owned policy catalog")


class FinalDayController:
    def __init__(self, config, observer_player):
        self.config = validate_config(config)
        self.observer_player = observer_player
        self.identity = contract_identity(config)
        self.planner = None
        self.started = False
        self.failed = False
        self.last_step = None
        self.diagnostics = {"handoff_actions": 0, "fallback_actions": 0, "last_error": None}
        if observer_player not in (0, 1):
            raise ValueError("one independent season controller per seat is required")

    def choose_action(self, observation, history=None):
        step = observation_step(observation)
        if self.last_step is not None and (step < self.last_step or (step == 0 and self.last_step >= 0)):
            self.planner = None
            self.started = self.failed = False
            self.diagnostics = {"handoff_actions": 0, "fallback_actions": 0, "last_error": None}
        self.last_step = step
        if self.config is None or int(observation["day"]) < self.config["start_day"]:
            return None
        if int(observation["player"]) != self.observer_player:
            raise ValueError("season controller cannot share seat state")
        if self.failed:
            self.diagnostics["fallback_actions"] += 1
            return None
        overage = float(observation.get("remainingOverageTime", 60.))
        if not self.started and (int(observation["hour"]) != 0 or overage < self.config["min_overage"]):
            self.failed = True
            self.diagnostics["fallback_actions"] += 1
            self.diagnostics["last_error"] = "insufficient budget or no dawn snapshot"
            return None
        try:
            require_pinned_rules()
            if observation.get("market", {}).get("params"):
                raise ValueError("custom market parameters are unsupported by owned native search")
            if history is None or history.player != self.observer_player:
                raise ValueError("season handoff requires continuous observer history")
            if self.planner is None:
                from ..season_search.search_policy import SearchPolicy
                self.planner = SearchPolicy(seconds=self.config["search_seconds"])
                self.planner.config["ponder"] = False
            # PolicyHistory.encode already advanced this observer on the actual
            # transition; no hidden opponent state or future seed is supplied.
            estimate = history.tracker.estimate()
            obs = deepcopy(observation)
            obs["_opponent_products"] = deepcopy(estimate.shed)
            obs["_opponent_carried_by_unit"] = deepcopy(history.tracker.carried_by_unit)
            obs["_opponent_overnight_fraction"] = {}  # irrelevant at day 29
            obs["_terminal_uncertainty"] = self.config["uncertainty_scale"]
            self.planner.seconds = min(self.config["search_seconds"], max(0., overage-self.config["reserve_seconds"])+.9)
            action = self.planner(obs)
            validate_action(action, observation)
        except Exception as error:
            self.failed = True
            self.diagnostics["fallback_actions"] += 1
            self.diagnostics["last_error"] = f"{type(error).__name__}: {error}"
            return None
        self.started = True
        self.diagnostics["handoff_actions"] += 1
        self.diagnostics["estimated_terminal_score"] = float(self.planner.header[9]) / 1_000_000.
        self.diagnostics["observation_step"] = observation_step(observation)
        return action
