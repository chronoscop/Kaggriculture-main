"""Commit one local work route while the incumbent runs the remaining farm."""
from __future__ import annotations

from .routes import Route, command_toward, generate, valid_stop


class RouteController:
    def __init__(self, baseline, choose=None, takeover=144, max_routes=24,
                 max_routes_per_day=2):
        self.baseline = baseline
        self.choose = choose or (lambda obs, routes: 0)
        self.takeover = takeover
        self.max_routes = max_routes
        self.max_routes_per_day = max_routes_per_day
        self.active: Route | None = None
        self.index = 0
        self.pending_buy = False
        self.receipt = None
        self.stalls = 0
        self.pending_sale = None
        self.last_decision = None
        self.next_decision_step = takeover
        self.last_signal = None
        self.route_day = -1
        self.routes_today = 0
        self.stats = {"decisions": 0, "route_starts": 0, "tasks": 0,
                      "aborts": 0, "purchases": 0, "sales": 0}

    @staticmethod
    def _unit_action(action, actor, command):
        if actor == 0:
            action["farmer"] = list(command)
        else:
            while len(action["hands"]) < actor:
                action["hands"].append([])
            action["hands"][actor - 1] = list(command)

    @staticmethod
    def _append_order(action, order):
        if any(row and row[:2] == list(order[:2]) for row in action["market"]):
            return True
        if len(action["market"]) >= 10:
            return False
        action["market"].append(list(order))
        return True

    def _finish_sale(self, obs, action):
        if self.pending_sale is None:
            return 0.0
        item = self.pending_sale
        held = obs["private"]["shed"].get(item, 0)
        credit = 0.0
        if held and self._append_order(action, ("SELL", item, held)):
            self.stats["sales"] += 1
            amount = min(held, sum(row[2] for row in action["market"]
                                   if len(row) >= 3 and row[:2] == ["SELL", item]))
            credit = 0.4 * amount * obs["market"]["prices"].get(item, 0)
        self.pending_sale = None
        return credit

    def _abort(self):
        self.stats["aborts"] += 1
        self.active = None
        self.index = 0
        self.pending_buy = False
        self.receipt = None
        self.stalls = 0

    def _signal(self, obs):
        farm = obs["farms"][obs["player"]]
        return (float(farm["money"]),
                sum(tile is None for row in farm["tiles"] for tile in row),
                len(farm["unlocked_quadrants"]),
                len(obs["town"].get("unlocked_shops", [])))

    def act(self, obs, configuration=None):
        self.last_decision = None
        base = self.baseline(obs, configuration)
        action = {"farmer": list(base.get("farmer") or ["PASS"]),
                  "hands": [list(v) for v in base.get("hands", [])],
                  "market": [list(v) for v in base.get("market", [])]}
        signal = self._signal(obs)
        previous = self.last_signal
        self.last_signal = signal
        event = previous is not None and (signal[0] - previous[0] >= 100 or
            signal[1] > previous[1] or signal[2] > previous[2] or signal[3] > previous[3])
        if obs["step"] < self.takeover:
            return action
        if obs["day"] != self.route_day:
            self.route_day = obs["day"]
            self.routes_today = 0
        cash_credit = self._finish_sale(obs, action)

        if self.receipt is not None:
            typ, item, _ = self.receipt
            inventory = obs["private"]["seeds"] if typ == "BUY_SEED" else obs["private"]["shed"]
            if inventory.get(item, 0) <= 0:
                self._abort()
            self.receipt = None

        if self.active is not None and self.active.actor > len(obs["farms"][obs["player"]]["hands"]):
            self._abort()
        if self.active is None:
            if (self.routes_today >= self.max_routes_per_day or
                    (obs["step"] < self.next_decision_step and not event)):
                return action
            routes = generate(obs, self.max_routes, cash_credit)
            if len(action["market"]) >= 10:
                routes = [route for route in routes if route.buy is None]
            chosen = int(self.choose(obs, routes))
            if chosen < 0 or chosen >= len(routes):
                raise ValueError("policy chose an invalid route index")
            self.last_decision = (routes, chosen)
            self.stats["decisions"] += 1
            self.next_decision_step = obs["step"] + 6
            if chosen:
                route = routes[chosen]
                self.active = route
                self.index = 0
                self.pending_buy = route.buy is not None
                self.stalls = 0
                self.stats["route_starts"] += 1
                self.routes_today += 1

        if self.active is None:
            return action
        route = self.active
        if self.pending_buy:
            if not self._append_order(action, route.buy):
                self._abort()
                return action
            self.pending_buy = False
            self.receipt = route.buy
            self.stats["purchases"] += 1
            # Market orders settle after unit actions; the new item cannot be
            # consumed in this turn. Let the incumbent work for one turn.
            return action

        farm = obs["farms"][obs["player"]]
        positions = [farm["farmer"]] + farm["hands"]
        current = tuple(positions[route.actor])
        while self.index < len(route.stops):
            stop = route.stops[self.index]
            move = command_toward(current, stop.pos)
            if move is not None:
                self._unit_action(action, route.actor, move)
                return action
            if stop.op[0] == "DROP":
                priv = obs["private"]
                bag = priv["inventories"][route.actor]
                free = 100 - sum(priv["shed"].values())
                if sum(bag.values()) > free:
                    self.stalls += 1
                    stock = sorted(priv["shed"].items(),
                                   key=lambda item: -item[1] * obs["market"]["prices"].get(item[0], 0))
                    for item, amount in stock:
                        if amount and self._append_order(action, ("SELL", item, amount)):
                            break
                    self._unit_action(action, route.actor, ("PASS",))
                    if self.stalls > 3:
                        self._abort()
                    return action
                self.stalls = 0
            if valid_stop(obs, route, self.index):
                self._unit_action(action, route.actor, stop.op)
                self.stats["tasks"] += 1
                if stop.op[0] == "DROP" and route.sell:
                    self.pending_sale = route.sell
                self.index += 1
                if self.index == len(route.stops):
                    self.active = None
                return action
            # A changing world can invalidate one stop without invalidating
            # the rest (another hand harvested, plant was already watered).
            if stop.op[0] in ("BUILD_COOP", "BUILD_PASTURE", "PICKUP", "PLACE"):
                self._abort()
                return action
            self.index += 1
        self._abort()
        return action
