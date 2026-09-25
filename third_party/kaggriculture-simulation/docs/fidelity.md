# Fidelity: how "byte-identical" is proven

The Rust engine is a statement-order-preserving port of
`kaggle_environments/envs/kaggriculture/kaggriculture.py` from
`kaggle-environments==1.32.7` (see `docs/version-pinning.md`).

## The differential suite

`tests/differential/test_differential.py` (and
`python -m kaggsim.fidelity certify`) runs synthetic episodes:

1. The episode is driven on the OFFICIAL interpreter with `env.step`.
   Closed-loop policies read official observations. Every action is first
   normalised through the tape encoding, so both engines consume identical
   input.
2. The recorded streams are replayed through `kagg episode`.
3. After EVERY step a full-state digest is compared: both moneys as IEEE-754
   bit patterns; farmer and hand positions; hires; unlocked quadrants; every
   field of every tile (plants, weeds, structures, animals); both seats'
   shed, seeds and carried inventories; market inventory and prices; the
   unlocked shops. Final banks are compared as well.

Policies (`kaggsim.policies`): `chaos` (every op, legal or not, malformed
and empty orders, empty hand slots), `random` (observation-aware random
play), `scripted` (a closed-loop farmer with animals, fertilizer, hires and
land) and `last_turn` (acts on the final turns). Seat assignments vary.
Defaults: 50 episodes per run, 500 in the nightly CI job
(`KAGGSIM_DIFF_EPISODES`).

`tests/differential/test_primitives.py` also checks the RNG (per-day
seeding, `random()`, `getrandbits`, `choice`) against CPython and the rule
tables against the official module.

## Invocation rules a driver must follow

An identical engine is necessary but not sufficient: a driver must also call
agents the way the official runner does. `kaggsim.serve.run_match`
implements these rules, and `tests/regressions` pins each one:

| Rule | What the official runner does |
|---|---|
| Entry point | the source is executed in an empty namespace with its directory on `sys.path`, and the agent is the LAST callable in that namespace by insertion order (`get_last_callable`). Re-binding a name keeps its first position, so a file that defines `agent`, then helpers, then `agent` again runs its last helper. `load_agent` does the same |
| Arity and configuration | calls `agent(obs, configuration)` truncated to `agent.__code__.co_argcount`; callables without `__code__` get both. `configuration` is an attribute-accessible dict with the resolved defaults (`seed` is `None`, `runTimeout` 1200) |
| Per-seat isolation | each seat gets a fresh deep copy of its observation with only its own `private` (plus `remainingOverageTime: 60`), so mutation cannot leak |
| Episode length | actions are solicited for steps 0..718 (719 actions); the final bank is read from the step-719 state |
| positional slots | `[]` hand entries and market orders keep their index |

The same rules apply to Python agents run by `kagg tournament` /
`kagg selfplay` through `python -m kaggsim.host`;
`tests/integration/test_orchestration.py` checks those games' banks
against the official runner (including an agent that vandalises its
observation and a 1-argument agent that acts on the final turns).

## First-divergence tracer

`python -m kaggsim.fidelity diverge a.py b.py --seed 3` plays two agent
files on the official runner and on `kagg serve`, then prints the first step
where the state digest or an emitted action differs, with the differing
field. Different actions for identical states usually mean the agent is not
deterministic (wall-clock budgets, unseeded RNG, global state).

## Not covered

* Non-default configurations (`marketParams`, board size, turns per day...).
  The Rust engine implements the default configuration only.
* Agent timeouts and exceptions: the official runner turns them into an
  ERROR status. `kaggsim.serve.run_match` raises; `kagg tournament` /
  `kagg selfplay` apply `on_error` (`forfeit` or `exclude`). Per-turn
  time limits are enforced only when `time_limits` is configured, and the
  observation's `remainingOverageTime` always reads 60.
