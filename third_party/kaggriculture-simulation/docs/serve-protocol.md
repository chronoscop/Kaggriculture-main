# `kagg serve` protocol

`kagg serve` is a line-oriented step environment on stdin/stdout. Every
request is one line; every response is exactly one line of JSON. Errors are
JSON too: `{"error": "..."}`. The process holds one current episode.

| Request | Response |
|---|---|
| `RESET <seed>` | initial state; both seats are driven by the caller |
| `RESET <seed> OPP <seat> <tapepath>` | initial state; seat `<seat>` replays a single-seat tape |
| `STEP <line>` | post-step state; `<line>` is the free seat's action. The other seat plays its tape (after `OPP`) or the empty action |
| `STEP2 <line0>\x1e<line1>` | post-step state; both seats |
| `LOADSTATE <json>` | the loaded state; continue from a full-state JSON (below). An optional `"seed"` key sets the RNG seed (default 0). A state the engine cannot play (not exactly 2 farms and 2 private blocks, a board that is not 10x10, a position off the board, a step outside 0..719) is rejected with an error |
| `ROLLOUT <H> <json>\x1e<lines0>\x1e<lines1>` | final state after stepping `H` times from `<json>`; `<linesN>` are tape lines joined by `\x1f`, indexed from the snapshot |
| `GENGAME <seed>\x1e<lines0>\x1e<lines1>` | `{"days": [...], "final": state}`: a whole game in one call; `days` holds one compact record per day boundary (`step`, `day`, `money`, `unlocked_shops`, market `inventory`) |
| `QUIT` | exits (no response) |

`<line>` is a tape line (see `docs/tape-format.md`): `farmer\thands\tmarket`.
`\x1e` (record separator) and `\x1f` (unit separator) are the raw control
bytes.

## State JSON

The official observation schema plus `done`, with BOTH seats' private
blocks:

```json
{"step": 0, "day": 0, "hour": 0, "done": false,
 "farms": [{"money": 3000, "farmer": [4, 4], "hands": [], "hires_today": 0,
            "unlocked_quadrants": ["NW"], "tiles": [["...10 cells..."]]}, {}],
 "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
 "town": {"unlocked_shops": []},
 "private": [{"shed": {}, "seeds": {}, "inventories": [{}]}, {}]}
```

(abbreviated). A driver must give each agent only its own `private[seat]`
plus `player = seat`; `kaggsim.serve.obs_for` does this, with a deep copy
per seat.

## Episode end

The official runner applies 719 actions (steps 0..718) and scores the
step-719 state. `kagg serve` marks `done: true` at step 719; further
`STEP`/`STEP2` requests are no-ops that return the terminal state again.
`ROLLOUT` and `GENGAME` stop at the same point, and `kagg batch` ignores any
tape line for step 719.

## Python client

```python
from kaggsim.serve import Serve, run_match

with Serve() as srv:
    js = srv.reset(3)
    js = srv.step2({"farmer": ["WATER"], "hands": [], "market": []}, {})
    banks = run_match(agent_a, agent_b, seed=3, srv=srv)
```

`kaggsim.env.Env` wraps the same process as a gym-style
`reset(seed)` / `step(a0, a1)` / `close()` environment.
