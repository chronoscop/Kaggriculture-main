# Tournaments

`kagg tournament <config.json>` plays a configurable schedule of games in
parallel and writes, into `<output.dir>/<name>/`:

| File | Content |
|---|---|
| `config.json` | the resolved configuration |
| `results.jsonl` | one row per game (resumable: games already recorded are skipped) |
| `summary.json` | standings, pairwise matrix, and the candidate's breakdowns |
| `summary.md` | the same as a Markdown report |

```
kagg template tournament > t.json          # starter config
kagg tournament t.json --workers 8         # run
kagg tournament t.json --set worlds.count=200 --set seats=seat0
kagg compare a/results.jsonl b/results.jsonl mine    # paired A/B
```

Python: `kaggsim.tournament.run_tournament(cfg)` returns `summary.json` as a
dict (`kaggsim.gauntlet.run_gauntlet` is a shortcut for "one file vs these
files").

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `name` | `"tournament"` | output sub-directory |
| `candidate` | - | your agent (required for `gauntlet`) |
| `panel` | - | list of opponent agents |
| `schedule` | `"gauntlet"` | `gauntlet` (candidate vs each panel agent) or `round_robin` (all pairs of candidate + panel) |
| `seats` | `"both"` | `both` (every pairing in both seat orders), `seat0`, `seat1`, `alternate` |
| `worlds` | `{"strategy": "range", "start": 0, "count": 20}` | seed strategy, below |
| `key_depth` | `2` | shops in the world key used by summaries |
| `world_weighting` | `"none"` | `none`, `uniform` (every realized world counts equally) or `{"WORLD": w, "*": w}` |
| `workers` | `2` | worker threads (and at most that many Python hosts) |
| `on_error` | `"forfeit"` | an agent exception loses the game (`forfeit`, like the official runner) or the game is kept without scores (`exclude`) |
| `time_limits` | `null` | per-turn budget for Python agents, like the official runner: `{"act_s": 1.0, "overage_s": 60.0}`; time over `act_s` is drawn from a per-game overage bank, and a seat that exhausts it loses the game (`on_error`) |
| `output` | `{"dir": "tournaments", "resume": true}` | output directory and resume switch |
| `python` | `{}` | `exe` (default `$KAGGSIM_PYTHON` or `python`), extra `path` entries, `stderr`: `inherit` or `null` |
| `samples` | `null` | per-step training samples, see [selfplay-and-data.md](selfplay-and-data.md) |
| `sinks` | `[]` | output hooks, see [selfplay-and-data.md](selfplay-and-data.md) |
| `progress` | `true` | progress line on stderr |

### Agent specs

| Type | Fields | Runs in |
|---|---|---|
| `python` | `path` to a submission `main.py` | Python host |
| `factory` | `ref` = `"module:callable"`, `kwargs` (gets `game_seed` if it accepts it) | Python host |
| `pypolicy` | `kind` = `scripted` / `random` / `chaos` / `last_turn`, `seed` | Python host |
| `tape` | `path` to a single-seat tape | Rust (open loop) |
| `builtin` | `kind` = `idle` / `random` / `chaos`, `seed` | Rust |

`"seed": "per_game"` derives a policy seed from the game seed. Every agent
needs a unique `name`. A game's id hashes both agents' content, their
names and the seed, so changing an agent makes the next run replay its
games while untouched pairings resume. Content means: for `python`, the
file plus every other `*.py` file in its directory (helper modules, not
recursive); for `tape`, the file; for `factory` / `pypolicy` / `builtin`,
the spec itself. Add any extra field (for example `"version": "2"`) to a
spec to force a replay when code the fingerprint cannot see has changed.

### World strategies (`worlds`)

```json
{"strategy": "list", "seeds": [1, 2, 3]}
{"strategy": "range", "start": 0, "count": 100}
{"strategy": "stratified", "pool": [0, 5000], "per_world": 4, "key_depth": 2,
 "include": [], "exclude": []}
{"strategy": "weighted", "pool": [0, 5000], "count": 200, "key_depth": 1,
 "weights": {"SHOP_A": 2, "*": 1}, "rng_seed": 7}
```

`stratified` and `weighted` target worlds with an idle-drive label computed
in Rust (`SHOP_A` stands for any shop name). A plan never repeats a seed:
`list` drops duplicates and `weighted` samples without replacement, so it
returns fewer than `count` seeds when the pool runs out. The realized world depends on play (see
[world-generator.md](world-generator.md)), so every game records the world
it actually realized, and summaries group by that. `target_world` in each
row keeps the intended label.

## Result rows

```json
{"game_id": "…", "seed": 17, "tag": "", "target_world": "SHOP_A|SHOP_B",
 "world": "SHOP_A|SHOP_B", "shops": ["SHOP_A", "SHOP_B", "…"],
 "agents": ["mine", "rnd"], "roles": ["candidate", "panel"],
 "banks": [5321, 2988], "scores": [1, 0], "margin": 2333, "steps": 719,
 "secs": 0.41, "act_ms_mean": [0.05, 0], "act_ms_max": [0.3, 0],
 "error": null, "forfeit": false}
```

`act_ms_*` is the time spent inside each agent call (Python seats only).

## Summary

* `standings`: per agent games, W/D/L, score (win 1, draw 0.5) with a 95%
  confidence interval, mean margin, errors, time per turn.
* `matrix`: score of every agent against every other.
* `focus` (the candidate): overall, `world_weighted_score`, and blocks by
  opponent, by realized world and by seat.

## Comparing two builds

Run the same config twice, changing only the candidate, then:

```
kagg compare runA/results.jsonl runB/results.jsonl mine [mine_b]
```

Games pair by (opponent, seed, the agent's seat); when a results file holds
several rows for the same game (resumed runs after an agent changed), the
latest row wins. The output is McNemar's exact test on the discordant pairs
(`better_a`, `better_b`, `p_value`), the mean score difference with a 95%
CI, and a per-opponent breakdown.
