# Self-play, samples and output hooks

## Self-play

`kagg selfplay <config.json>` plays an agent against itself (independent
instances in each seat) or against a pool, across worlds, and streams
labelled per-step samples to output hooks. Every sink and summary lands in
`<output.dir>/<name>/`.

```json
{
  "name": "sp",
  "agent": {"name": "me", "type": "python", "path": "main.py"},
  "opponents": {"mode": "pool", "p_mirror": 0.5, "rng_seed": 1,
                "pool": [{"name": "rnd", "type": "builtin", "kind": "random",
                          "seed": "per_game"}]},
  "worlds": {"strategy": "stratified", "pool": [0, 5000], "per_world": 25,
             "key_depth": 2},
  "seats": "both",
  "workers": 8,
  "samples": {"features": ["time", "money", "market", "shed", "tiles"],
              "labels": ["outcome", "return_to_go"], "gamma": 0.99},
  "sinks": [{"type": "jsonl", "path": "data/sp.jsonl"}]
}
```

`opponents.mode`: `mirror` (default), or `pool` where each game's opponent
is drawn from `pool`, or the agent itself with probability `p_mirror`,
deterministically from `rng_seed`. Other keys (`seats`, `workers`,
`on_error`, `output`, `python`, `key_depth`) work as for tournaments. With
no `sinks`, samples and game rows go to `<dir>/<name>/samples.jsonl`.
`summary.json` reports games, errors, games/s, the agent's score, records
written per sink and the realized-world distribution.

## Samples (`samples` block; tournaments accept it too)

| Key | Default | Meaning |
|---|---|---|
| `features` | `[]` (all groups) | groups: `time money market shed seeds carried hands quadrants tiles shops`; `["none"]` for no features |
| `include_obs` | `false` | add the seat's full observation (`obs`), e.g. for a custom extractor |
| `include_state` | `false` | add the full two-seat state (`state`) |
| `include_action` | `true` | add the action as a tape line (`action_line`) |
| `action_json` | `false` | also add the action dict returned by Python agents (`action`) |
| `labels` | `["outcome", "margin", "return_to_go"]` | see below |
| `gamma` | `1.0` | discount for `return_to_go` |
| `margin_scale` | `1.0` | multiplier for `margin` |
| `seats` | `"all"` | `all`, `candidate` (seats with role candidate), or a list like `[0]` |
| `agents` | `[]` | only record these agent names |
| `stride` | `1` | record every n-th step |
| `steps` | `[0, 719]` | `[lo, hi)` step range |
| `sample_rate` | `1.0` | keep each sample with this probability (deterministic per game) |
| `rng_seed` | `0` | seed for `sample_rate` |

Features are computed from the seat's OWN view (public board, market, town
and its own private block), before the step is applied.

Labels:

| Label | Value for the sample's seat |
|---|---|
| `outcome` | 1 win, 0.5 draw, 0 loss |
| `won` | final bank > opponent's final bank |
| `margin` | (final bank - opponent final bank) x `margin_scale` |
| `bank` | final bank |
| `reward` | bank change on this step |
| `return_to_go` | sum over k of gamma^k x reward(t+k), to the end of the game |
| `steps_left` | 719 - step |

Samples are buffered per game and written when it ends, because outcome
labels are only known then.

A sample record:

```json
{"record": "sample", "game_id": "…", "seed": 17, "tag": "selfplay",
 "seat": 0, "agent": "me", "opponent": "rnd", "role": "candidate",
 "world": "SHOP_A|SHOP_B", "target_world": "SHOP_A|SHOP_B",
 "step": 120, "day": 5,
 "features": {"step": 120, "money_me": 2710, "…": 0},
 "action_line": "WATER\t;NORTH\tSELL WHEAT 6",
 "labels": {"outcome": 1, "return_to_go": 2140.5}}
```

Game records are the result rows with `"record": "game"`.

## Sinks (output hooks)

```json
{"type": "jsonl",   "path": "out/samples.jsonl", "records": ["sample"]}
{"type": "stdout",  "records": ["game"]}
{"type": "command", "records": ["sample"],
 "argv": ["python", "-m", "kaggsim.processor", "--features",
          "mypkg.feats:extract", "--out", "data/train.jsonl"]}
```

`records` selects `game` and/or `sample` (default both). A `command` sink
starts ONE process for the run and streams the records to its stdin, so any
language can process them. Each sink has a writer thread behind a bounded
channel: a slow consumer slows the run down instead of buffering it in
memory. A sink command that exits non-zero fails the run.

## Custom features and labels (`kaggsim.processor`)

`python -m kaggsim.processor` reads the stream and writes JSONL:

| Option | Meaning |
|---|---|
| `--features module:fn` | `fn(sample) -> dict`, merged into `features` |
| `--labels module:fn` | `fn(sample) -> dict`, merged into `labels` |
| `--keep a,b,c` | keep only these fields |
| `--records sample,game` | which records to write (default `sample`) |
| `--keep-obs` | keep `obs` / `state` (dropped by default after extraction) |
| `--processor module:Class` + `--opt k=v` | use your own `kaggsim.processor.Processor` subclass (`on_game`, `on_sample`, `close`) |
| `--out file` | output (default stdout) |

`kaggsim.processor.basic_features(obs)` recomputes the Rust feature set
from an observation, for extractors that extend it. Set
`samples.include_obs: true` when your function needs the observation.

Example: `scripts/examples/selfplay_example.py` with
`scripts/examples/custom_features.py`.
