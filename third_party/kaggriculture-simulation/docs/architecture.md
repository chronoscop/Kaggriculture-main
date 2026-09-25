# Architecture

```
            +---------------------------- kagg (one Rust process) ----------------------------+
            |                                                                                  |
 config --> |  scheduler --> work queue --> worker thread 1 ... worker thread N                |
 (JSON)     |  (gauntlet /   (games)        |  engine State (a few kB per game)              |
            |   round robin,                |  builtin policies / tapes: in-thread           |
            |   self-play,                  |  Python agents: ONE host process per worker ---+--> python -m kaggsim.host
            |   world strategy)             |  samples: features + labels (Rust)             |    (slot 0 / slot 1 = seats)
            |                               v                                                |
            |                        result channel --> results.jsonl (resumable)            |
            |                                       \-> sinks: jsonl | stdout | command ------+--> any processor (stdin)
            |                                       \-> summary.json / summary.md            |
            +----------------------------------------------------------------------------------+
```

## Why Rust at the core

Everything that is not agent code runs in Rust: the engine, scheduling,
world selection, the parallel game loop, feature extraction, labelling,
output sinks and statistics. Rust gives:

* **Speed.** The step function runs at hundreds of thousands of steps per
  second on one thread; games between built-in policies or tapes run at
  hundreds of games per second (see [benchmarks.md](benchmarks.md)).
* **Small, predictable memory.** A game's state is a few kilobytes. The
  measured peak resident size of `kagg` running 1,000 games on 2 worker
  threads was about 7 MiB (see [benchmarks.md](benchmarks.md)).
* **Memory safety without a garbage collector.** Worker threads share only
  a work queue and a result channel. Sinks use bounded channels, so a slow
  processor applies back-pressure instead of growing memory.

## Where Python remains

Only where it must:

* **Python agents.** Kaggle submissions are Python, so they run in Python.
  Each worker thread owns ONE small `python -m kaggsim.host` process
  (about 22 MiB) that holds up to two agents in slots (slot = seat). Each
  game re-executes the agent file's cached, compiled code in a fresh
  namespace, so no module state leaks between games. A game with Python
  agents in both seats costs one round trip per step.
* **Fidelity tooling.** The official engine is Python, so the differential
  certifier, the replay-economics analyzer and the dev-only forced-world
  patch are Python.
* **Thin front ends.** `kaggsim.tournament`, `kaggsim.selfplay`,
  `kaggsim.gauntlet`, `kaggsim.serve`/`kaggsim.env` write configs and talk to
  `kagg`; `kaggsim.processor` is a ready-made receiving end for sink
  streams.

The Python package has no runtime dependencies beyond the standard library.

## Crates and packages

| Path | Role |
|---|---|
| `src-rust/kagg-engine` | engine library: state, step function, market, rules, MT19937, JSON, tapes, worlds, features, fixture policies |
| `src-rust/kagg-sim` | orchestration library: agent specs + Python host protocol, seeding strategies, runner, samples, sinks, tournament, self-play, statistics |
| `src-rust/kagg-cli` | the `kagg` binary |
| `src-python/kaggsim` | Python toolkit (host, serve client, env, tapes, worlds, fidelity, economy, forced, stats, processor, front ends, benchmark) |

## Isolation and fidelity guarantees

* Agents are invoked exactly like the official runner: arity-truncated
  `(obs, configuration)`, a fresh attribute-accessible observation per call
  containing only the seat's own private block, the resolved default
  configuration, and 719 actions per episode. See [fidelity.md](fidelity.md).
* Actions cross the process boundary as positional tape lines, the same
  encoding the differential suite certifies.
* `tests/integration/test_orchestration.py` plays Python agents through
  `kagg tournament` and checks every bank against the official runner.
