# AGENTS.md: guidance for AI coding agents working on this repository

This file is for automated coding agents (and humans) changing
kaggriculture-simulation itself. Read it before editing anything.

## What this repository is

A byte-identical Rust port of the Kaggle Kaggriculture engine
(`kaggle-environments==1.32.7`), a parallel Rust orchestration layer
(tournaments, self-play, labelled data export), and a small stdlib-only
Python toolkit around it. Correctness means **identical state to the
official engine at every step**; speed and low memory are the reason the
project exists.

## Layout

| Path | What lives there |
|---|---|
| `src-rust/Cargo.toml` | the Cargo workspace (there are no Cargo files at the repository root) |
| `src-rust/kagg-engine/src/` | the engine: `engine.rs` (step function), `state.rs`, `market.rs` (1.32.7 price table), `rules.rs`, `mt19937.rs`, `json.rs`, `tape.rs`, `obsjson.rs`, `loadstate.rs`, `obsstate.rs`, `world.rs`, `features.rs`, `policies.rs` (fixtures) |
| `src-rust/kagg-sim/src/` | orchestration: `agent.rs` (specs + Python host protocol), `runner.rs`, `samples.rs`, `sink.rs`, `seeding.rs`, `tournament.rs`, `selfplay.rs`, `stats.rs`, `util.rs` |
| `src-rust/kagg-cli/` | the `kagg` binary (`src/main.rs`, `serve.rs`, `batch.rs`, `diag.rs`, `run.rs`) and its integration tests (`tests/cli.rs`) |
| `src-python/kaggsim/` | Python package: `host.py`, `serve.py`, `env.py`, `tape.py`, `worlds.py`, `fidelity.py`, `official.py`, `economy.py`, `forced.py`, `stats.py`, `processor.py`, `tournament.py`, `selfplay.py`, `gauntlet.py`, `batch.py`, `benchmark.py`, `policies.py`, `constants.py`, `binary.py`, `_util.py` |
| `tests/unit` | Python unit tests (no engine binary or official engine needed) |
| `tests/regressions` | rule regressions (most need the official engine) |
| `tests/differential` | step-by-step Rust vs official |
| `tests/integration` | orchestration, CLIs, scripts and examples |
| `scripts/` | `dev.py` (task runner behind the Makefile), `scrub_audit.py`, `fetch_official.py`, `examples/` |
| `docs/` | reference documentation; keep it consistent with the README |

## Build and test (exact commands)

```
cargo build --manifest-path src-rust/Cargo.toml --release -j 4   # src-rust/target/release/kagg
cargo test  --manifest-path src-rust/Cargo.toml -j 4 -- --test-threads=2
cargo fmt   --manifest-path src-rust/Cargo.toml --all --check
cargo clippy --manifest-path src-rust/Cargo.toml -j 4 --all-targets -- -D warnings
python scripts/fetch_official.py           # pinned official engine -> .pinwork/official
export KAGGSIM_OFFICIAL_PATH=.pinwork/official   # or pip install --no-deps kaggle-environments==1.32.7
pip install jsonschema requests pytest
KAGGSIM_REQUIRE_OFFICIAL=1 python -m pytest -q   # everything, 50 differential episodes
python -m kaggsim.fidelity certify --episodes 20 # with PYTHONPATH=src-python
python scripts/scrub_audit.py
```

`make build test lint certify scrub` (or `python scripts/dev.py <task>`)
runs the same things; `make docker-linux docker-test` builds the static
Linux binary and runs the whole suite on Linux in Docker. Keep parallelism modest on shared machines
(`CARGO_JOBS`, `TEST_THREADS`, `WORKERS`).

## Invariants that must never break

1. **Byte-identical parity.** Every change to `kagg-engine` must keep the
   differential suite green: full-state digest equal after every step.
   Preserve the interpreter's statement order (player 0's actions before
   player 1's; market settled per ORDER INDEX in lockstep; prices refreshed
   after each index; farm 0's end-of-day before farm 1's on the shared RNG).
2. **Positional empty slots.** An empty `[]` hand entry or market order
   keeps its index (`engine::positional`, `kaggsim.tape.action_to_line`).
   Never filter empties out of tape lines.
3. **719 actions.** Episodes apply actions for steps 0..718 and read the
   final banks from the step-719 state (`state::FINAL_STEP`). `batch`,
   `serve`, `ROLLOUT`, `GENGAME`, the runner and every Python driver stop
   there.
4. **Per-seat observation isolation.** Each seat gets a fresh copy of its
   own observation containing only its own `private` block (plus
   `remainingOverageTime`). Never hand an agent the full state.
5. **Official invocation.** `agent(obs, configuration)` truncated to
   `co_argcount`; callables without `__code__` get both; `configuration` is
   the resolved default (`seed: None`, `runTimeout: 1200`, ...).
6. **Engine hash pin.** `kaggsim.official` refuses any engine whose
   `kaggriculture.py`, spec or runner files differ from the pinned hashes,
   or whose RNG is patched. Do not weaken or bypass this check to make a
   test pass.
7. **1.32.7 price table.** `market.rs` holds the 1.32.7 table only
   (CARROT/TOMATO/EGG hinge scarcity curves). Do not reintroduce older
   tables.
8. **Fresh agents per game.** Python agents are re-executed in a new
   namespace for every game (compiled code may be cached).

## How to verify a change

* Engine or tape code: `cargo test`, then `pytest tests/regressions
  tests/differential` with the official engine, then
  `python -m kaggsim.fidelity certify --episodes 100 --base-seed <new>`.
* Orchestration (`kagg-sim`, host, processor): `cargo test` and
  `pytest tests/integration` (includes an official-parity check of games
  played through the Python host).
* Anything user-visible: update the README and the matching page in
  `docs/` in the same change.

## Conventions

* Python package: standard library only at runtime. The official engine is
  an optional dependency used only for fidelity tooling.
* Performance-sensitive or parallel work belongs in Rust; Python is for
  agent hosting, official-engine tooling and thin front ends.
* No agent strategy code. Policies in `policies.rs` / `policies.py` are
  deliberately simple fixtures.
* Tests use synthetic data only (fixture policies, generated tapes). Never
  add real competition replays, tapes, ratings or team names.
* Every new function, CLI flag and module needs a test.
* Write files with an editor tool, not shell heredocs: heredocs can mangle
  backslash escapes in Rust and Python string literals.

## Never commit

`.pinwork/` (downloaded wheels, extracted engine), `.scratch/`,
`.internal/` (untracked local notes), `.scrub_terms.txt`, `target/`,
`src-rust/target/`, run outputs (`tournaments/`, `selfplay/`,
`gauntlets/`). They are in
`.gitignore`; keep it that way. Run `python scripts/scrub_audit.py` before
committing.

## Porting a new Kaggle engine version

1. `pip download --no-deps kaggle-environments==X -d .pinwork/` and hash
   `kaggle_environments/envs/kaggriculture/kaggriculture.py` and
   `kaggriculture.json` inside the wheel.
2. Diff the new `kaggriculture.py` against the pinned one; port every
   behavioural change to `src-rust/kagg-engine` in statement order.
3. Update `market::ENGINE_VERSION`, `PINNED_VERSION` / `PINNED_*_SHA256`
   (engine, spec and `PINNED_RUNNER_SHA256`) in
   `src-python/kaggsim/official.py`, `KAGGLE_ENVIRONMENTS_VERSION` in
   `.github/workflows/ci.yml`, `pyproject.toml` (`official` extra) and
   `docs/version-pinning.md`.
4. Re-certify: the full test suite plus
   `python -m kaggsim.fidelity certify --episodes 500`.
