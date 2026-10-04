# Owned full-action simulator

Adapted from `msdsm/kaggriculture-solution`, commit
`84057a0fda4238ccdebc46f9bf5496c6c4b2e00d`, `native/engine`.
The copied source is maintained here and needs no reference checkout at runtime.
This is separate from the historical mixed/plan/event simulator.

The Python module is `route_rl_action_engine`. Project additions preserve the
official-prefix conditional supports, all ten market positions including NOOP,
and exact absolute SELL requests. No opponent private inventory is used to
construct a conditional support. Feature and inventory-history equivalence is
covered by differential tests; those tests do not establish playing strength.

Build on Linux with `python tools/build_action_native.py --jobs 2`; add `--offline`
when Cargo dependencies are cached. The tool writes the extension and a verified
source/binary receipt into `src/route_rl/ppo/_native/` for package inclusion.
The build uses the pinned Cargo lockfile. No training starts during building.
