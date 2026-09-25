# World generator

The "world" of an episode is, above all, its sequence of unlocked town
shops. Shops set town demand (which products drain from the market, and how
fast), so the first two shops shape the whole season.

## The world is realized by play, not by the seed

The engine seeds ONE `random.Random((seed * 1_000_003) ^ day)` per day. At
each end of day it first spawns weeds, one `random()` draw per EMPTY
unlocked tile (farm 0, then farm 1), and then, every third day, draws a shop
with `choice()` from the same generator. How many tiles are empty depends on
what both players planted, dug and built. So the shops depend on:

* the seed,
* both players' action streams,
* the seat order.

A catalog built by driving idle (PASS) players labels worlds that real play
may never visit. `kaggsim.worlds` computes the REALIZED world instead.

## Split points

* First shop: drawn at the end of day 2 (during step 71). Actions at steps
  `>= 72` cannot change it.
* Second shop: drawn at the end of day 5 (during step 143). Actions at steps
  `>= 144` cannot change it.

`tests/regressions/test_worlds.py` checks both invariances, and checks that
changes before a split do move its key.

If a candidate differs from a baseline only at or after a split, the
baseline's world key for that split is still correct for the candidate, so
per-world grouping computed once on the baseline prefix stays valid.

## API

```python
from kaggsim.worlds import realized_world, build_map, build_catalog, idle_catalog

realized_world(srv, seed, stream0, stream1, k=2)   # "SHOP1|SHOP2"
build_map(stream, [opp_a, opp_b], seeds)           # {(opp_i, seed, seat): key}
build_catalog(lambda: PolicyA(), lambda: PolicyB(), seeds, mirror=True)
idle_catalog(seeds)                                # the PASS/PASS baseline
```

Streams are tape paths, lists of tape lines, or lists of action dicts.
`build_catalog` plays CLOSED-LOOP policies through `kagg serve`, and its
catalog describes that policy pair only. CLI:

```
python -m kaggsim.worlds catalog --seeds 0-99 --a scripted:1 --b random:2 --mirror --out worlds.json
python -m kaggsim.worlds catalog --seeds 0-99      # idle baseline
```

## Seed strategies for runs

Tournaments and self-play choose seeds with a world strategy (`list`,
`range`, `stratified`, `weighted`), computed in Rust; see
[tournaments.md](tournaments.md). They target worlds with the idle-drive
label and record the world each game actually realized.

`kagg worlds <seeds> [key_depth] [threads]` prints an idle-drive catalog
as JSON.

## Continuation screening (open loop)

`kaggsim.worlds.screen_continuations(sources, prefix, opponents, seeds,
at_step=144, threads=2, top_k=10)` (CLI: `python -m kaggsim.worlds
screen ...`):

1. turns replays (or tape pairs / single tapes) into per-seat tapes;
2. labels each tape with the world realized by its own recorded pair;
3. for each evaluation cell (opponent tape, seed, seat), computes the world
   the `prefix` realizes there (splicing at `>= 144` cannot change it);
4. for every world, splices `prefix[:at_step] + candidate[at_step:]` for
   each candidate labelled with that world and plays it against the
   opponent tapes on that world's cells with `kagg batch` (score: win 1,
   draw 0.5);
5. ranks the candidates per world, compares the leader with the runner-up
   (McNemar exact) and writes `<out>.json` and a Markdown summary
   `<out>.md`.

Tapes do not react to the board. A stream that ranks well open loop can
still lose as a live agent: treat the result as SCREENING and confirm with
live agents (`kagg tournament` / `kaggsim.gauntlet`).

## Low-noise A/B on the official engine (development use only)

`kaggsim.forced` patches the official interpreter so that weeds keep the
engine's stream while the shop draw uses an independent one; `forced_shops`
can pin the draw outright. Two builds then see the same shops however they
farm. **This changes the game**: the fidelity tools refuse to run while it
is active; play patched games with `kaggsim.forced.run_agents` /
`kaggsim.forced.make`.
`python -m kaggsim.forced --selftest` demonstrates the coupling, the split
and the pin.
