# Tape format

## Action line

One seat's action for one step:

```
<farmer tokens> TAB <hand;hand;...> TAB <order;order;...>
```

* Tokens within an action are separated by single spaces:
  `PLANT WHEAT`, `PICKUP EGG 3`, `SELL MILK 12`, `HIRE`.
* The hands and market fields are **positional**. Each `;`-separated
  segment is one entry, and **an empty segment is an empty action `[]` that
  keeps its index**. `;WATER` means "hand 0 does nothing, hand 1 waters";
  `;SELL WHEAT 3` means "order 0 is empty, order 1 sells". A wholly empty
  field means zero entries.
* Missing trailing fields are empty. An empty farmer field is `PASS`.

Why positional: the official interpreter applies `hands[i]` to hand `i`, and
settles the market one ORDER INDEX at a time in lockstep across both seats.
For index `i`, both seats' `i`-th orders are quoted at the same inventory,
unit by unit, and prices refresh after each index. Dropping an interior `[]`
would give a later action to the wrong hand and change which orders settle
at the same index.

Encoding rules (`kaggsim.tape.action_to_line`) mirror how the interpreter
reads an action:

* a non-dict action is the empty action;
* a non-list or empty hand / order entry becomes an empty segment;
* the third element (the count) goes through `int()` (`2.9 -> 2`,
  `True -> 1`); other elements are names;
* a token containing a space, tab, `;` or a control separator cannot be a
  valid name and is replaced by `?` (the engine ignores it either way).

## Files

* **Single-seat tape**: `SEED <n>` then one action line per step. Line `t`
  is the action at step `t`. Only steps 0..718 are applied; a step-719 line
  is ignored (see `docs/fidelity.md`). Steps past the end of a tape are
  empty actions.
* **Two-seat tape** (`kagg episode`, `kagg bench`): `SEED <n>` then, per
  step, the seat-0 line followed by the seat-1 line.
* **Batch jobs** (`kagg batch jobs.tsv [threads]`): one job per line,
  `seed \t tapeA \t tapeB`; seed `-` uses tapeA's header. Output in input
  order: `idx \t seed \t bank0 \t bank1`, or `idx \t ERR \t message`.

## Tools (`python -m kaggsim.tape`)

* `from-replay <replay.json> <seat> <out.tape>`: one seat of a replay.
  Replay alignment: `steps[t+1][seat].action` is the action applied at
  engine step `t`.
* `validate <tape>`: shape legality (known ops, argument shapes, at most 10
  market orders, unencodable tokens). Through the Python API
  (`validate_tape(path, hand_counts)`) it also checks that `hands` aligns
  with the live hand count. Game-state legality (money, position, seeds) is
  a silent no-op in the engine and needs a simulation.
* `splice <prefix> <suffix> <at_step> <out>`: steps `0..at_step-1` from
  the prefix tape, `at_step..718` from the suffix tape, the SEED of the
  prefix; both inputs are shape-validated over the steps used
  (`--no-validate` to skip). World split points: the first shop is drawn
  while step 71 runs, so a splice at `>= 72` keeps the prefix's first shop;
  the second while step 143 runs, so `>= 144` keeps both (for the same
  opponent stream and seat). Python: `splice_tapes(prefix, suffix,
  at_step, out)`, which returns a report including `world_kept`.
* `to-agent <tape> <main.py> [--align-hands]`: a standalone, open-loop
  `main.py` that replays the tape by observation step. Verbatim by default,
  which reproduces `kagg batch` exactly on the official runner.

Python helpers: `write_tape`, `read_tape`, `write_episode_tape`, `splice_tapes`,
`replay_actions`, `replay_seed`, `replay_to_tape`, `batch_jobs`, and
`kaggsim.batch.run_batch`.
