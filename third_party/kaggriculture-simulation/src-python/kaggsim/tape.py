"""Tapes: the line format ``kagg`` consumes, and tools around it.

A *tape line* encodes one seat's action for one step::

    <farmer tokens> TAB <hand;hand;...> TAB <order;order;...>

Hands and market orders are POSITIONAL. An empty or non-list entry becomes an
empty segment and keeps its index, because the official interpreter applies
``hands[i]`` to hand ``i`` and settles the market one ORDER INDEX at a time in
lockstep across both seats. Dropping an interior ``[]`` would hand a later
action to the wrong worker and change which orders settle at the same index.

A *single-seat tape* file is ``SEED <n>`` followed by one line per step
(steps 0..718 matter; a step-719 line is never applied). A *two-seat tape*
(``kagg episode`` / ``kagg bench``) interleaves seat 0 and seat 1 per step.
Full spec: ``docs/tape-format.md``.
"""
from __future__ import annotations

import json
import os
import re

from .constants import (ANIMALS, CROPS, FIRST_SHOP_STEP, MARKET_OPS,
                        MAX_MARKET_ORDERS, N_ACTIONS, PRODUCTS,
                        SECOND_SHOP_STEP, UNIT_OPS)

# Characters a token may not contain: the separators, anything a line
# splitter might treat as a line break, and all other control characters.
_BAD_CHARS = frozenset([" ", ";", "\x7f", "\x85", "\u2028", "\u2029"]
                       + [chr(c) for c in range(32)])
_INT = re.compile(r"^[+-]?[0-9]+$")


# ------------------------------------------------------------- encoding --

def _token(v, count: bool = False) -> str:
    """One action element as a tape token, mirroring how the interpreter
    reads it: counts go through ``int()``; everything else is compared as a
    name, so a value that cannot be a name becomes an unmatched token."""
    if count and isinstance(v, (bool, int, float)):
        try:
            return str(int(v))
        except (OverflowError, ValueError):
            return "?"
    if count and isinstance(v, str):
        return v if _INT.match(v) else "?"
    s = v if isinstance(v, str) else json.dumps(v)
    if not s or any(c in s for c in _BAD_CHARS):
        return "?"
    return s


def _encode_list(a) -> str:
    return " ".join(_token(v, count=(i == 2)) for i, v in enumerate(a))


def action_to_line(action) -> str:
    """One seat's action dict (official schema) -> one tape line.

    ``None`` / non-dict actions are the empty action (farmer PASS, no hands,
    no orders), which is what the interpreter does with them.
    """
    seq = (list, tuple)          # JSON turns tuples into lists on the wire
    action = action if isinstance(action, dict) else {}
    farmer = action.get("farmer", ["PASS"])
    f = _encode_list(farmer) if isinstance(farmer, seq) and farmer else "PASS"
    hands = action.get("hands") or []
    market = action.get("market") or []
    if not isinstance(hands, seq):
        hands = []
    if not isinstance(market, seq):
        market = []
    h = ";".join(_encode_list(a) if isinstance(a, seq) else "" for a in hands)
    m = ";".join(_encode_list(o) if isinstance(o, seq) else "" for o in market)
    return f"{f}\t{h}\t{m}"


def _parse_tokens(seg: str):
    return [int(t) if _INT.match(t) else t for t in seg.split(" ") if t]


def _positional(field: str):
    return [] if field == "" else field.split(";")


def line_to_action(line: str) -> dict:
    """Inverse of :func:`action_to_line` (numbers come back as ints)."""
    parts = line.rstrip("\r\n").split("\t")
    parts += [""] * (3 - len(parts))
    farmer = _parse_tokens(parts[0]) or ["PASS"]
    return {"farmer": farmer,
            "hands": [_parse_tokens(h) for h in _positional(parts[1])],
            "market": [_parse_tokens(o) for o in _positional(parts[2])]}


# ------------------------------------------------------------ tape files --

def write_tape(path: str, seed: int, actions) -> None:
    """Write a single-seat tape (``SEED`` header + one line per action)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"SEED {int(seed)}\n")
        for a in actions:
            fh.write((a if isinstance(a, str) else action_to_line(a)) + "\n")


def read_tape(path: str):
    """(seed, [line, ...]) from a single-seat tape."""
    with open(path, encoding="utf-8", newline="") as fh:
        # split on "\n" only, exactly like kagg (str.splitlines() would also
        # split on form feeds and other characters)
        lines = [ln[:-1] if ln.endswith("\r") else ln
                 for ln in fh.read().split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or not lines[0].startswith("SEED "):
        raise ValueError(f"{path}: tape must start with 'SEED <n>'")
    return int(lines[0][5:].strip()), lines[1:]


def write_episode_tape(path: str, seed: int, actions0, actions1) -> None:
    """Two-seat tape (``kagg episode`` format): seat 0 then seat 1 per step."""
    n = min(len(actions0), len(actions1))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"SEED {int(seed)}\n")
        for i in range(n):
            for a in (actions0[i], actions1[i]):
                fh.write((a if isinstance(a, str) else action_to_line(a))
                         + "\n")


# ----------------------------------------------------------- replay IO --

def load_replay(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def replay_seed(rep: dict) -> int:
    """The episode seed recorded in a replay (``info.seed``)."""
    info = rep.get("info") or {}
    cfg = rep.get("configuration") or {}
    seed = info.get("seed", cfg.get("seed"))
    if seed is None:
        raise ValueError("replay has no recorded seed")
    return int(seed)


def replay_actions(rep: dict, seat: int):
    """Per-step action dicts for one seat.

    ALIGNMENT: ``steps[t].action`` is the action that PRODUCED state ``t``,
    so the action applied at engine step ``t`` lives in ``steps[t + 1]``.
    ``steps[0]`` is the initial state. Errored steps become ``{}``.
    """
    out = []
    for st in (rep.get("steps") or [])[1:]:
        a = st[seat].get("action") if seat < len(st) else None
        out.append(a if isinstance(a, dict) else {})
    return out


def replay_to_tape(rep: dict, seat: int, out_path: str) -> None:
    """Write ONE seat's recorded action stream as a single-seat tape."""
    write_tape(out_path, replay_seed(rep), replay_actions(rep, seat))


def replay_state(rep: dict, step: int) -> dict:
    """The FULL two-seat state recorded at ``step`` of a replay, in the
    ``kagg serve`` state format (``LOADSTATE`` / ``ROLLOUT``), including the
    episode ``seed`` so a continuation sees the real weeds and shops.

    A replay stores each seat's observation with that seat's own private
    block, so both private blocks are available here even though no single
    agent ever saw them together.
    """
    steps = rep["steps"]
    if not 0 <= step < len(steps):
        raise ValueError(f"step {step} outside the replay (0..{len(steps) - 1})")
    o0 = steps[step][0]["observation"]
    o1 = steps[step][1]["observation"]
    return {"step": o0["step"], "day": o0["day"], "hour": o0["hour"],
            "done": False, "farms": o0["farms"], "market": o0["market"],
            "town": o0["town"], "private": [o0["private"], o1["private"]],
            "seed": replay_seed(rep)}


# ------------------------------------------------------------ validator --

def validate_action(action, n_hands=None):
    """Legality problems of one action dict, as a list of strings.

    Checks the SHAPE the interpreter accepts (known ops, argument shapes,
    <= 10 market orders, hand alignment when ``n_hands`` is known). It does
    not check game-state legality (money, position, seeds): those are
    silent no-ops in the engine and only a simulation can tell.
    """
    issues = []
    if not isinstance(action, dict):
        return ["action is not a dict (engine treats it as PASS)"]

    def unit(a, where):
        if not isinstance(a, list) or not a:
            if a != []:
                issues.append(f"{where}: not a non-empty list (no-op)")
            return
        op = a[0]
        if not isinstance(op, str) or op not in UNIT_OPS:
            issues.append(f"{where}: unknown op {op!r}")
            return
        arity = UNIT_OPS[op]
        nargs = len(a) - 1
        if op == "PLANT" and (nargs < 1 or a[1] not in CROPS):
            issues.append(f"{where}: PLANT needs a crop")
        elif op == "PICKUP" and (nargs < 1 or a[1] not in PRODUCTS + ANIMALS):
            issues.append(f"{where}: PICKUP needs a product or animal")
        elif op == "PLACE" and (nargs < 1 or a[1] not in PRODUCTS + ANIMALS):
            issues.append(f"{where}: PLACE needs a product or animal")
        elif arity == 0 and nargs:
            issues.append(f"{where}: {op} takes no arguments")
        if op in ("PICKUP", "PLACE") and nargs >= 2 and not _is_count(a[2]):
            issues.append(f"{where}: count {a[2]!r} is not a positive int")
        for v in a:
            if isinstance(v, str) and any(c in v for c in _BAD_CHARS):
                issues.append(f"{where}: token {v!r} is not encodable")

    farmer = action.get("farmer", ["PASS"])
    unit(farmer, "farmer")
    hands = action.get("hands", [])
    if not isinstance(hands, list):
        issues.append("hands is not a list")
        hands = []
    for i, h in enumerate(hands):
        unit(h, f"hands[{i}]")
    if n_hands is not None and len(hands) != n_hands:
        issues.append(f"hands has {len(hands)} entries but the farm has "
                      f"{n_hands} hands (entries must align positionally)")
    market = action.get("market", [])
    if not isinstance(market, list):
        issues.append("market is not a list")
        market = []
    if len(market) > MAX_MARKET_ORDERS:
        issues.append(f"{len(market)} market orders; only the first "
                      f"{MAX_MARKET_ORDERS} are processed")
    for i, o in enumerate(market):
        where = f"market[{i}]"
        if o == []:
            continue                      # positional placeholder, legal
        if not isinstance(o, list):
            issues.append(f"{where}: not a list (skipped)")
            continue
        op = o[0]
        if not isinstance(op, str) or op not in MARKET_OPS:
            issues.append(f"{where}: unknown order {op!r}")
            continue
        if MARKET_OPS[op] == 0:
            continue
        if len(o) < 3:
            issues.append(f"{where}: {op} needs an item and a count")
            continue
        if not _is_count(o[2]):
            issues.append(f"{where}: count {o[2]!r} is not a positive int")
        items = {"BUY_SEED": CROPS, "SELL": PRODUCTS, "BUY_PRODUCT":
                 ["WHEAT", "FERTILIZER"], "BUY_ANIMAL": ANIMALS}[op]
        if o[1] not in items:
            issues.append(f"{where}: {op} of {o[1]!r} is ignored by the "
                          f"engine")
    return issues


def _is_count(v) -> bool:
    if isinstance(v, bool):
        return False
    try:
        return int(v) > 0
    except (TypeError, ValueError):
        return False


def validate_tape(path: str, hand_counts=None):
    """{step: [issues]} for a single-seat tape (empty dict = clean).

    ``hand_counts[t]``, if given, is the number of hands at step ``t`` (for
    the alignment check); record it from a simulation of the tape.
    """
    _, lines = read_tape(path)
    out = {}
    for t, line in enumerate(lines):
        n = hand_counts[t] if hand_counts is not None \
            and t < len(hand_counts) else None
        issues = validate_action(line_to_action(line), n)
        if t >= N_ACTIONS:
            issues.append("step >= 719: never applied by the official runner")
        if issues:
            out[t] = issues
    return out


# ------------------------------------------------- tape -> runnable agent --

_AGENT_TEMPLATE = '''"""Open-loop tape agent generated by kaggsim.tape.tape_to_agent.

Replays a fixed action stream by observation step. It does not look at the
board, so it only reproduces the recorded game when the opponent's actions
and the episode seed also match.
"""

import re

_LINES = {lines!r}
_INT = re.compile(r"^[+-]?[0-9]+$")


def _tokens(seg):
    return [int(t) if _INT.match(t) else t for t in seg.split(" ") if t]


def _positional(field):
    return [] if field == "" else field.split(";")


def agent(obs, configuration=None):
    step = obs["step"] if isinstance(obs, dict) else obs.step
    if step >= len(_LINES):
        return {{"farmer": ["PASS"], "hands": [], "market": []}}
    parts = _LINES[step].split("\\t")
    parts += [""] * (3 - len(parts))
    action = {{"farmer": _tokens(parts[0]) or ["PASS"],
              "hands": [_tokens(h) for h in _positional(parts[1])],
              "market": [_tokens(o) for o in _positional(parts[2])]}}
    if {align!r}:
        me = obs["player"]
        n = len(obs["farms"][me]["hands"])
        hands = action["hands"][:n]
        action["hands"] = hands + [["PASS"]] * (n - len(hands))
    return action
'''


def tape_to_agent(tape_path: str, out_path: str, align_hands: bool = False):
    """Write a standalone ``main.py`` that replays a single-seat tape.

    With ``align_hands`` the hands list is trimmed/padded to the live hand
    count; the default replays the tape verbatim, which is what makes the
    agent reproduce ``kagg batch`` exactly on the official runner.
    """
    _, lines = read_tape(tape_path)
    src = _AGENT_TEMPLATE.format(lines=lines[:N_ACTIONS], align=align_hands)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    return out_path


# ---------------------------------------------------------------- splice --

class TapeError(ValueError):
    """A tape failed validation."""


def splice_tapes(prefix: str, suffix: str, at_step: int, out: str,
                 validate: bool = True) -> dict:
    """Write ``out`` = steps ``0..at_step-1`` of ``prefix`` + steps
    ``at_step..718`` of ``suffix``, with the SEED of ``prefix``.

    Missing lines (a tape shorter than the range) are empty actions. With
    ``validate`` both inputs are shape-checked over the steps actually used
    and :class:`TapeError` is raised on any issue.

    World split points: the first shop is drawn while step 71 runs, the
    second while step 143 runs. Splicing at ``at_step >= 72``
    (:data:`FIRST_SHOP_STEP`) keeps the prefix's first shop; at
    ``at_step >= 144`` (:data:`SECOND_SHOP_STEP`) it keeps both, for the same
    opponent stream and seat. Returns a small report dict.
    """
    at_step = int(at_step)
    if not 0 <= at_step <= N_ACTIONS:
        raise TapeError(f"at_step must be in 0..{N_ACTIONS}, got {at_step}")
    seed, pre = read_tape(prefix)
    _, suf = read_tape(suffix)
    if validate:
        problems = {}
        for name, lines, rng in (("prefix", pre, range(0, at_step)),
                                 ("suffix", suf, range(at_step, N_ACTIONS))):
            for t in rng:
                if t < len(lines):
                    issues = validate_action(line_to_action(lines[t]))
                    if issues:
                        problems[f"{name}[{t}]"] = issues
        if problems:
            first = sorted(problems.items())[:5]
            raise TapeError(f"{len(problems)} invalid step(s), e.g. {first}")
    empty = action_to_line({})
    lines = [(pre[t] if t < len(pre) else empty) if t < at_step
             else (suf[t] if t < len(suf) else empty)
             for t in range(N_ACTIONS)]
    write_tape(out, seed, lines)
    keeps = ("both shops" if at_step >= SECOND_SHOP_STEP else
             "first shop" if at_step >= FIRST_SHOP_STEP else "no shop")
    return {"out": out, "seed": seed, "at_step": at_step,
            "prefix_lines": min(len(pre), at_step),
            "suffix_lines": max(0, min(len(suf), N_ACTIONS) - at_step),
            "world_kept": keeps}


# ------------------------------------------------------------ batch jobs --

def batch_jobs(pairs, jobs_path: str) -> str:
    """Write a ``kagg batch`` jobs file.

    ``pairs`` is an iterable of ``(seed, tape_a, tape_b)``; seed ``None``
    means "use tape_a's SEED header".
    """
    with open(jobs_path, "w", encoding="utf-8", newline="\n") as fh:
        for seed, a, b in pairs:
            s = "-" if seed is None else str(int(seed))
            fh.write(f"{s}\t{os.path.abspath(a)}\t{os.path.abspath(b)}\n")
    return jobs_path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m kaggsim.tape",
                                 description="tape tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("from-replay", help="replay JSON -> single-seat tape")
    p.add_argument("replay")
    p.add_argument("seat", type=int)
    p.add_argument("out")
    p = sub.add_parser("validate", help="shape-check a single-seat tape")
    p.add_argument("tape")
    p = sub.add_parser("splice", help="prefix[0:at) + suffix[at:719]")
    p.add_argument("prefix")
    p.add_argument("suffix")
    p.add_argument("at_step", type=int)
    p.add_argument("out")
    p.add_argument("--no-validate", action="store_true")
    p = sub.add_parser("to-agent", help="tape -> standalone main.py")
    p.add_argument("tape")
    p.add_argument("out")
    p.add_argument("--align-hands", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "from-replay":
        replay_to_tape(load_replay(args.replay), args.seat, args.out)
    elif args.cmd == "validate":
        bad = validate_tape(args.tape)
        for t, issues in sorted(bad.items()):
            for i in issues:
                print(f"step {t}: {i}")
        print(f"{len(bad)} step(s) with issues")
        return 1 if bad else 0
    elif args.cmd == "splice":
        try:
            rep = splice_tapes(args.prefix, args.suffix, args.at_step,
                               args.out, not args.no_validate)
        except TapeError as exc:
            print(f"error: {exc}")
            return 1
        print(json.dumps(rep))
    elif args.cmd == "to-agent":
        tape_to_agent(args.tape, args.out, args.align_hands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
