"""Play two agents on the Rust engine; optionally check the official runner.

    python examples/play_match.py                      # two built-in policies
    python examples/play_match.py a.py b.py --seed 3 --official
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src-python"))

from kaggsim.policies import RandomPolicy, ScriptedFarmer  # noqa: E402
from kaggsim.serve import load_agent, run_match  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agents", nargs="*")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--official", action="store_true",
                    help="also play on the official runner (needs "
                         "kaggle-environments==1.32.7)")
    args = ap.parse_args()

    def make():
        if len(args.agents) == 2:
            return load_agent(args.agents[0]), load_agent(args.agents[1])
        return ScriptedFarmer(1), RandomPolicy(2)

    banks = run_match(*make(), args.seed)
    print(f"rust     : {banks[0]:,.0f} / {banks[1]:,.0f}")
    if args.official:
        from kaggsim import official
        off, _ = official.run_agents(*make(), args.seed)
        print(f"official : {off[0]:,.0f} / {off[1]:,.0f} -> "
              f"{'MATCH' if tuple(off) == tuple(banks) else 'MISMATCH'}")


if __name__ == "__main__":
    main()
