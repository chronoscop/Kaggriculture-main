"""Record a game as tapes, validate them, and turn one into a main.py."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src-python"))

from kaggsim.batch import run_batch  # noqa: E402
from kaggsim.policies import RandomPolicy, ScriptedFarmer  # noqa: E402
from kaggsim.serve import run_match  # noqa: E402
from kaggsim.tape import (action_to_line, tape_to_agent, validate_tape,  # noqa: E402,E501
                          write_tape)

out = tempfile.mkdtemp()
banks, trace = run_match(ScriptedFarmer(4), RandomPolicy(5), 21, record=True)
tapes = []
for seat in (0, 1):
    p = os.path.join(out, f"seat{seat}.tape")
    write_tape(p, 21, [action_to_line(t["actions"][seat]) for t in trace])
    tapes.append(p)
    print(p, "issues:", len(validate_tape(p)))
print("serve banks:", banks)
print("batch banks:", run_batch([(None, tapes[0], tapes[1])]))
print("agent file :", tape_to_agent(tapes[0], os.path.join(out, "main.py")))
