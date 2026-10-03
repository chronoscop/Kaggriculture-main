"""Private JSON-lines worker for isolated candidate evaluation."""
from contextlib import redirect_stdout
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def main() -> None:
    os.environ["KAGGRICULTURE_RAISE_AGENT_ERRORS"] = "1"
    kind, path = sys.argv[1], Path(sys.argv[2]).resolve()
    with redirect_stdout(sys.stderr):
        if kind == "policy":
            from .inference import load_policy
            policy = load_policy(path)
            agent = lambda observation, configuration: policy(observation)
        elif kind == "entry":
            sys.path.insert(0, str(path.parent))
            spec = importlib.util.spec_from_file_location("_evaluation_agent", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            agent = module.agent
        else:
            raise ValueError("unknown evaluation worker kind")
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                action = agent(request["observation"], request["configuration"])
            response = {"action": action}
        except Exception:
            traceback.print_exc(file=sys.stderr)
            response = {"error": "agent execution failed; see stderr"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
