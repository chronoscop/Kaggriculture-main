"""Isolated JSON-lines evaluation worker with checkpoint-specific execution."""
from contextlib import redirect_stdout
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def main() -> None:
    os.environ['KAGGRICULTURE_RAISE_AGENT_ERRORS'] = '1'
    kind, path = sys.argv[1], Path(sys.argv[2]).resolve()
    with redirect_stdout(sys.stderr):
        if kind == 'policy':
            from .inference import load_policy
            policy = load_policy(path)
            agent = lambda observation, configuration: policy(observation)
        elif kind == 'entry':
            context = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else path
            sys.path.insert(0, str(context.parent))
            spec = importlib.util.spec_from_file_location('_paired_opponent', context)
            module = importlib.util.module_from_spec(spec)
            exec(compile(path.read_bytes(), str(context), 'exec'), module.__dict__)
            agent = module.agent
        else:
            raise ValueError('unknown evaluation worker kind')
    print(json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                action = agent(request['observation'], request['configuration'])
            response = {'action': action}
        except Exception:
            traceback.print_exc(file=sys.stderr)
            response = {'error': 'agent execution failed; see stderr'}
        print(json.dumps(response), flush=True)


if __name__ == '__main__':
    main()
