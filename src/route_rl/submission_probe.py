"""Isolated stdlib-only subprocess used by submission_check; not packaged."""
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import time

def main():
    if hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(allowed)})
    started = time.perf_counter()
    namespace = runpy.run_path(str(Path(sys.argv[1]).resolve()))
    agent = namespace["agent"]
    print(json.dumps(dict(ready=True, startup_seconds=time.perf_counter() - started,
        rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        cpu_affinity=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        imported_heavy_libraries=sorted(set(sys.modules) & {"torch", "numpy", "kaggsim"}))),
        flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        policy = namespace["_submission"].policy
        original = policy.choose
        if request.get("stress"):
            # Deliberately force many legal decisions after scoring every
            # candidate. This is a performance probe, not a policy evaluation.
            def stress_choose(plan, jobs):
                before = policy.budget_cutoffs
                choice = original(plan, jobs)
                if policy.budget_cutoffs > before:
                    return choice
                return next((i for i, j in enumerate(jobs) if j.op[0] == "BUY_SEED"), 0)
            policy.choose = stress_choose
        start = time.perf_counter()
        action = agent(request["observation"], request.get("configuration"))
        duration = time.perf_counter() - start
        policy.choose = original
        print(json.dumps(dict(action=action, seconds=duration,
            rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            budget_cutoffs=namespace["_submission"].policy.budget_cutoffs)), flush=True)

if __name__ == "__main__":
    main()
