"""Isolated stdlib-only process; intentionally excluded from submission."""
import json,os,resource,runpy,sys,time
def main():
    if hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    start=time.perf_counter()
    ns=runpy.run_path(sys.argv[1])
    print(json.dumps(dict(startup_seconds=time.perf_counter()-start,
                          heavy_imports=sorted(set(sys.modules)&{"torch","numpy","kaggsim"}),
                          cpu_affinity=len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None)),flush=True)
    for line in sys.stdin:
        obs=json.loads(line);start=time.perf_counter()
        action=ns["agent"](obs)
        print(json.dumps(dict(action=action,seconds=time.perf_counter()-start,
                              rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)),flush=True)
if __name__=="__main__":main()
