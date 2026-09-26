"""Explicit v4 submission measurement; no training or automatic upload."""
import argparse,json,select,subprocess,sys,tempfile
from pathlib import Path
from .archive import unpack,MAX_RAM
from ..paths import BASELINE,add_kaggsim
add_kaggsim()
from kaggsim.serve import Serve,obs_for,load_agent,call_agent

def read(proc):
    if not select.select([proc.stdout],[],[],30)[0]:raise TimeoutError("submission exceeded 30-second watchdog")
    line=proc.stdout.readline()
    if not line:raise RuntimeError(proc.stderr.read()[-3000:])
    return json.loads(line)

def check(archive,seeds,max_seconds):
    games=[];timings=[];peak=0;startups=[]
    with tempfile.TemporaryDirectory(prefix="local-v4-check-") as tmp:
        root=Path(tmp)/"kaggle_simulations"/"agent"
        unpacked=unpack(archive,root)
        metadata=json.loads((root/"_local_economy"/"metadata.json").read_text())
        with Serve() as server:
            for seed in seeds:
                for seat in (0,1):
                    state=server.reset(seed);opponent=load_agent(str(BASELINE))
                    proc=subprocess.Popen([sys.executable,"-I","-S","-u",str(Path(__file__).with_name("resource_probe.py")),str(root/"main.py")],
                                          cwd=tmp,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
                    try:
                        startups.append(read(proc))
                        while state["step"]<719:
                            proc.stdin.write(json.dumps(obs_for(state,seat))+"\n");proc.stdin.flush()
                            result=read(proc);timings.append(result["seconds"]);peak=max(peak,result["rss_bytes"])
                            other=call_agent(opponent,obs_for(state,1-seat))
                            state=server.step2(result["action"],other) if seat==0 else server.step2(other,result["action"])
                        games.append(dict(seed=seed,seat=seat,own_cash=state["farms"][seat]["money"],opponent_cash=state["farms"][1-seat]["money"]))
                    finally:
                        proc.terminate()
                        try:proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
                        for stream in (proc.stdin,proc.stdout,proc.stderr):stream.close()
    times=sorted(timings)
    return dict(passed=peak<MAX_RAM and max(times)<max_seconds and all(not s["heavy_imports"] for s in startups),
                archive=str(archive),compressed_bytes=Path(archive).stat().st_size,unpacked_bytes=unpacked,
                peak_rss_bytes=peak,max_action_seconds=max(times),p99_action_seconds=times[int(.99*(len(times)-1))],
                startups=startups,games=games,metadata=metadata,
                limitations="Measured games only; synthetic stress and target-platform validation are separate pending stages.")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True);p.add_argument("--seeds",type=int,nargs="+",default=[9103])
    p.add_argument("--max-action-seconds",type=float,default=1.);p.add_argument("--out",type=Path)
    a=p.parse_args();report=check(a.archive,a.seeds,a.max_action_seconds)
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["passed"]:raise SystemExit(1)
if __name__=="__main__":main()
