"""Independent held-out evaluation: native v7 agent versus the frozen farm2945 reference."""
from __future__ import annotations
import argparse, hashlib, json, selectors, subprocess, time
from pathlib import Path
from .paths import PROJECT_ROOT, BASELINE, BASELINE_NAME, add_kaggsim

class NativeAgent:
    def __init__(self, binary: Path, checkpoint: str):
        self.process = subprocess.Popen([str(binary), '--checkpoint', checkpoint], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
    def __call__(self, observation):
        self.process.stdin.write(json.dumps(observation, separators=(',', ':'))+'\n')
        self.process.stdin.flush()
        if not self.selector.select(timeout=60):
            raise TimeoutError('native agent did not answer within 60 seconds')
        line = self.process.stdout.readline()
        if not line: raise RuntimeError(f'native agent exited: {self.process.poll()}')
        return json.loads(line)
    def close(self):
        self.selector.close()
        if self.process.poll() is None:
            self.process.stdin.close()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
        self.process.stdout.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,help='native v7 checkpoint path, or heuristic')
    p.add_argument('--seeds',type=int,nargs='+',default=[9001,9002])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--binary',type=Path,default=PROJECT_ROOT/'native/target/release/mixed-agent')
    a=p.parse_args()
    if not a.binary.is_file():p.error('build native mixed-agent with --features train first')
    if len(set(a.seeds))!=len(a.seeds):p.error('duplicate evaluation seeds')
    checkpoint=a.checkpoint
    if checkpoint!='heuristic':
        path=Path(checkpoint).resolve();ck=json.loads(path.read_text())
        if ck.get('schema')!='mixed-production-v7-ppo-v1':p.error('requires a v7 native checkpoint')
        run=ck.get('run',{});start=int(run['seed']);count=int(ck['iteration'])*int(run['games_per_update'])//2
        if any(start<=seed<start+count for seed in a.seeds):p.error('evaluation seeds overlap recorded training seeds')
        checkpoint=str(path)
    add_kaggsim()
    from kaggsim.serve import Serve,load_agent,call_agent,obs_for
    games=[]
    with Serve() as server:
        for seed in a.seeds:
            for seat in (0,1):
                native=NativeAgent(a.binary.resolve(),checkpoint)
                try:
                    baseline=load_agent(str(BASELINE));state=server.reset(seed);seconds=[0.,0.];started=time.perf_counter()
                    while state['step']<719:
                        t=time.perf_counter();own=native(obs_for(state,seat));seconds[0]+=time.perf_counter()-t
                        t=time.perf_counter();other=call_agent(baseline,obs_for(state,1-seat));seconds[1]+=time.perf_counter()-t
                        state=server.step2(own,other) if seat==0 else server.step2(other,own)
                    own=state['farms'][seat]['money'];other=state['farms'][1-seat]['money']
                    game=dict(seed=seed,seat=seat,own_cash=own,opponent_cash=other,margin=own-other,seconds=time.perf_counter()-started,native_agent_seconds=seconds[0],reference_agent_seconds=seconds[1])
                    games.append(game);print(json.dumps(game),flush=True)
                finally:native.close()
    report=dict(schema='mixed-production-v7-evaluation-v1',checkpoint=checkpoint,baseline_name=BASELINE_NAME,baseline_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),games=games,mean_margin=sum(g['margin'] for g in games)/len(games),win_rate=sum(g['margin']>0 for g in games)/len(games))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
