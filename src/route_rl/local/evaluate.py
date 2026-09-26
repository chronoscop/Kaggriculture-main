"""Explicit, staged evaluation commands; no economic acceptance is inferred."""
import argparse,json,random
from pathlib import Path
from .settings import Settings,SCHEMA
from .runner import Serve,run_game,compare,summarize
from .policy import Policy,chooser
from .training import read_checkpoint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("baseline","keep","random","fixed","checkpoint"),default="keep")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--seeds",type=int,nargs="+",default=[9001,9002])
    p.add_argument("--out",type=Path)
    p.add_argument("--trace-dir",type=Path)
    p.add_argument("--verify-identity",action="store_true")
    p.add_argument("--max-changes",type=int,default=None)
    p.add_argument("--crop",choices=("WHEAT","CARROT","TOMATO","STRAWBERRY","MELON"),default="CARROT")
    p.add_argument("--stochastic",action="store_true",help="sample instead of deployment greedy gating")
    args=p.parse_args()
    if args.verify_identity and args.mode!="keep":p.error("identity comparison requires --mode keep")
    settings=Settings(max_changes=args.max_changes if args.max_changes is not None else 1)
    model=None
    if args.mode=="checkpoint":
        if not args.checkpoint:p.error("--checkpoint is required")
        import torch
        torch.set_num_threads(1)
        ck=read_checkpoint(args.checkpoint,torch);settings=Settings(**ck["settings"])
        if args.max_changes is not None and args.max_changes!=settings.max_changes:p.error("checkpoint decision limits cannot be changed during evaluation")
        model=Policy(torch,settings.initial_change_probability);model.module.load_state_dict(ck["network"]);model.module.eval()
    games=[]
    with Serve() as server:
        for seed in args.seeds:
            for seat in (0,1):
                reference=run_game(server,seed,seat,baseline=True)
                rng=random.Random(seed*2+seat)
                def choose(obs,event,candidates,ledger):
                    if args.mode=="fixed":
                        return next((i for i,c in enumerate(candidates) if c.kind=="CROP" and c.crop==args.crop),0)
                    if args.mode=="random" and rng.random()<settings.initial_change_probability:
                        return rng.randrange(1,len(candidates))
                    return 0
                if model:
                    torch.manual_seed(seed*2+seat)
                    choose=chooser(model,torch,"cpu",deterministic=not args.stochastic)
                tracefile=None
                if args.trace_dir:
                    args.trace_dir.mkdir(parents=True,exist_ok=True)
                    tracefile=(args.trace_dir/f"{seed}_{seat}.jsonl").open("w")
                try:
                    trace=(lambda row:tracefile.write(json.dumps(row)+"\n")) if tracefile else None
                    game=reference if args.mode=="baseline" else run_game(server,seed,seat,choose,settings,trace=trace,verify_keep=args.verify_identity)
                finally:
                    if tracefile:tracefile.close()
                result=compare(game,reference);games.append(result)
                print(json.dumps(result),flush=True)
    report=dict(schema=SCHEMA,mode=args.mode,deterministic=not args.stochastic,settings=settings.dict(),summary=summarize(games),games=games)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["summary"]),flush=True)

if __name__=="__main__":main()
