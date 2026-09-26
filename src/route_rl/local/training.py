"""Event-level PPO on baseline-preserving economic modifications; no BC stage."""
from __future__ import annotations
import argparse,hashlib,json,random,time
from pathlib import Path
from dataclasses import asdict
from ..paths import BASELINE
from .settings import Settings,SCHEMA,ENGINE
from .encoding import CONTEXT_SIZE,CANDIDATE_SIZE
from .policy import Policy,chooser
from .runner import Serve,run_game,compare,summarize

def batch(rows,torch,device):
    width=max(len(r["features"]) for r in rows)
    x=torch.zeros((len(rows),width,CANDIDATE_SIZE),device=device)
    mask=torch.zeros((len(rows),width),dtype=torch.bool,device=device)
    for i,r in enumerate(rows):
        n=len(r["features"]);x[i,:n]=torch.tensor(r["features"],device=device);mask[i,:n]=True
    context=torch.tensor([r["context"] for r in rows],device=device)
    return context,x,mask

def update(model,optimizer,torch,episodes,device,epochs=2,batch_size=64):
    rows=[r for trajectory in episodes for r in trajectory]
    if not rows:return dict(samples=0,updates=0)
    advantage=torch.tensor([r["return"]-r["value"] for r in rows],device=device)
    # No per-minibatch normalization: with one decision, normalization would
    # erase all learning. Scale by dispersion without forcing zero mean.
    if len(rows)>1:advantage=advantage/advantage.std(unbiased=False).clamp_min(1.)
    old=torch.tensor([r["logp"] for r in rows],device=device)
    actions=torch.tensor([r["action"] for r in rows],device=device)
    returns=torch.tensor([r["return"] for r in rows],device=device)
    losses=[];kls=[];stopped=False
    for _ in range(epochs):
        for ids in torch.randperm(len(rows),device=device).split(batch_size):
            selected=[rows[i] for i in ids.tolist()]
            c,x,m=batch(selected,torch,device)
            lp,value=model(c,x,m);new=lp.gather(1,actions[ids,None]).squeeze(1)
            ratio=(new-old[ids]).exp()
            kl=float(((ratio-1)-(new-old[ids])).mean().detach())
            if kl>.02:
                stopped=True;break
            dist=torch.distributions.Categorical(logits=lp)
            loss=-torch.minimum(ratio*advantage[ids],ratio.clamp(.9,1.1)*advantage[ids]).mean()
            loss=loss+.5*(value-returns[ids]).square().mean()-.001*dist.entropy().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.module.parameters(),.5);optimizer.step()
            losses.append(float(loss.detach()));kls.append(kl)
        if stopped:break
    return dict(samples=len(rows),updates=len(losses),loss=sum(losses)/len(losses) if losses else None,
                mean_kl=sum(kls)/len(kls) if kls else None,kl_stopped=stopped)

def evaluate(server,model,torch,settings,seeds,device,reference_cache,deterministic=True):
    games=[]
    for seed in seeds:
        for seat in (0,1):
            key=(seed,seat)
            if key not in reference_cache:reference_cache[key]=run_game(server,seed,seat,baseline=True)
            # Matched RNG across checkpoints reduces sampling noise. Independent
            # seeds, not current game outcomes, select evaluation randomness.
            torch.manual_seed(seed*2+seat)
            game=run_game(server,seed,seat,chooser(model,torch,device,deterministic=deterministic),settings)
            games.append(compare(game,reference_cache[key]))
    return dict(summary=summarize(games),games=games,deterministic=deterministic)

def read_checkpoint(path,torch):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck.get("schema")!=SCHEMA:raise ValueError("v4 baseline-local checkpoint required; v2/v3/BC checkpoints are incompatible")
    if ck.get("engine")!=ENGINE or ck.get("baseline_sha256")!=hashlib.sha256(BASELINE.read_bytes()).hexdigest():
        raise ValueError("baseline/engine provenance differs")
    Settings(**ck["settings"]).validate()
    return ck

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes",type=int,default=100,help="cumulative update count")
    p.add_argument("--games-per-update",type=int,default=16)
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--out",type=Path,default=Path("runs/local_v4"))
    p.add_argument("--device",default="auto")
    p.add_argument("--eval-every",type=int,default=5)
    p.add_argument("--eval-seeds",type=int,nargs="+",default=[9001,9002])
    p.add_argument("--start-day",type=int)
    p.add_argument("--max-changes",type=int,help="accepted modifications per season; default 1")
    p.add_argument("--max-active",type=int)
    p.add_argument("--max-extra-cash",type=float)
    p.add_argument("--change-probability",type=float)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--resume",type=Path)
    args=p.parse_args()
    if args.episodes<1 or args.games_per_update<2 or args.games_per_update%2 or args.eval_every<1 or args.lr<=0:
        p.error("positive episodes/eval-every/lr and even games-per-update >=2 required")
    train_seeds=set(range(args.seed+1,args.seed+1+args.episodes*(args.games_per_update//2)))
    if train_seeds & set(args.eval_seeds):p.error("training/evaluation seed overlap")
    if len(args.eval_seeds)!=len(set(args.eval_seeds)):p.error("evaluation seeds must be unique")
    import torch
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    checkpoint=read_checkpoint(args.resume,torch) if args.resume else None
    base=dict(checkpoint["settings"]) if checkpoint else Settings().dict()
    for arg,key in (("start_day","start_day"),("max_changes","max_changes"),("max_active","max_active"),
                    ("max_extra_cash","max_extra_cash"),("change_probability","initial_change_probability")):
        value=getattr(args,arg)
        if value is not None:
            if checkpoint and value!=base[key]:p.error("resume must preserve settings; use a new experiment for changed scope")
            base[key]=value
    settings=Settings(**base).validate()
    if settings.max_changes==0:p.error("max-changes=0 is an identity evaluation, not a training run")
    model=Policy(torch,settings.initial_change_probability);model.module.to(device)
    optimizer=torch.optim.Adam(model.module.parameters(),lr=args.lr)
    first=1;best=float("-inf")
    if checkpoint:
        if checkpoint["seed"]!=args.seed or checkpoint["games_per_update"]!=args.games_per_update:
            p.error("resume must preserve seed and games-per-update")
        if checkpoint.get("eval_seeds")!=args.eval_seeds:p.error("resume must preserve evaluation seeds")
        if checkpoint.get("lr")!=args.lr:p.error("resume must preserve lr")
        model.module.load_state_dict(checkpoint["network"]);optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.startswith("cuda") and checkpoint.get("cuda_rng") is not None:torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        first=checkpoint["iteration"]+1;best=checkpoint.get("best_delta_margin",best)
    elif (args.out/"latest.pt").exists():p.error("output already has a run; use --resume or another --out")
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema=SCHEMA,engine=ENGINE,baseline_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
                  settings=settings.dict(),seed=args.seed,games_per_update=args.games_per_update,
                  eval_seeds=args.eval_seeds,lr=args.lr,
                  objective="paired full-season margin improvement / 10000",initialization="baseline_keep_gate")
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    reference_cache={}
    with Serve() as server:
        for iteration in range(first,args.episodes+1):
            episodes=[];games=[];started=time.perf_counter()
            for i in range(args.games_per_update):
                seed=args.seed+(iteration-1)*(args.games_per_update//2)+i//2+1;seat=i%2
                key=seed,seat
                if key not in reference_cache:reference_cache[key]=run_game(server,seed,seat,baseline=True)
                rows=[]
                game=run_game(server,seed,seat,chooser(model,torch,device,rows),settings)
                game=compare(game,reference_cache[key]);reward=game["delta_margin"]/10000.
                for row in rows:row["return"]=reward
                episodes.append(rows);games.append(game)
                print(json.dumps(dict(stage="rollout",iteration=iteration,game=i+1,
                                      seed=seed,seat=seat,own_cash=game["own_cash"],delta_margin=game["delta_margin"],
                                      events=len(rows),changes=game["stats"].get("changes",0))),flush=True)
            rollout_seconds=time.perf_counter()-started
            started=time.perf_counter()
            metrics=update(model,optimizer,torch,episodes,device)
            metrics.update(iteration=iteration,rollout_seconds=rollout_seconds,update_seconds=time.perf_counter()-started,
                           summary=summarize(games),games=games,device=device)
            with (args.out/"metrics.jsonl").open("a") as f:f.write(json.dumps(metrics)+"\n")
            print(json.dumps(dict(stage="update",iteration=iteration,summary=metrics["summary"],samples=metrics["samples"])),flush=True)
            if iteration%args.eval_every==0 or iteration==args.episodes:
                cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
                report=evaluate(server,model,torch,settings,args.eval_seeds,device,reference_cache)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
                report["iteration"]=iteration
                with (args.out/"eval.jsonl").open("a") as f:f.write(json.dumps(report)+"\n")
                score=report["summary"]["mean_delta_margin"]
                if score>best:
                    best=score
                    torch.save(dict(manifest,network=model.module.state_dict(),iteration=iteration,
                                    validation=report,best_delta_margin=best),args.out/"best.pt")
                print(json.dumps(dict(stage="evaluation",iteration=iteration,summary=report["summary"])),flush=True)
            torch.save(dict(manifest,network=model.module.state_dict(),optimizer=optimizer.state_dict(),
                            iteration=iteration,best_delta_margin=best,torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None),args.out/"latest.pt")

if __name__=="__main__":main()
