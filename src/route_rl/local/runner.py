"""Paired full-season economics; reference trajectories are never policy inputs."""
from collections import Counter
from ..paths import BASELINE,add_kaggsim
from .adapter import LocalEconomy
from .settings import Settings
add_kaggsim()
from kaggsim.serve import Serve,load_agent,call_agent,obs_for

def run_game(server,seed,seat,choose=None,settings=None,baseline=False,trace=None,verify_keep=False):
    state=server.reset(seed)
    opponent=load_agent(str(BASELINE))
    controller=None if baseline else LocalEconomy(choose,settings or Settings(),trace=trace)
    own=load_agent(str(BASELINE)) if baseline else controller.act
    oracle=load_agent(str(BASELINE)) if verify_keep else None
    checked=0
    while state["step"]<719:
        view=obs_for(state,seat)
        action=call_agent(own,view)
        if oracle:
            import copy
            expected=call_agent(oracle,copy.deepcopy(view))
            if action!=expected:
                raise AssertionError(f"KEEP differs from baseline: seed={seed} seat={seat} step={state['step']}")
            checked+=1
        other=call_agent(opponent,obs_for(state,1-seat))
        state=server.step2(action,other) if seat==0 else server.step2(other,action)
    if controller:controller.observe(obs_for(state,seat))
    return dict(seed=seed,seat=seat,own_cash=state["farms"][seat]["money"],
                opponent_cash=state["farms"][1-seat]["money"],
                stats=controller.report() if controller else {},identity_checked_steps=checked)

def compare(policy,reference):
    margin=policy["own_cash"]-policy["opponent_cash"]
    baseline_margin=reference["own_cash"]-reference["opponent_cash"]
    return dict(policy,margin=margin,reference_own_cash=reference["own_cash"],
                reference_opponent_cash=reference["opponent_cash"],
                delta_cash=policy["own_cash"]-reference["own_cash"],
                delta_margin=margin-baseline_margin)

def summarize(games):
    import statistics
    if not games:return dict(games=0)
    ds=[g["delta_margin"] for g in games]
    # Seats share a seed; aggregate before computing uncertainty.
    by_seed={}
    for g in games:by_seed.setdefault(g["seed"],[]).append(g["delta_margin"])
    paired=[statistics.mean(x) for x in by_seed.values()]
    std=statistics.stdev(paired) if len(paired)>1 else None
    counters=Counter()
    for g in games:
        for k,v in g["stats"].items():
            if isinstance(v,(int,float)):counters[k]+=v
    total=counters.get("opportunities",0)
    return dict(games=len(games),seeds=len(by_seed),mean_delta_margin=statistics.mean(ds),
                median_delta_margin=statistics.median(ds),mean_delta_cash=statistics.mean(g["delta_cash"] for g in games),
                worst_quartile_delta_margin=statistics.mean(sorted(ds)[:max(1,len(ds)//4)]),
                paired_seed_standard_error=std/(len(paired)**.5) if std is not None else None,
                win_rate=sum(g["margin"]>0 for g in games)/len(games),
                improvement_rate=sum(d>0 for d in ds)/len(ds),
                modification_rate=counters.get("changes",0)/total if total else 0.,
                totals=dict(counters))
