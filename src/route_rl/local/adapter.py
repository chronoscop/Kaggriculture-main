"""Runtime baseline adapter. The original source is loaded in isolated globals.

Hooks execute inside the baseline's crop-contract layer, before the final
inventory and market guards. KEEP calls the unmodified baseline functions.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
from collections import Counter
from .settings import Settings,SAFE,FIRST
from .contracts import Opportunity,Contract,crop_candidates

_COMPILED={}
def load_namespace(path):
    path=Path(path).resolve()
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest not in _COMPILED:_COMPILED[digest]=compile(raw,str(path),"exec")
    code=_COMPILED[digest]
    ns={"__file__":str(path)}
    exec(code,ns)
    entry=[v for v in ns.values() if callable(v)][-1]
    for name in ("_db_plan","_db_execute","_db_visits","_db_schedule","_db_reserve","_PLANNER_NS"):
        if name not in ns:raise ValueError("unsupported baseline interface: "+name)
    if not ns.get("DB_ENABLED") or ns.get("DB_SHADOW"):
        raise ValueError("baseline crop hook is disabled/shadow-only")
    return ns,entry,digest

class LocalEconomy:
    def __init__(self, choose=None, settings=None, baseline_path=None, trace=None):
        if baseline_path is None:
            from ..paths import BASELINE
            baseline_path=BASELINE
        self.path=Path(baseline_path)
        self.settings=(settings or Settings()).validate()
        self.choose=choose or (lambda *args:0)
        self.trace=trace
        self._load()

    def _load(self):
        self.ns,self.entry,self.baseline_sha256=load_namespace(self.path)
        self.original_plan=self.ns["_db_plan"];self.original_execute=self.ns["_db_execute"]
        self.ns["_db_plan"]=self._plan_hook
        self.ns["_db_execute"]=self._execute_hook
        self.contracts={};self.seen=set();self.credit=Counter()
        self.pending=[];self.pending_stock={}
        self.stats=Counter();self.extra_cash=0.;self.last_step=-1;self.error=None
        self.history=[];self.seat=None

    def log(self,event,obs,**data):
        if self.trace:self.trace(dict(event=event,step=obs["step"],**data))

    def _finish(self,pos,obs,reason):
        c=self.contracts.pop(pos)
        missing=set(c.candidate.plan.get("harvest_days",[]))-c.confirmed_harvest_days
        status=("partial" if missing else "completed") if reason=="harvested" else ("deferred" if reason=="deferred" else "cancelled")
        self.stats[status]+=1
        self.history.append(dict(key=c.opportunity.key,crop=c.candidate.crop,reason=reason,
                                 harvests=c.harvests,harvested_units=c.harvested_units,missing_harvest_days=sorted(missing)))
        self.log("contract_end",obs,pos=pos,reason=reason,harvests=c.harvests)

    def observe(self,obs):
        farm=obs["farms"][obs["player"]];private=obs["private"]
        for pos,actor,op,before,day in self.pending:
            c=self.contracts.get(pos)
            if c is None:continue
            tile=farm["tiles"][pos[1]][pos[0]]
            matches=isinstance(tile,dict) and tile.get("crop")==c.candidate.crop
            success=False
            if op=="PLANT":
                success=matches and tile.get("planted_day")==day
                if success:c.status="growing";c.planted_day=day
            elif op=="WATER":
                success=matches and (tile.get("watered_today") if obs["day"]==day else tile.get("consecutive_unwatered")==0)
            elif op=="HARVEST":
                after=private["inventories"][actor].get(c.candidate.crop,0) if actor<len(private["inventories"]) else 0
                quantity=max(0,after-before)
                success=quantity>0
                if success:
                    c.harvests+=1;c.harvested_units+=quantity
                    due=next((d for d in c.candidate.plan["harvest_days"] if d<=day and d not in c.confirmed_harvest_days),day)
                    c.confirmed_harvest_days.add(due);self.credit[c.candidate.crop]+=quantity
            elif op=="DIG":success=tile is None
            self.stats["confirmed_"+op.lower() if success else "receipt_misses"]+=1
        self.pending=[]
        for crop,stock in self.pending_stock.items():
            # No added BUY_PRODUCT for these crops. Baseline purchases make
            # this a conservative lower bound on actual sales.
            sold=max(0,stock-private["shed"].get(crop,0))
            self.credit[crop]=max(0,self.credit[crop]-sold)
        self.pending_stock={}
        for pos,c in list(self.contracts.items()):
            tile=farm["tiles"][pos[1]][pos[0]]
            if c.status=="await_release":
                if obs["step"]>c.opportunity.step and tile is None:c.status="await_seed"
                elif obs["step"]>c.opportunity.step:
                    self._finish(pos,obs,"original_harvest_failed")
                    continue
            if c.candidate.kind=="DEFER":
                if obs["step"]>c.candidate.plan["release_step"]:self._finish(pos,obs,"deferred")
                continue
            if c.status=="growing":
                if tile is None:
                    self._finish(pos,obs,"harvested" if c.harvests else "crop_lost")
                    continue
                if not isinstance(tile,dict) or tile.get("crop")!=c.candidate.crop or tile.get("planted_day")!=c.planted_day:
                    self._finish(pos,obs,"crop_replaced")
                    continue
                previous=obs["day"]-1
                if previous>=c.planted_day and tile.get("consecutive_unwatered",0)>0 and previous not in c.warned_days:
                    c.warned_days.add(previous);self.stats["missed_service_days"]+=1
                    self.log("service_window_missed",obs,pos=pos,day=previous)
            elif obs["step"]>c.candidate.plan["start"]+24:
                self._finish(pos,obs,"planting_window_missed")

    def _plan_hook(self,obs,action,st):
        # Reserve RL-owned sites from the baseline's own optional crop layer.
        placeholders=[]
        for pos,c in self.contracts.items():
            if pos not in st["sites"]:
                st["sites"][pos]=dict(crop=c.candidate.crop,status="rl_owned",start=c.candidate.plan["start"])
                placeholders.append(pos)
        try:self.original_plan(obs,action,st)
        finally:
            for pos in placeholders:st["sites"].pop(pos,None)
        self.stats["hook_calls"]+=1
        try:self._opportunities(obs,action,st)
        except Exception as exc:
            # Baseline wrappers catch exceptions; do not silently train on a
            # bypassed policy. Raise at our outer boundary after baseline returns.
            self.error=exc

    def _opportunities(self,obs,action,st):
        cfg=self.settings
        if obs["day"]<cfg.start_day or self.stats["changes"]>=cfg.max_changes or len(self.contracts)>=cfg.max_active:return
        farm=obs["farms"][obs["player"]]
        positions=[farm["farmer"]]+farm["hands"]
        units=[action.get("farmer") or ["PASS"]]+action.get("hands",[])
        targets=[]
        for actor,(pos,cmd) in enumerate(zip(positions,units)):
            pos=tuple(pos);tile=farm["tiles"][pos[1]][pos[0]]
            if not cmd or cmd[0]!="HARVEST" or pos in self.contracts or pos in st["sites"]:continue
            if not isinstance(tile,dict) or tile.get("crop") not in ("WHEAT","CARROT","MELON"):continue
            if tile.get("yield_units",0)<=0 or obs["day"]-tile["planted_day"]<FIRST[tile["crop"]]:continue
            key=f'{obs["player"]}:{pos}:{tile["crop"]}:{tile["planted_day"]}'
            if key not in self.seen:targets.append((key,pos,actor,tile["crop"]))
        if not targets:return
        visits=self.ns["_db_visits"](obs,action)
        for key,pos,actor,old in targets:
            if self.stats["changes"]>=cfg.max_changes or len(self.contracts)>=cfg.max_active:break
            path=visits.get(pos,[])
            future=next(((t,i) for t,i,op in path if op=="PLANT" and obs["step"]<t<=obs["step"]+24),None)
            if future is None:continue
            tape=self.ns["_ca_tape"](obs["player"],future[0])
            cmds=[tape.get("farmer") or ["PASS"]]+tape.get("hands",[])
            original=cmds[future[1]][1] if future[1]<len(cmds) and len(cmds[future[1]])>1 else old
            opportunity=Opportunity(key,obs["step"],pos,actor,original,old)
            self.seen.add(key)
            cash=max(0,min(cfg.max_extra_cash-self.extra_cash,
                          farm["money"]-self.ns["_db_reserve"](obs,action)))
            occupied={(t,i) for c in self.contracts.values() for t,i,_ in c.candidate.slots}
            candidates=crop_candidates(self.ns,obs,action,opportunity,path,occupied,cash)
            if len(action.get("market",[]))>=10:
                candidates=[c for c in candidates if c.kind!="CROP"]
            if len(candidates)==1:
                self.stats["no_alternative"]+=1;continue
            index=int(self.choose(obs,opportunity,candidates,dict(active=len(self.contracts),changes=self.stats["changes"])))
            if not 0<=index<len(candidates):raise ValueError("invalid local economic choice")
            self.stats["opportunities"]+=1
            self.log("decision",obs,key=key,pos=pos,selected=index,
                     candidates=[dict(kind=c.kind,crop=c.crop,cost=c.cost,plan=c.plan,slots=c.slots) for c in candidates])
            if index==0:self.stats["kept"]+=1;continue
            candidate=candidates[index]
            self.contracts[pos]=Contract(opportunity,candidate)
            self.stats["changes"]+=1;self.extra_cash+=candidate.cost
            if candidate.kind=="CROP":
                # Purchase a dedicated seed rather than spend seeds reserved by
                # another baseline project. Plant only on a later real receipt.
                action["market"].append(["BUY_SEED",candidate.crop,1])
            self.log("contract_start",obs,key=key,pos=pos,crop=candidate.crop)

    def _execute_hook(self,obs,action,st):
        self.original_execute(obs,action,st)
        if not self.contracts:return
        try:self._execute(obs,action)
        except Exception as exc:self.error=exc

    def _execute(self,obs,action):
        ns=self.ns;farm=obs["farms"][obs["player"]]
        f,pr=ns["_PLANNER_NS"]["_clone_state"](farm,obs["private"])
        positions=[farm["farmer"]]+farm["hands"]
        units=[list(action.get("farmer") or ["PASS"])]+[list(c or ["PASS"]) for c in action.get("hands",[])]
        reserved=Counter(cmd[1] for pos,cmd in zip(positions,units)
                         if cmd[0]=="PLANT" and tuple(pos) not in self.contracts)
        for actor,(pos,cmd) in enumerate(zip(positions,units)):
            pos=tuple(pos);c=self.contracts.get(pos)
            if c and c.status!="await_release" and cmd[0] in SAFE:
                tile=f["tiles"][pos[1]][pos[0]];plan=c.candidate.plan;crop=c.candidate.crop
                chosen=["PASS"]
                if c.candidate.kind=="CROP":
                    if tile is None and c.status=="await_seed" and (obs["step"],actor)>=(plan["start"],plan["start_unit"]):
                        if pr["seeds"].get(crop,0)-reserved[crop]>0:
                            # Recheck same-day watering using actual positions.
                            later=ns["_ca_visits"](obs,action,pos,min(718,(obs["day"]+1)*24-1),start=obs["step"])
                            if any((t,i)>(obs["step"],actor) and op in SAFE for t,i,op in later):chosen=["PLANT",crop]
                    elif isinstance(tile,dict) and tile.get("crop")==crop:
                        age=obs["day"]-tile["planted_day"]
                        due=any(day<=obs["day"] and day not in c.confirmed_harvest_days for day in plan["harvest_days"])
                        if crop in ("TOMATO","STRAWBERRY") and obs["day"]>=plan["finish"] and c.harvests and not tile.get("yield_units",0):chosen=["DIG"]
                        elif not tile.get("watered_today"):chosen=["WATER"]
                        elif tile.get("yield_units",0)>0 and age>=FIRST[crop] and due:chosen=["HARVEST"]
                if chosen!=cmd:
                    self.stats["unit_edits"]+=1
                    self.log("unit_edit",obs,actor=actor,pos=pos,before=cmd,after=chosen)
                units[actor]=chosen
                if chosen[0] in ("PLANT","WATER","HARVEST","DIG"):
                    before=pr["inventories"][actor].get(crop,0)
                    self.pending.append((pos,actor,chosen[0],before,obs["day"]))
            ns["_PLANNER_NS"]["_apply_unit_action"](f,pr,actor,units[actor],10,obs["day"],24,100)
        action["farmer"]=units[0];action["hands"]=units[1:]

    def act(self,obs,configuration=None):
        if self.last_step>=0 and (obs["step"]<=self.last_step or obs["player"]!=self.seat):self._load()
        self.seat=obs["player"];self.observe(obs)
        action=self.entry(obs,configuration)
        if self.error is not None:raise RuntimeError("local baseline hook failed") from self.error
        # Liquidate only confirmed incremental crop output not already sold by
        # the baseline. Wheat remains under its feed/sales management.
        for crop,credit in self.credit.items():
            if crop=="WHEAT" or credit<=0:continue
            already=sum(o[2] for o in action.get("market",[]) if o and o[:2]==["SELL",crop])
            n=min(credit,max(0,obs["private"]["shed"].get(crop,0)-already))
            if n and len(action["market"])<10:action["market"].append(["SELL",crop,int(n)])
        final_units=[action.get("farmer") or ["PASS"]]+action.get("hands",[])
        retained=[]
        for record in self.pending:
            pos,actor,op,before,day=record
            cmd=final_units[actor] if actor<len(final_units) else ["PASS"]
            c=self.contracts.get(pos)
            matches=cmd and cmd[0]==op and (op!="PLANT" or c and len(cmd)>1 and cmd[1]==c.candidate.crop)
            if matches:retained.append(record)
            else:
                self.stats["overridden_after_hook"]+=1
                self.log("baseline_guard_override",obs,pos=pos,actor=actor,requested=op,actual=cmd)
        self.pending=retained
        if self.credit:
            stock=self.ns["projected_shed"](action,self.ns["FarmView"](obs))
            self.pending_stock={crop:stock.get(crop,0) for crop in self.credit}
        self.last_step=obs["step"]
        return action

    def report(self):
        return dict(self.stats,active_contracts=len(self.contracts),extra_cash_committed=self.extra_cash,
                    contracts=self.history,unfinished=[c.opportunity.key for c in self.contracts.values()])
