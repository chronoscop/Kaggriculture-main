"""Plans contain future service reservations, not just a replacement crop."""
from dataclasses import dataclass, field
from .settings import CROPS, SAFE, COST

@dataclass(frozen=True)
class Candidate:
    kind: str
    crop: str = ""
    plan: dict = field(default_factory=dict)
    cost: float = 0.
    slots: tuple = ()

@dataclass(frozen=True)
class Opportunity:
    key: str
    step: int
    pos: tuple
    actor: int
    original_crop: str
    old_crop: str

@dataclass
class Contract:
    opportunity: Opportunity
    candidate: Candidate
    status: str = "await_release"
    planted_day: int = -1
    harvests: int = 0
    harvested_units: int = 0
    confirmed_harvest_days: set = field(default_factory=set)
    warned_days: set = field(default_factory=set)

def crop_candidates(ns, obs, action, opportunity, visits, occupied_slots, cash_available):
    """All crops considered. Only execution/resource feasibility is filtered."""
    result = [Candidate("KEEP", opportunity.original_crop)]
    for crop in CROPS:
        if crop == opportunity.original_crop or COST[crop] > cash_available:
            continue
        plan = ns["_db_schedule"](crop, visits, opportunity.step + 1)
        if plan is None:
            continue
        start = (plan["start"], plan["start_unit"])
        slots = [(start[0],start[1],"PLANT")]
        usable = [(t,i,op) for t,i,op in visits if (t,i)>start and op in SAFE]
        valid = True
        for day in range(plan["born"],plan["finish"]+1):
            on_day = [(t,i) for t,i,_ in usable if t//24==day]
            need = 2 if day in plan["harvest_days"] else 1
            if len(on_day)<need:
                valid=False
                break
            slots.append((*on_day[0],"WATER"))
            if need==2:
                slots.append((*on_day[1],"HARVEST"))
        if not valid:
            continue
        if crop in ("TOMATO","STRAWBERRY"):
            final = max((t,i) for t,i,op in slots if op=="HARVEST")
            clear = next(((t,i) for t,i,_ in usable if (t,i)>final and t<=min(718,final[0]+24)),None)
            if clear is None:
                continue
            if any(op not in SAFE for t,i,op in visits if final<(t,i)<=clear):
                continue
            slots.append((*clear,"DIG"))
        if any((t,i) in occupied_slots for t,i,_ in slots):
            continue
        plan=dict(plan,release_step=max(t for t,_,_ in slots))
        result.append(Candidate("CROP",crop,plan,float(COST[crop]),tuple(slots)))
    # Defer this planting window; resume the baseline from the following day.
    first=next(((t,i) for t,i,op in visits if t>opportunity.step and op=="PLANT"),None)
    if first:
        end=min(718,(first[0]//24+1)*24-1)
        blocked=tuple((t,i,"PASS") for t,i,op in visits if first[0]<=t<=end and op in SAFE)
        if blocked and not any((t,i) in occupied_slots for t,i,_ in blocked):
            result.append(Candidate("DEFER",plan=dict(start=first[0],born=first[0]//24,
                finish=end//24,release_step=end,harvest_days=[]),slots=blocked))
    return result
