"""Small state/candidate inputs; no old profit ranking used as action labels."""
from .settings import CROPS,PRODUCTS,SHOPS
CONTEXT_SIZE, CANDIDATE_SIZE = 96, 32

def padded(values,n):
    if len(values)>n: raise ValueError("feature overflow")
    return values+[0.]*(n-len(values))

def encode(obs, opportunity, candidates, active, changes):
    farm=obs["farms"][obs["player"]];private=obs["private"]
    crops={c:0 for c in CROPS};animals=0
    for row in farm["tiles"]:
        for tile in row:
            if isinstance(tile,dict):
                if tile.get("crop") in crops:crops[tile["crop"]]+=1
                animals+=int(bool(tile.get("animal")))
    prices=obs["market"]["prices"];inventory=obs["market"].get("inventory",{})
    x=[obs["step"]/719,obs["hour"]/24,farm["money"]/10000,
       obs["farms"][1-obs["player"]]["money"]/10000,
       len(farm["hands"])/24,len(farm["unlocked_quadrants"])/4,
       active/4,changes/20,animals/30,opportunity.pos[0]/10,opportunity.pos[1]/10,opportunity.actor/25]
    x += [prices.get(c,0)/200 for c in PRODUCTS]
    x += [inventory.get(c,0)/1000 for c in PRODUCTS]
    x += [private["shed"].get(c,0)/100 for c in PRODUCTS]
    x += [private["seeds"].get(c,0)/20 for c in CROPS]
    x += [sum(b.get(c,0) for b in private["inventories"])/100 for c in PRODUCTS]
    x += [crops[c]/100 for c in CROPS]
    x += [float(s in obs["town"]["unlocked_shops"]) for s in SHOPS]
    x += [float(opportunity.original_crop==c) for c in CROPS]
    x += [float(opportunity.old_crop==c) for c in CROPS]
    features=[]
    for c in candidates:
        p=c.plan;start=p.get("start",opportunity.step)
        v=[float(c.kind==kind) for kind in ("KEEP","CROP","DEFER")]
        v += [float(c.crop==crop) for crop in CROPS]
        v += [c.cost/100,(start-opportunity.step)/24,
              (p.get("finish",obs["day"])-obs["day"])/30,
              len(p.get("harvest_days",[]))/4,len(c.slots)/40,
              sum(op=="WATER" for _,_,op in c.slots)/30,
              (p.get("release_step",opportunity.step)-opportunity.step)/719,
              prices.get(c.crop,0)/200,private["seeds"].get(c.crop,0)/20,
              inventory.get(c.crop,0)/1000]
        features.append(padded(v,CANDIDATE_SIZE))
    return padded(x,CONTEXT_SIZE),features
