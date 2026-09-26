"""Standard-library-only CPU deployment of the small event policy."""
import hashlib,json,math
from pathlib import Path
from .adapter import LocalEconomy
from .settings import Settings,SCHEMA
from .encoding import encode
from .policy import greedy

def linear(x,w,b):
    return [sum(a*v for a,v in zip(row,x))+bias for row,bias in zip(w,b)]
def lsm(values):
    m=max(values);z=m+math.log(sum(math.exp(v-m) for v in values))
    return [v-z for v in values]

class NativePolicy:
    def __init__(self,weights):
        self.w=weights
    def layer(self,x,prefix,activation=True):
        out=linear(x,self.w[prefix+".weight"],self.w[prefix+".bias"])
        return [math.tanh(v) for v in out] if activation else out
    def probabilities(self,context,candidates):
        z=self.layer(self.layer(context,"context.0"),"context.2")
        g=self.layer(z,"gate",False)[0]
        if len(candidates)==1:return [0.]
        keep=-max(g,0)-math.log1p(math.exp(-abs(g)))
        change=-max(-g,0)-math.log1p(math.exp(-abs(g)))
        scores=[]
        for c in candidates[1:]:
            v=self.layer(c,"candidate.0")
            scores.append(self.layer(self.layer(v+z,"score.0"),"score.2",False)[0])
        return [keep]+[change+lp for lp in lsm(scores)]
    def choose(self,obs,opportunity,candidates,ledger):
        context,features=encode(obs,opportunity,candidates,ledger["active"],ledger["changes"])
        return greedy(self.probabilities(context,features))

class SubmissionAgent:
    def __init__(self,directory):
        self.root=Path(directory)
        self.metadata=json.loads((self.root/"metadata.json").read_text())
        if self.metadata["schema"]!=SCHEMA:raise ValueError("incompatible local economic policy")
        raw=(self.root/"weights.json").read_bytes()
        baseline=(self.root/"baseline.py").read_bytes()
        if hashlib.sha256(raw).hexdigest()!=self.metadata["weights_sha256"]:raise ValueError("weights checksum mismatch")
        if hashlib.sha256(baseline).hexdigest()!=self.metadata["baseline_sha256"]:raise ValueError("baseline checksum mismatch")
        self.policy=NativePolicy(json.loads(raw));self.controllers={}
    def act(self,obs,configuration=None):
        seat=obs["player"]
        if seat not in self.controllers:
            self.controllers[seat]=LocalEconomy(self.policy.choose,Settings(**self.metadata["settings"]),
                                               self.root/"baseline.py")
        return self.controllers[seat].act(obs,configuration)
