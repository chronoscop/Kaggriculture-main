"""Binary keep/change gate followed by a conditional alternative policy."""
import math
from .encoding import CONTEXT_SIZE,CANDIDATE_SIZE

class Policy:
    def __init__(self,torch,initial_change_probability=.03):
        self.torch=torch;nn=torch.nn
        self.context=nn.Sequential(nn.Linear(CONTEXT_SIZE,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.candidate=nn.Sequential(nn.Linear(CANDIDATE_SIZE,64),nn.Tanh())
        self.score=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        self.gate=nn.Linear(64,1)
        self.value=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        self.module=nn.ModuleDict(dict(context=self.context,candidate=self.candidate,score=self.score,gate=self.gate,value=self.value))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias,math.log(initial_change_probability/(1-initial_change_probability)))
    def __call__(self,context,candidates,mask):
        torch=self.torch;F=torch.nn.functional
        z=self.context(context);v=self.candidate(candidates)
        scores=self.score(torch.cat((v,z[:,None,:].expand(-1,v.shape[1],-1)),-1)).squeeze(-1)
        gate=self.gate(z).squeeze(-1)
        alternatives=mask.clone();alternatives[:,0]=False
        has_alternative=alternatives.any(-1)
        conditional=scores.masked_fill(~alternatives,-1e9).log_softmax(-1)
        lp=conditional+F.logsigmoid(gate)[:,None]
        keep=torch.where(has_alternative,F.logsigmoid(-gate),torch.zeros_like(gate))
        lp=torch.cat((keep[:,None],lp[:,1:]),-1).masked_fill(~mask,-1e9)
        return lp,self.value(z).squeeze(-1)

def greedy(log_probs):
    # Greedy at the binary gate, not global leaf argmax (candidate-count bias).
    import math
    if len(log_probs)<2 or math.exp(log_probs[0])>=.5:return 0
    return max(range(1,len(log_probs)),key=log_probs.__getitem__)

def chooser(model,torch,device,rows=None,deterministic=False):
    from .encoding import encode
    def choose(obs,opportunity,candidates,ledger):
        context,features=encode(obs,opportunity,candidates,ledger["active"],ledger["changes"])
        c=torch.tensor([context],dtype=torch.float32,device=device)
        x=torch.tensor([features],dtype=torch.float32,device=device)
        mask=torch.ones((1,len(features)),dtype=torch.bool,device=device)
        with torch.no_grad():
            lp,value=model(c,x,mask)
            dist=torch.distributions.Categorical(logits=lp)
            idx=greedy(lp[0].tolist()) if deterministic else int(dist.sample().item())
        if rows is not None:
            rows.append(dict(context=context,features=features,action=idx,logp=float(lp[0,idx]),
                             value=float(value.item()),step=obs["step"],key=opportunity.key))
        return idx
    return choose
