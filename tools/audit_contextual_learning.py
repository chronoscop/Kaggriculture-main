"""Bounded offline audit: archived inputs only; no simulator or deployment writes.
PyTorch mirror is checked against event-menu-audit native predictions before use.
"""
import json, pathlib, hashlib, copy, math, time
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
root=pathlib.Path('runs/contextual_learning_final_audit')
source=pathlib.Path('runs/event_policy_contextual_overnight')
rows=[json.loads(x) for x in (root/'sets.jsonl').read_text().splitlines()]
ck=json.load(open(source/'checkpoint_000010.json'))['model']
initial=json.load(open(source/'initial.json'))['model']
N=len(rows); W=max(len(r['full_row']['features']) for r in rows)
x=torch.tensor([r['full_row']['context'] for r in rows],dtype=torch.float32)
f=torch.zeros(N,W,32);mask=torch.zeros(N,W,dtype=torch.bool);scores=torch.zeros(N,W)
for i,r in enumerate(rows):
 n=len(r['full_row']['features']);f[i,:n]=torch.tensor(r['full_row']['features']);mask[i,:n]=True
 scores[i,:n]=torch.tensor([float(v[0]>v[1])+.5*float(v[0]==v[1]) for v in r['all_terminal_cash']])
target=scores-scores[:,:1];valid=mask.clone();valid[:,0]=False
positive=target>0;opportunity=positive.any(1)
class Net:
 def __init__(self,j,adam=False):
  self.p={k:torch.nn.Parameter(torch.tensor(v['data'],dtype=torch.float32).reshape(v['shape'])) for k,v in j['weights'].items() if isinstance(v,dict)}
  self.opt=torch.optim.Adam(list(self.p.values()),lr=j['learning_rate'])
  if adam:
   for k,p in self.p.items():
    a=j['optimizer'][k]
    if int(a['step']):self.opt.state[p]={'step':torch.tensor(float(a['step'])),'exp_avg':torch.tensor(a['m']).reshape(p.shape),'exp_avg_sq':torch.tensor(a['v']).reshape(p.shape)}
 def logits(self,ids,shuffle=False):
  xx=x[ids].clone();ff=f[ids];xx[:,17:43:3]=xx[:,17:43:3]*.01-1
  xx=xx/torch.sqrt(1+xx.square());cc=ff/torch.sqrt(1+ff.square())
  if shuffle:xx=xx.roll(1,0)
  def lin(t,k):return F.linear(t,self.p[k+'.weight'],self.p[k+'.bias'])
  z=lin(lin(xx,'context.0').tanh(),'context.2').tanh()
  v=lin(cc,'candidate.0').tanh()
  out=lin(lin(torch.cat([v,z[:,None,:].expand(-1,W,-1)],-1),'score.0').tanh(),'score.2').squeeze(-1)
  return out+ff[:,:,30]
 def loss(self,ids):
  z=self.logits(ids);d=z-z[:,:1]
  return (((d-target[ids]).square()*valid[ids]).sum(1)/valid[ids].sum(1)).mean()
 def step(self,ids):
  self.opt.zero_grad();l=self.loss(ids);l.backward();torch.nn.utils.clip_grad_norm_(list(self.p.values()),.5);self.opt.step();return float(l.detach())
 def report(self,ids):
  with torch.no_grad():
   z=self.logits(ids).masked_fill(~mask[ids],-1e9);a=z.argmax(1);s=scores[ids];best=s.masked_fill(~mask[ids],-1).max(1).values
   actual=s.gather(1,a[:,None]).squeeze(1);opp=opportunity[ids];bestidx=s.masked_fill(~mask[ids],-1).argmax(1)
   gaps=(z.gather(1,bestidx[:,None])-z[:,:1]).squeeze(1)
   altered=self.logits(ids,True).masked_fill(~mask[ids],-1e9).argmax(1)
   return dict(sets=len(ids),opportunities=int(opp.sum()),anchor=int((a==0).sum()),positive_choices=int((actual>s[:,0]).sum()),positive_on_opportunities=int(((actual>s[:,0])&opp).sum()),worse_choices=int((actual<s[:,0]).sum()),mean_score_gain=float((actual-s[:,0]).mean()),best_choice_rate=float((actual==best).float().mean()),loss=float(self.loss(ids)),positive_option_gap_mean=float(gaps[opp].mean()),shuffled_context_changes=int((a!=altered).sum()))
ids=torch.arange(N);m=Net(ck,True)
# Verify model math with the actual Rust/libtorch inference, including its prior.
native=[json.loads(l) for l in (root/'native/train_before.jsonl').read_text().splitlines()]
with torch.no_grad():probs=m.logits(ids).masked_fill(~mask,-1e9).softmax(-1)
max_error=max(abs(float(probs[i,j])-v) for i,r in enumerate(native) for j,v in enumerate(r['probabilities']))
assert max_error<2e-6,max_error
report={'native_probability_max_error':max_error,'sets':N,'positive_sets':int(opportunity.sum()),'positive_arms':int(positive.sum()),'negative_arms':int(((target<0)&valid).sum()),'tie_arms':int(((target==0)&valid).sum()),'before':m.report(ids)}
# Every bank row is a unique complete set; positive cases in both pools.
for pool in ['history','recent']:
 bank=json.load(open(source/'checkpoint_000010.json'))['comparison_bank'][pool]
 bankids={p['evidence']['candidate_set_id'] for p in bank}
 report[pool+'_positive_sets']=sum(r['candidate_set_id'] in bankids and bool(opportunity[i]) for i,r in enumerate(rows))
m.opt.zero_grad();m.loss(ids).backward()
report['gradients']={k:float(p.grad.norm()) for k,p in m.p.items() if p.grad is not None}
# Analytic logit derivative: positive arms should move UP (negative derivative).
with torch.no_grad():
 z=m.logits(ids);d=z-z[:,:1];g=2*(d-target)/valid.sum(1)[:,None]/N
 report['positive_target_gradient_up']=int((g[positive]<0).sum())
 report['positive_predicted_gains']=[float(d[positive].min()),float(d[positive].max()),float(d[positive].mean())]
report['one_native_equivalent_adam_loss']=m.step(ids);report['after_one_step']=m.report(ids)
# Isolate one real winning choice. This is a gradient/unit-fit check, NOT validation.
single=Net(ck,True);one=opportunity.nonzero()[0].reshape(1)
report['single_case_id']=rows[int(one)]['candidate_set_id'];report['single_before']=single.report(one)
for _ in range(80):single.step(one)
report['single_after_80_steps']=single.report(one)
# Fixed-budget fit of the whole bank, preserving original Adam; no new labels.
fit=Net(ck,True);gen=torch.Generator().manual_seed(927);curve=[];started=time.monotonic()
for epoch in range(1,101):
 perm=torch.randperm(N,generator=gen)
 for batch in perm.split(64):fit.step(batch)
 if epoch in [1,5,20,50,100]:curve.append({'epoch':epoch,**fit.report(ids)})
report['whole_bank_fit']=curve
# True seed-heldout diagnostic starts from the INITIAL untrained proposal, not
# the round10 model which has already seen all 80 seeds. No tuning on test.
seeds=sorted({r['seed'] for r in rows},key=lambda s:hashlib.sha256(f'contextual-final:{s}'.encode()).hexdigest())
train_seeds=set(seeds[:60]);tr=torch.tensor([i for i,r in enumerate(rows) if r['seed'] in train_seeds]);te=torch.tensor([i for i,r in enumerate(rows) if r['seed'] not in train_seeds]);fresh=Net(initial)
for _ in range(100):
 perm=tr[torch.randperm(len(tr),generator=gen)]
 for batch in perm.split(64):fresh.step(batch)
report['seed_split']={'train_seeds':60,'test_seeds':20,'train':fresh.report(tr),'heldout':fresh.report(te),'initialized_from':'initial.json zero-head proposal','updates':400}
report['offline_seconds']=time.monotonic()-started
(root/'learning_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
