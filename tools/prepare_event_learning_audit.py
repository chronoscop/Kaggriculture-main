import json, hashlib, collections
from pathlib import Path
src=Path('runs/event_policy_candidate_prefix_trial'); out=Path('runs/event_learning_audit');out.mkdir(exist_ok=False)
rows=[json.loads(l) for l in (src/'comparisons.jsonl').read_text().splitlines()]
score=lambda v: (v[0]>v[1])+.5*(v[0]==v[1])
seeds=sorted({r['seed'] for r in rows},key=lambda s:hashlib.sha256(f'event-audit-20260929:{s}'.encode()).hexdigest())
train_seeds=set(seeds[:60]);test_seeds=set(seeds[60:]); assert len(train_seeds)==60 and len(test_seeds)==20
train=[r for r in rows if r['seed'] in train_seeds]; test=[r for r in rows if r['seed'] in test_seeds]
small=[]; used=set()
for stage in ['arrangement','revision']:
 for sign in [-1,1]:
  eligible=[r for r in train if r['stage']==stage and (score(r['alternative_cash'])-score(r['reference_cash']))*sign>0]
  eligible.sort(key=lambda r:hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest())
  n=0
  for r in eligible:
   # Avoid exact repeated observations/pairs, not merely adjacent JSON entries.
   f=r['full_row'];features=f['features'];key=json.dumps([f['context'],features[r['reference_index']],features[r['alternative_index']]])
   if key in used: continue
   small.append(r);used.add(key);n+=1
   if n==6:break
  assert n==6,(stage,sign,n)
for name,a in [('train',train),('heldout',test),('memorize',small)]:
 (out/f'{name}.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in a))
initial=json.load((src/'initial.json').open());collection=json.load((src/'collection_sources/collection_000001.json').open());assert initial['model']['weights']==collection['candidate_prefix_policy']['weights'];assert initial['iteration']=='0'
(out/'initial_weights.json').write_text(json.dumps(initial['model']['weights']))
files=[src/f for f in ['initial.json','latest.json','best.json','comparisons.jsonl','metrics.jsonl','evaluations.jsonl','sequences.jsonl','train.log']]
protocol={'source':str(src),'objective':'terminal_match_score_difference_v1','split':'sha256(event-audit-20260929:seed), first60 train last20 heldout; all rows grouped by seed','train_seeds':sorted(train_seeds),'heldout_seeds':sorted(test_seeds),'train_rows':len(train),'heldout_rows':len(test),'memorize_rows':len(small),'memorize_epochs':1000,'generalization_epochs':200,'learning_rate':0.0003,'batch_size':64,'device':'cuda','initial':'initial.json model.weights; verified equal collection_000001 candidate_prefix_policy.weights','optimizer':'fresh Adam in both experiments; same native update_improvement; no replay duplication','evaluation':'heldout evaluated only before and after predetermined training, no checkpoint selection','limitations':['Recorded choices were collected adaptively in the original run. Heldout labels never enter this audit fitting, but this is retrospective grouped generalization, not a prospective independent policy evaluation.','Two measured options can be scored; full-set argmax outside measured options remains unknown.','One fixed 60/20 split and one initialization; no significance fishing.'], 'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print({k:protocol[k] for k in ['train_rows','heldout_rows','memorize_rows']})
for name,a in [('train',train),('heldout',test),('memorize',small)]:print(name,collections.Counter((r['stage'],score(r['alternative_cash'])-score(r['reference_cash'])) for r in a))
