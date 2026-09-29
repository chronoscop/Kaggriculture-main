"""Offline native audit summaries. Recorded branch scores are NOT game win rates."""
import collections, hashlib, json, random, statistics, struct, sys
from pathlib import Path
root=Path(sys.argv[1] if len(sys.argv)>1 else 'runs/event_learning_audit')
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()]
def score(v):return float(v[0]>v[1])+.5*(v[0]==v[1])
def selected(r):return score(r['alternative_cash'] if r['pair_choose_alternative'] else r['reference_cash'])
def metrics(rs):
 if not rs:return {'rows':0}
 ds=[r for r in rs if r['target']]; pos=[r for r in ds if r['target']>0];neg=[r for r in ds if r['target']<0]
 known=[r for r in rs if r['full_argmax'] in [r['reference_index'],r['alternative_index']]]
 return dict(rows=len(rs),decisive=len(ds),accuracy=sum(r['pair_choose_alternative']==(r['target']>0) for r in ds)/len(ds) if ds else None,
  better_alternative_recall=sum(r['pair_choose_alternative'] for r in pos)/len(pos) if pos else None,
  better_reference_recall=sum(not r['pair_choose_alternative'] for r in neg)/len(neg) if neg else None,
  reference_accuracy=len(neg)/len(ds) if ds else None,
  mean_recorded_selected_score=statistics.mean(map(selected,rs)),
  mean_recorded_reference_score=statistics.mean(score(r['reference_cash']) for r in rs),
  mean_recorded_oracle_score=statistics.mean(max(score(r['reference_cash']),score(r['alternative_cash'])) for r in rs),
  full_argmax_supported=len(known),full_argmax_unknown=len(rs)-len(known),
  full_argmax_known_decisive=sum(bool(r['target']) for r in known),
  full_argmax_known_decisive_correct=sum(bool(r['target']) and (r['full_argmax']==r['alternative_index'])==(r['target']>0) for r in known))
def stages(rs):
 d={s:metrics([r for r in rs if s=='all' or r['stage']==s]) for s in ['all','arrangement','revision']}
 d['candidate_prefix_revision']=metrics([r for r in rs if r['source']=='candidate_prefix' and r['stage']=='revision'])
 d['keep_reference_only']=metrics([r for r in rs if r['reference_plan']['keep']])
 return d
def bootstrap(before,after,baseline):
 groups=collections.defaultdict(list)
 for b,a in zip(before,after):
  assert (b['seed'],b['seat'],b['step'],b['reference_index'],b['alternative_index'])==(a['seed'],a['seat'],a['step'],a['reference_index'],a['alternative_index'])
  if baseline=='keep' and not a['reference_plan']['keep']:continue
  groups[a['seed']].append(selected(a)-(selected(b) if baseline=='before' else score(a['reference_cash'])))
 vals=[statistics.mean(v) for v in groups.values()]
 if not vals:return None
 rng=random.Random(20260929);boots=sorted(statistics.mean(rng.choices(vals,k=len(vals))) for _ in range(10000))
 return {'seeds':len(vals),'mean_per_seed_recorded_gain':statistics.mean(vals),'descriptive_bootstrap_95_interval':[boots[249],boots[9749]],'note':'retrospective comparisons, not prospective full-game win rate'}
result={}
for task in ['memorization','generalization']:
 p=root/task
 if not (p/'heldout_after.jsonl').exists():continue
 result[task]={}
 for split in ['train','heldout']:
  b=rows(p/f'{split}_before.jsonl');a=rows(p/f'{split}_after.jsonl')
  result[task][split]={'before':stages(b),'after':stages(a),'after_vs_before':bootstrap(b,a,'before'),'after_vs_keep':bootstrap(b,a,'keep')}
  if task=='generalization' and split=='heldout':
   wrong=[dict(x,before_choose_alternative=y['pair_choose_alternative']) for y,x in zip(b,a) if x['target'] and x['pair_choose_alternative']!=(x['target']>0)]
   (root/'heldout_errors.json').write_text(json.dumps(wrong,indent=2)+'\n')
 result[task]['training']=rows(p/'training.jsonl')
protocol=json.load((root/'protocol.json').open());result['source_unchanged']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protocol['source_sha256'].items()}
def key(r):
 f=r['full_row'];v=f['context']+f['features'][r['reference_index']]+f['features'][r['alternative_index']]
 return hashlib.sha256(struct.pack('<'+'f'*len(v),*v)).hexdigest()
train=rows(root/'train.jsonl');test=rows(root/'heldout.jsonl')
for name,rs in [('train',train),('heldout',test),('memorize',rows(root/'memorize.jsonl'))]:
 groups=collections.defaultdict(list)
 for r in rs:groups[key(r)].append(score(r['alternative_cash'])-score(r['reference_cash']))
 result.setdefault('input_identity',{})[name]={'unique_pairs':len(groups),'rows':len(rs),'groups_with_opposite_labels':sum(any(x>0 for x in v) and any(x<0 for x in v) for v in groups.values())}
result['input_identity']['shared_train_heldout_pairs']=len({key(r) for r in train}&{key(r) for r in test})

# Merge all measured alternatives from each actual state before declaring an argmax unknown.
def state_key(r):
 f=r['full_row'];v=f['context']+[x for a in f['features'] for x in a]
 return (r['seed'],r['seat'],r['opponent'],r['step'],hashlib.sha256(struct.pack('<'+'f'*len(v),*v)).hexdigest())
result['state_support']={}
for task,filename in [('memorization','memorize'),('generalization','heldout')]:
 if task not in result:continue
 rs=rows(root/f'{filename}.jsonl');groups=collections.defaultdict(list)
 for i,r in enumerate(rs):groups[state_key(r)].append(i)
 out={}
 for when in ['before','after']:
  ps=rows(root/task/f'heldout_{when}.jsonl');unknown=0
  for ids in groups.values():
   choices={ps[i]['full_argmax'] for i in ids};assert len(choices)==1
   tested={rs[i][k] for i in ids for k in ['reference_index','alternative_index']}
   unknown+=next(iter(choices)) not in tested
  out[when]={'states':len(groups),'unknown_argmax':unknown}
 result['state_support'][task]=out

(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for task in ['memorization','generalization']:
 if task not in result:continue
 for split in ['train','heldout']:
  d=result[task][split];print(task,split)
  for stage in ['all','arrangement','revision','candidate_prefix_revision','keep_reference_only']:
   a=d['before'][stage];b=d['after'][stage];print(stage,'n',b['rows'],'decisive',b.get('decisive'),'accuracy',a.get('accuracy'),'->',b.get('accuracy'),'score',a.get('mean_recorded_selected_score'),'->',b.get('mean_recorded_selected_score'),'reference',b.get('mean_recorded_reference_score'),'unknown',a.get('full_argmax_unknown'),'->',b.get('full_argmax_unknown'))
  print('paired',d['after_vs_before'],'keep',d['after_vs_keep'])
print('identity',result['input_identity']);print('source unchanged',all(result['source_unchanged'].values()))
