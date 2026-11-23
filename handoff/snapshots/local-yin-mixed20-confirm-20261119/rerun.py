import os,sys,json,csv,hashlib,statistics,datetime
from pathlib import Path
from collections import Counter,defaultdict
B=Path('/Users/darthsid/Documents/ChatGPT/GURU')
O=Path(__file__).resolve().parent
P=B/'operations/local-yin-two-sigma-tables-20261116'
sys.path.insert(0,str(P))
import torch
from run import verify
from engine import Experiment
from kde import inputs
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
torch.set_num_threads(1)
s=verify()
assert not (O/'SPEC.json').exists(), 'Already registered; do not overwrite'
fits={}
for m in s['models']:
 v=m['record']['variant'];fits[v]={}
 for ex in s['experiments']:
  f=json.loads((P/'calibration'/f'{v}-{ex}.json').read_text())
  if ex=='adjusted_sigma':f=f['fit']
  fits[v][ex]={k:f[k] for k in ['sigma','p1','p2']}
spec=dict(parent=str(P),parent_spec_sha256=sha(P/'SPEC.json'),runner_sha256=sha(__file__),seeds=s['report_seeds'],designs=s['design_definitions'],fits=fits,models=[m['record'] for m in s['models']],noise='UU/UI: study p1,test p1; IU/II: study p2,test p1; transfer unchanged from faces_houses to faces_objects',calibration='Reuse frozen face UU/II fits; no refitting',reporting='Same 50 fitting/report seeds, not independent validation',created_utc=stamp())
save(O/'SPEC.json',spec);save(O/'LATEST_STATUS.json',dict(status='RUNNING',started_utc=stamp()))
prior=list(csv.DictReader((P/'results/all_records.csv').open()))
lookup={(r['experiment'],r['variant'],r['design'],int(r['simulation_seed']),r['condition'],r['category']):r for r in prior}
rows=[];plans={};trial_rows=[]
for m in s['models']:
 v=m['record']['variant']
 for ex in s['experiments']:
  f=fits[v][ex]
  for d in ['faces_houses','faces_objects']:
   e=Experiment(m['cache'],d)
   for seed in s['report_seeds']:
    e.draws.clear()
    for c in s['conditions']:
     sn=f['p1'] if c[0]=='U' else f['p2']
     p,mem,q=inputs(e,seed,c,sn,f['p1'])
     assert Counter(cat for cat,r in p['study_order'])==Counter({cat:20 for cat in s['design_definitions'][d]})
     assert Counter(t['category'] for t in p['trials'])==Counter({cat:12 for cat in s['design_definitions'][d]})
     for cat,g in p['groups'].items():
      assert set(g['tested_old'])<=set(g['studied']) and not set(g['novel'])&set(g['studied'])
     plans[f'{d}:{seed}']=p
     scores=e.familiarity(q,mem,f['sigma']).reshape(24,2)
     ok=(scores[:,0]>scores[:,1]).tolist()
     for j,t in enumerate(p['trials']):
      trial_rows.append(dict(variant=v,experiment=ex,design=d,simulation_seed=seed,condition=c,category=t['category'],old=t['old'],new=t['new'],old_score=float(scores[j,0]),new_score=float(scores[j,1]),correct=int(ok[j])))
     for cat in s['design_definitions'][d]:
      mask=''.join('1' if ok[j] else '0' for j,t in enumerate(p['trials']) if t['category']==cat)
      old=lookup[ex,v,d,seed,c,cat]
      assert mask.count('1')==int(old['correct_pairs'])
      if old['correct_mask']:assert mask==old['correct_mask']
      assert all(float(old[k])==f[k] for k in f)
      rows.append(dict(experiment=ex,variant=v,design=d,simulation_seed=seed,condition=c,category=cat,**f,study_noise=sn,test_noise=f['p1'],n_trials=12,correct_pairs=mask.count('1'),errors=mask.count('0'),correct_mask=mask))
   print('Completed',v,ex,d,flush=True)
assert len(rows)==3200 and len(trial_rows)==38400 and len(plans)==100
for name,data in [('all_records.csv',rows),('all_trials.csv',trial_rows)]:
 with (O/name).open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
save(O/'STIMULUS_PLANS.json',plans)
groups=defaultdict(list)
for r in rows:groups[r['experiment'],r['variant'],r['design'],r['category'],r['condition']].append(r['errors'])
assert len(groups)==64 and all(len(x)==50 for x in groups.values())
lines=['# Confirmed mixed-category Yin rerun','','Errors out of 12, mean ± sample SD across 50 simulation seeds (101–150). Same seeds used for face calibration; this is not independent validation. Both experiments use the same epoch-124, training-seed-46 LP/CNN checkpoints.','','Study 20 faces + 20 houses first; transfer the fitted noise unchanged to a separate study of 20 faces + 20 objects. Each test has 12 old/new pairs per category, within-category comparisons only. Ten study fixations and 32 test fixations per image.','','Every category count reproduced the previous results exactly.']
for ex in s['experiments']:
 lines+=['',f'## {ex.replace("_"," ")}', '', '| Model | Study group | Outcome | σ | p1 / p2 | UU | UI | IU | II |','|---|---|---|---:|---|---:|---:|---:|---:|']
 for d in s['designs']:
  for cat in s['design_definitions'][d]:
   for v in ['lp','cnn']:
    f=fits[v][ex];cells=[]
    for c in s['conditions']:
     a=groups[ex,v,d,cat,c];cells.append(f'{statistics.mean(a):.2f} ± {statistics.stdev(a):.2f}')
    label='20 faces + 20 houses' if d=='faces_houses' else '20 faces + 20 objects'
    lines.append(f'| {v.upper()} | {label} | {cat} | {f["sigma"]:.4f} | {f["p1"]:.3f} / {f["p2"]:.3f} | '+' | '.join(cells)+' |')
lines+=['','UU/UI use p1 at study and test. IU/II use p2 at study and p1 at test. Noise and sigma remain identical across the two study groups within each model/experiment.','','The object bank includes 16 categories outside the revised training diet. Adjusted sigma was previously selected using face and house outcomes.']
(O/'RESULTS.md').write_text('\n'.join(lines)+'\n')
receipt=dict(passed=True,completed_utc=stamp(),spec_sha256=sha(O/'SPEC.json'),records=3200,trials=38400,plans=100,seeds_per_cell=50,twenty_study_items_per_category_verified=True,twelve_within_category_test_pairs_verified=True,noise_transfer_unchanged=True,all_counts_match_previous=True,records_sha256=sha(O/'all_records.csv'),trials_sha256=sha(O/'all_trials.csv'))
save(O/'VERIFICATION.json',receipt);save(O/'LATEST_STATUS.json',dict(status='COMPLETED_VERIFIED',**receipt));print(json.dumps(receipt,indent=2))
