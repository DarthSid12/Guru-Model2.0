import csv,json,math,hashlib,statistics as st,sys
from pathlib import Path
from collections import defaultdict
B=Path('/Users/darthsid/Documents/ChatGPT/GURU');O=B/'operations/local-yin-new-sigma-two-noise-20261116';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((O/'SPEC.json').read_text());run=json.loads((O/'RUN.json').read_text());valid=json.loads((O/'results/VALIDATED.json').read_text());completion=json.loads((O/'COMPLETION.json').read_text())
assert run['registration_complete'] and run['spec_sha256']==sha(O/'SPEC.json')==valid['spec_sha256'] and valid['passed']
assert completion['exit_code']==0 and not completion['timed_out'] and completion['log_sha256']==sha(O/'logs/run.log')
for n,h in s['files'].items():assert sha(O/n)==h,n
for n,h in s['external_files'].items():assert sha(n)==h,n
assert valid['csv_sha256']==sha(O/'results/all_records.csv') and valid['trials_sha256']==sha(O/'results/all_trials.csv') and valid['summary_sha256']==sha(O/'results/SUMMARY.json')
assert s['calibration_seeds']==list(range(101,151)) and s['validation_seeds']==list(range(201,251)) and set(s['validation_seeds']).isdisjoint(s['sigma_selection_seeds'])
smoke=json.loads((O/'results/SMOKE_PASSED.json').read_text());assert smoke['passed'] and len(smoke['pilot'])==16
models={m['record']['variant']:m for m in s['models']};fits={}
for v,m in models.items():
 f=json.loads((O/'calibration'/(v+'.json')).read_text());fits[v]=f
 assert f['spec_sha256']==sha(O/'SPEC.json') and f['sigma']==m['new_sigma'] and f['calibration_design']=='faces_houses'
 for condition,key in [('UU','UU_fit'),('II','II_fit')]:
  q=f[key];assert q['target']==s['targets'][condition] and q['condition']==condition
  assert q['test_noise_held_at']==(None if condition=='UU' else f['p1'])
  rates={x['rate']:x for x in q['curve']}
  for rate,x in rates.items():
   assert sorted(y['seed'] for y in x['seed_counts'])==s['calibration_seeds'] and all(y['n_trials']==12 and 0<=y['correct']<=12 for y in x['seed_counts'])
   assert math.isclose(sum(y['correct'] for y in x['seed_counts'])/600,x['accuracy'],abs_tol=1e-12)
  def best(allowed):return min(allowed,key=lambda p:(abs(rates[p]['accuracy']-q['target']),p))
  coarse=best(s['coarse_noise_grid']);assert coarse==q['coarse_best']
  fine=[i/1000 for i in range(max(0,round(coarse*1000)-50),min(500,round(coarse*1000)+50)+1,5)];assert fine==q['fine_grid']
  fb=best(set(s['coarse_noise_grid'])|set(fine));assert fb==q['fine_best']
  ultra=[i/1000 for i in range(max(0,round(fb*1000)-5),min(500,round(fb*1000)+5)+1)];assert ultra==q['ultrafine_grid']
  assert set(rates)==set(s['coarse_noise_grid'])|set(fine)|set(ultra)
  chosen=best(rates);assert chosen==q['selected_noise']==f['p1' if condition=='UU' else 'p2'] and q['achieved']==rates[chosen]['accuracy']
  assert q['target_within_sampled_range']==(min(x['accuracy'] for x in rates.values())<=q['target']<=max(x['accuracy'] for x in rates.values()))
rows=list(csv.DictReader((O/'results/all_records.csv').open()));trials=list(csv.DictReader((O/'results/all_trials.csv').open()));assert len(rows)==6400 and len(trials)==76800
bytrial=defaultdict(list);seen=set()
for t in trials:
 key=(t['variant'],t['design'],t['method'],int(t['simulation_seed']),t['condition'],t['category']);uid=(*key,int(t['old_row']));assert uid not in seen;seen.add(uid)
 assert int(t['correct'])==int(float(t['old_log_familiarity'])>float(t['novel_log_familiarity']))
 assert int(t['old_identity_correct'])==int(t['old_winner_category']==t['category'] and t['old_winner_row']==t['old_row'])
 bytrial[key].append(t)
groups=defaultdict(list);paired=defaultdict(dict);records={}
for r in rows:
 v=r['variant'];m=models[v];f=fits[v];seed=int(r['simulation_seed']);method=r['method'];cond=r['condition'];cat=r['category'];design=r['design']
 assert seed in s['validation_seeds'] and cond in s['conditions'] and int(r['training_seed'])==46 and int(r['epoch'])==124 and r['checkpoint_sha256']==m['record']['sha256']
 if method=='sigma2_two_noise_originalfit':a,b=m['original_two_noise']['p1'],m['original_two_noise']['p2'];sig=2.
 else:
  a,b=f['p1'],f['p2'] if method!='new_sigma_single_p1' else f['p1'];sig=2. if method=='sigma2_two_noise_matchedfit' else m['new_sigma']
 assert float(r['p1'])==a and float(r['p2'])==b and float(r['sigma'])==sig
 assert float(r['study_noise'])==(a if cond[0]=='U' else b) and float(r['test_noise'])==a
 ts=bytrial[v,design,method,seed,cond,cat];assert len(ts)==12 and int(r['n_trials'])==12
 mask=''.join(t['correct'] for t in ts);assert mask==r['correct_mask'] and mask.count('1')==int(r['correct_pairs']) and int(r['errors'])==12-mask.count('1')
 assert int(r['old_correct_memory_count'])==sum(int(t['old_identity_correct']) for t in ts)
 assert int(r['old_cross_category_memory_count'])==sum(t['old_winner_category']!=cat for t in ts)
 key=(v,design,method,cat,cond);groups[key].append(r);records[v,design,method,seed,cond,cat]=r;paired[v,design,cat,cond,seed][method]=int(r['errors'])
# Where study is upright,the two new-sigma arms must be literally identical.
for key,r in records.items():
 v,design,method,seed,cond,cat=key
 if method=='new_sigma_two_noise' and cond in ['UU','UI']:assert r['correct_mask']==records[v,design,'new_sigma_single_p1',seed,cond,cat]['correct_mask']
summary=json.loads((O/'results/SUMMARY.json').read_text());assert len(groups)==128 and len(summary['cells'])==128 and len(summary['paired_comparisons'])==96
for c in summary['cells']:
 rs=groups[tuple(c[k] for k in ['variant','design','method','category','condition'])];assert sorted(int(r['simulation_seed']) for r in rs)==s['validation_seeds']
 for name,vals in [('errors',[int(r['errors']) for r in rs]),('old_identity_accuracy_pct',[100*int(r['old_correct_memory_count'])/12 for r in rs]),('old_cross_category_match_pct',[100*int(r['old_cross_category_memory_count'])/12 for r in rs])]:
  for field,value in [('mean',st.mean(vals)),('sd',st.stdev(vals)),('sem',st.stdev(vals)/math.sqrt(50))]:assert math.isclose(c[name][field],value,abs_tol=1e-12)
for c in summary['paired_comparisons']:
 vals=[paired[c['variant'],c['design'],c['category'],c['condition'],seed][c['control']]-paired[c['variant'],c['design'],c['category'],c['condition'],seed]['new_sigma_two_noise'] for seed in s['validation_seeds']]
 for field,value in [('mean',st.mean(vals)),('sd',st.stdev(vals)),('sem',st.stdev(vals)/math.sqrt(50))]:assert math.isclose(c['paired_improvement_errors'][field],value,abs_tol=1e-12)
# Replay primary decisions through the immutable original engine rather than
# through the new run's score helper. Also replay both selected face fits with
# the full probe batch,checking the fitting subset did not alter outcomes.
sys.path.insert(0,str(O));from engine import Experiment
from kde import inputs
import torch
torch.set_num_threads(1);trial_replays=0;cal_replays=0
for v,m in models.items():
 f=fits[v];e=Experiment(m['cache'],'faces_houses')
 for cond,key in [('UU','UU_fit'),('II','II_fit')]:
  expected={x['seed']:x['correct'] for x in next(x for x in f[key]['curve'] if x['rate']==f[key]['selected_noise'])['seed_counts']}
  for seed in s['calibration_seeds']:
   e.draws.clear();sn=f['p1'] if cond[0]=='U' else f['p2'];p,mem,q=inputs(e,seed,cond,sn,f['p1']);values=e.familiarity(q,mem,m['new_sigma']);ok=values.reshape(24,2)[:,0]>values.reshape(24,2)[:,1]
   assert sum(int(ok[i]) for i,t in enumerate(p['trials']) if t['category']=='faces')==expected[seed];cal_replays+=1
 for design in s['designs']:
  e=Experiment(m['cache'],design)
  for seed in s['validation_seeds']:
   e.draws.clear()
   for cond in s['conditions']:
    sn=f['p1'] if cond[0]=='U' else f['p2'];p,mem,q=inputs(e,seed,cond,sn,f['p1']);values=e.familiarity(q,mem,m['new_sigma'])
    for cat in s['design_definitions'][design]:
     ts=bytrial[v,design,'new_sigma_two_noise',seed,cond,cat];lookup={int(t['old_row']):t for t in ts}
     for i,t in enumerate(p['trials']):
      if t['category']!=cat:continue
      r=lookup[t['old']];assert float(r['old_log_familiarity'])==float(values[2*i]) and float(r['novel_log_familiarity'])==float(values[2*i+1]);trial_replays+=1
 done=json.loads((O/'results'/(v+'-DONE.json')).read_text());assert done['original_control_records_replayed']==200
 assert done['csv_sha256']==sha(O/'results'/(v+'-raw.csv')) and done['trials_sha256']==sha(O/'results'/(v+'-trials.csv')) and done['calibration_sha256']==sha(O/'calibration'/(v+'.json'))
assert trial_replays==19200 and cal_replays==200
report=dict(passed=True,category_seed_rows=6400,trial_rows=76800,cells=128,simulation_seeds_per_cell=50,validation_seeds='201-250,disjointfromsigma-selection',original_parent_control_records_replayed=400,primary_trial_scores_replayed_exactly_through_original_engine=trial_replays,selected_calibration_seed_results_replayed_full_batch=cal_replays,all_calibration_grids_and_face_only_fit_selection_verified=True,all_noise_rules_and_object_transfer_verified=True,all_counts_means_sample_SD_SEM_and_paired_effects_verified=True,source_and_checkpoint_cache_provenance_hashes_verified=True,spec_sha256=sha(O/'SPEC.json'),results_sha256=valid['csv_sha256'])
(O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
