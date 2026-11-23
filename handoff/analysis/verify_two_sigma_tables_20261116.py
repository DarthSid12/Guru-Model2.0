import sys,json,csv,math,hashlib,statistics as st
from pathlib import Path
from collections import defaultdict
import torch
B=Path('/Users/darthsid/Documents/ChatGPT/GURU');O=B/'operations/local-yin-two-sigma-tables-20261116';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((O/'SPEC.json').read_text());r=json.loads((O/'RUN.json').read_text());v=json.loads((O/'results/VALIDATED.json').read_text());c=json.loads((O/'COMPLETION.json').read_text())
assert r['registration_complete'] and r['spec_sha256']==sha(O/'SPEC.json')==v['spec_sha256']
assert c['exit_code']==0 and not c['timed_out'] and c['log_sha256']==sha(O/'logs/run.log')
for n,h in s['files'].items():assert sha(O/n)==h,n
for n,h in s['external_files'].items():assert sha(n)==h,n
assert v['passed'] and v['csv_sha256']==sha(O/'results/all_records.csv') and v['summary_sha256']==sha(O/'results/SUMMARY.json')
assert s['report_seeds']==s['calibration_seeds']==list(range(101,151))
models={m['record']['variant']:m for m in s['models']};fits={}
for variant,m in models.items():
 f=json.loads((O/'calibration'/(variant+'-fixed_sigma.json')).read_text());fits[variant,'fixed_sigma']=f
 af=json.loads((O/'calibration'/(variant+'-adjusted_sigma.json')).read_text());assert af['source_sha256']==sha(af['source_file']);fits[variant,'adjusted_sigma']=af['fit']
 for cond,key in [('UU','UU_fit'),('II','II_fit')]:
  q=f[key];values={x['rate']:x for x in q['curve']};target=s['targets'][cond]
  for rate,item in values.items():
   assert sorted(x['seed'] for x in item['seed_counts'])==s['calibration_seeds'] and all(x['n_trials']==12 for x in item['seed_counts'])
   assert math.isclose(sum(x['correct'] for x in item['seed_counts'])/600,item['accuracy'],abs_tol=1e-12)
  def best(rates):return min(rates,key=lambda p:(abs(values[p]['accuracy']-target),p))
  coarse=best(s['coarse_noise_grid']);assert coarse==q['coarse_best']
  fine=[i/1000 for i in range(max(0,round(coarse*1000)-50),min(500,round(coarse*1000)+50)+1,5)];assert fine==q['fine_grid']
  fb=best(set(s['coarse_noise_grid'])|set(fine));assert fb==q['fine_best']
  ultra=[i/1000 for i in range(max(0,round(fb*1000)-5),min(500,round(fb*1000)+5)+1)];assert ultra==q['ultrafine_grid'] and set(values)==set(s['coarse_noise_grid'])|set(fine)|set(ultra)
  chosen=best(values);assert q['selected_noise']==chosen==f['p1' if cond=='UU' else 'p2'] and q['achieved']==values[chosen]['accuracy']
  assert q['test_noise_held_at']==(None if cond=='UU' else f['p1'])
rows=list(csv.DictReader((O/'results/all_records.csv').open()));assert len(rows)==3200
old={(r['variant'],r['design'],int(r['simulation_seed']),r['condition'],r['category']):r for r in csv.DictReader(Path(s['adjusted_calibration_set_csv']).open())}
G=defaultdict(list);P=defaultdict(dict);lookup={}
for row in rows:
 var=row['variant'];ex=row['experiment'];m=models[var];f=fits[var,ex];seed=int(row['simulation_seed']);cond=row['condition'];cat=row['category'];d=row['design']
 assert row['checkpoint_sha256']==m['record']['sha256'] and int(row['training_seed'])==46 and int(row['epoch'])==124 and seed in s['calibration_seeds']
 sig=2. if ex=='fixed_sigma' else m['new_sigma'];assert float(row['sigma'])==sig and float(row['p1'])==f['p1'] and float(row['p2'])==f['p2']
 assert float(row['study_noise'])==(f['p1'] if cond[0]=='U' else f['p2']) and float(row['test_noise'])==f['p1']
 assert int(row['correct_pairs'])+int(row['errors'])==int(row['n_trials'])==12 and math.isclose(float(row['accuracy_pct']),100*int(row['correct_pairs'])/12,abs_tol=1e-12)
 if ex=='fixed_sigma':assert len(row['correct_mask'])==12 and row['correct_mask'].count('1')==int(row['correct_pairs'])
 else:
  src=old[var,d,seed,cond,cat];assert int(row['errors'])==int(src['errors']) and float(row['sigma'])==float(src['sigma'])
 key=(ex,var,d,cat,cond);G[key].append(row);pk=(ex,var,d,cat,seed);assert cond not in P[pk];P[pk][cond]=int(row['errors'])
 assert (ex,var,d,seed,cond,cat) not in lookup;lookup[ex,var,d,seed,cond,cat]=row
summary=json.loads((O/'results/SUMMARY.json').read_text());assert len(G)==len(summary['cells'])==64 and len(summary['contrasts'])==16
for cell in summary['cells']:
 vals=G[tuple(cell[k] for k in ['experiment','variant','design','category','condition'])];assert sorted(int(x['simulation_seed']) for x in vals)==s['report_seeds']
 for name,data in [('errors',[int(x['errors']) for x in vals]),('accuracy_pct',[100*int(x['correct_pairs'])/12 for x in vals])]:
  for field,value in [('mean',st.mean(data)),('sd',st.stdev(data)),('sem',st.stdev(data)/math.sqrt(50))]:assert math.isclose(cell[name][field],value,abs_tol=1e-12)
for ct in summary['contrasts']:
 key=tuple(ct[k] for k in ['experiment','variant','design','category']);vals=[]
 for seed in s['report_seeds']:
  d=P[key+(seed,)];vals.append(d['UU']-(d['UI']+d['IU']+d['II'])/3)
 for name,data in [('error_points',vals),('accuracy_percentage_points',[-100*x/12 for x in vals])]:
  for field,value in [('mean',st.mean(data)),('sd',st.stdev(data)),('sem',st.stdev(data)/math.sqrt(50))]:assert math.isclose(ct[name][field],value,abs_tol=1e-12)
sys.path.insert(0,str(O));from engine import Experiment
from kde import inputs
torch.set_num_threads(1);checked=0
for var,m in models.items():
 for ex in s['experiments']:
  f=fits[var,ex];sigma=2. if ex=='fixed_sigma' else m['new_sigma']
  for design in s['designs']:
   e=Experiment(m['cache'],design)
   for seed in s['report_seeds']:
    e.draws.clear()
    for cond in s['conditions']:
     sn=f['p1'] if cond[0]=='U' else f['p2'];p,mem,q=inputs(e,seed,cond,sn,f['p1']);score=e.familiarity(q,mem,sigma);ok=(score.reshape(24,2)[:,0]>score.reshape(24,2)[:,1]).tolist()
     for cat in s['design_definitions'][design]:
      row=lookup[ex,var,design,seed,cond,cat];mask=''.join('1' if ok[i] else '0' for i,t in enumerate(p['trials']) if t['category']==cat);assert mask.count('1')==int(row['correct_pairs'])
      if ex=='fixed_sigma':assert mask==row['correct_mask']
      if design=='faces_houses' and cat=='faces' and cond in ['UU','II']:
       q=f['UU_fit' if cond=='UU' else 'II_fit'];expected=next(x for x in q['curve'] if x['rate']==q['selected_noise']);x=next(x for x in expected['seed_counts'] if x['seed']==seed);assert x['correct']==mask.count('1')
      checked+=1
assert checked==3200
receipt=dict(passed=True,rows=3200,cells=64,paired_contrasts=16,original_engine_category_decisions_replayed=checked,simulation_seeds_per_cell=50,refitted_sigma2_calibration_grid_verified=True,adjusted_fits_and_results_imported_without_change=True,same_models_and_report_seeds_verified=True,study_only_noise_and_object_transfer_verified=True,all_error_accuracy_means_SD_SEM_and_paired_contrasts_verified=True,all_source_and_external_hashes_verified=True,spec_sha256=sha(O/'SPEC.json'),csv_sha256=v['csv_sha256'])
(O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
