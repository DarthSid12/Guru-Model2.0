"""Portable replay of the latest fixed-sigma Yin experiments from pinned caches."""
import argparse,csv,json,hashlib,math,statistics,sys
from collections import defaultdict
from pathlib import Path
import torch
from engine import Experiment
from kde import inputs
from calibration import fit_axis
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def moments(xs):
 sd=statistics.stdev(xs)
 return dict(mean=statistics.mean(xs),sd=sd,sem=sd/math.sqrt(len(xs)))
def pooled_scores(probes,memory,sigma=2.):
 """One KDE over all supplied study fixations, with the original numerical floor."""
 return Experiment.familiarity(probes,memory.reshape(1,-1,256),sigma)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--cache-root',required=True,type=Path,help='Parent of the two model cache directories in registry.json')
 ap.add_argument('--output',required=True,type=Path,help='New output directory; existing directories are rejected')
 ap.add_argument('--calibrate-uu',action='store_true',help='Recompute the single UU-only fit with the original coarse/fine/ultrafine grid')
 ap.add_argument('--pooled-houses',action='store_true',help='Also score houses using separate house memories and a pooled house KDE')
 a=ap.parse_args();reg=json.loads((HERE/'registry.json').read_text())
 if a.output.exists():ap.error('Output already exists; choose a new path to preserve previous runs.')
 for model,files in reg['cache_sha256'].items():
  for name,digest in files.items():
   path=a.cache_root/model/name
   if not path.is_file() or sha(path)!=digest:ap.error(f'Cache missing or hash mismatch: {path}')
 torch.set_num_threads(1);a.output.mkdir(parents=True);rows=[];plans={};noise=dict(reg['noise'])
 write_json(a.output/'RUN.json',dict(registry_sha256=sha(HERE/'registry.json'),source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},sigma=2.,seeds=reg['seeds'],calibrate_uu=a.calibrate_uu,pooled_houses=a.pooled_houses,cache_root=str(a.cache_root.resolve()),designs=['faces_houses','faces_objects'],study_per_category=20,test_pairs_per_category=12,study_fixations=10,test_fixations=32))
 for model in reg['models']:
  variant=model['variant'];cache=a.cache_root/model['model']
  if a.calibrate_uu:
   settings=dict(calibration_seeds=reg['seeds'],coarse_noise_grid=[i/20 for i in range(11)])
   fit=fit_axis(Experiment(cache,'faces_houses'),settings,2.,'UU',.9258333333333333)
   noise[variant]=fit['selected_noise'];write_json(a.output/f'{variant}-UU-calibration.json',fit)
  rate=noise[variant]
  for design in ['faces_houses','faces_objects']:
   e=Experiment(cache,design)
   for seed in reg['seeds']:
    e.draws.clear()
    for cond in ['UU','UI','IU','II']:
     plan,mem,probes=inputs(e,seed,cond,rate,rate);plans[f'{design}:{seed}']=plan
     scores=e.familiarity(probes,mem,2.).reshape(24,2)
     methods=[('original_max40',list(range(24)),scores)]
     if a.pooled_houses and design=='faces_houses':
      ids=[i for i,(cat,row) in enumerate(plan['study_order']) if cat=='houses']
      trials=[i for i,t in enumerate(plan['trials']) if t['category']=='houses']
      q=probes[[j for i in trials for j in [2*i,2*i+1]]];hm=mem[ids]
      methods += [('house_max20',trials,e.familiarity(q,hm,2.).reshape(12,2)),('house_pool200',trials,pooled_scores(q,hm).reshape(12,2))]
     for method,indices,sc in methods:
      ok=(sc[:,0]>sc[:,1]).tolist()
      for cat in sorted({plan['trials'][i]['category'] for i in indices}):
       mask=''.join(str(int(ok[j])) for j,i in enumerate(indices) if plan['trials'][i]['category']==cat)
       assert len(mask)==12
       rows.append(dict(variant=variant,model=model['model'],checkpoint_sha256=model['sha256'],design=design,method=method,seed=seed,condition=cond,category=cat,sigma=2.,noise=rate,n_trials=12,correct=mask.count('1'),accuracy=100*mask.count('1')/12,correct_mask=mask))
   print(f'{variant}: completed {design}, noise={rate}',flush=True)
 with (a.output/'all_records.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 groups=defaultdict(list)
 for r in rows:groups[r['variant'],r['design'],r['method'],r['category'],r['condition']].append(r)
 cells=[]
 for key,rs in groups.items():
  assert sorted(r['seed'] for r in rs)==reg['seeds']
  cells.append(dict(zip(['variant','design','method','category','condition'],key),**moments([r['accuracy'] for r in rs])))
 expected=2400 if a.pooled_houses else 1600
 assert len(rows)==expected
 write_json(a.output/'SUMMARY.json',dict(units='accuracy percent',uncertainty='sample SD and SEM over 50 simulation seeds; not training-seed uncertainty',reporting='same fitting/report seeds101-150, not held-out validation',noise=noise,cells=cells))
 write_json(a.output/'STIMULUS_PLANS.json',plans)
 write_json(a.output/'VALIDATED.json',dict(passed=True,rows=len(rows),cells=len(cells),seeds_per_cell=50,raw_sha256=sha(a.output/'all_records.csv')))
if __name__=='__main__':main()
