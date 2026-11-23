import csv,json,math,os,statistics,subprocess,sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from common import ROOT,sha,atomic,spec,verify,CONDITIONS,PROTOCOLS,DESIGNS,HUMAN,SEEDS,rates

def write_csv(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.csv.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 tmp.replace(path)
def load_exp(model,design):
 import torch
 from engine import Experiment
 torch.set_num_threads(1);folder=ROOT/'cache'/model['model'];d=json.loads((folder/'DONE.json').read_text())
 assert d['model']==model and d['spec_sha256']==sha(ROOT/'SPEC.json')
 from pack_houses import verify_packed
 verify_packed();assert d['house_packing_sha256']==sha(ROOT/'PACKED_HOUSES.json')
 for n,h in d['files'].items():assert sha(folder/n)==h
 return Experiment(folder,design)
def fit(get,condition,target,test_noise=None):
 def at(p):return get(condition,p,p if test_noise is None else test_noise)['means']['faces']
 curve={p:at(p) for p in [round(i*.05,2) for i in range(11)]};best=min(curve,key=lambda p:(abs(curve[p]-target),p))
 for p in sorted({round(best+i*.01,2) for i in range(-4,5) if 0<=round(best+i*.01,2)<=.5}):curve[p]=at(p)
 best=min(curve,key=lambda p:(abs(curve[p]-target),p))
 return best,dict(target=target,achieved=curve[best],curve=curve,test_noise_held_at=test_noise,status='within_sampled_range' if min(curve.values())<=target<=max(curve.values()) else 'target_outside_sampled_range')
def calibration_tasks():return [(m,d) for m in spec()['models'] for d in DESIGNS if 'faces' in DESIGNS[d]]
def calibrate(index):
 m,design=calibration_tasks()[index];folder=ROOT/'calibration'/m['model']/design;folder.mkdir(parents=True,exist_ok=True);dest=folder/'calibration.json'
 if dest.exists():
  assert json.loads(dest.read_text())['spec_sha256']==sha(ROOT/'SPEC.json');return
 exp=load_exp(m,design);journal=folder/'samples.jsonl';records={}
 if journal.exists():
  good=0
  with journal.open('rb') as f:
   for line in f:
    try:r=json.loads(line)
    except json.JSONDecodeError:break
    assert r['spec_sha256']==sha(ROOT/'SPEC.json') and r['key'] not in records
    records[r['key']]=r;good+=len(line)
  if good!=journal.stat().st_size:
   with journal.open('r+b') as f:f.truncate(good)
 def get(condition,sn,tn):
  key=f'{condition}:{sn:.2f}:{tn:.2f}'
  if key not in records:
   values=[]
   for seed in SEEDS:exp.draws.clear();values.append(exp.measure(seed,condition,sn,tn))
   r=dict(key=key,spec_sha256=sha(ROOT/'SPEC.json'),means={c:statistics.mean(v[c] for v in values) for c in DESIGNS[design]},seeds=list(SEEDS),outcomes=values)
   with journal.open('a') as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
   records[key]=r;print('PROBE',m['model'],design,key,r['means'],flush=True)
  return records[key]
 p1,uufit=fit(get,'UU',HUMAN['faces']['UU']);fits={}
 for protocol in PROTOCOLS:
  if protocol=='single_uu':p2=p1;iifit={'fitted':False,'held_out':True}
  else:p2,iifit=fit(get,'II',HUMAN['faces']['II'],p1 if protocol=='study_only_fitted' else None)
  fits[protocol]=dict(p1=p1,p2=p2,UU_fit=uufit,II_fit=iifit)
 atomic(dest,dict(spec_sha256=sha(ROOT/'SPEC.json'),model=m['model'],design=design,anchor='faces',fits=fits,calibration_seeds=list(SEEDS),journal_sha256=sha(journal)))
 print('CALIBRATED',m['model'],design,{k:(v['p1'],v['p2']) for k,v in fits.items()},flush=True)
def evaluation_tasks():return [(m,d) for m in spec()['models'] for d in DESIGNS]
def evaluate(index):
 m,design=evaluation_tasks()[index];folder=ROOT/'results'/m['model']/design;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'DONE.json').exists():
  d=json.loads((folder/'DONE.json').read_text());assert d['spec_sha256']==sha(ROOT/'SPEC.json') and d['csv_sha256']==sha(folder/'results.csv');return
 cd=design if 'faces' in DESIGNS[design] else 'faces_only';cp=ROOT/'calibration'/m['model']/cd/'calibration.json';cal=json.loads(cp.read_text());assert cal['spec_sha256']==sha(ROOT/'SPEC.json')
 exp=load_exp(m,design);rows=[]
 for start in range(101,151,5):
  path=folder/f'seeds-{start}-{start+4}.csv';receipt=path.with_suffix('.json')
  if receipt.exists():
   d=json.loads(receipt.read_text());assert d['spec_sha256']==sha(ROOT/'SPEC.json') and d['csv_sha256']==sha(path)
   selected=list(csv.DictReader(path.open()))
  else:
   selected=[]
   for seed in range(start,start+5):
    exp.draws.clear();samples={}
    for protocol in PROTOCOLS:
     p1,p2=[cal['fits'][protocol][p] for p in ['p1','p2']]
     for condition in CONDITIONS:
      sn,tn=rates(protocol,condition,p1,p2);key=(condition,sn,tn)
      if key not in samples:samples[key]=exp.measure(seed,condition,sn,tn)
      for cat,acc in samples[key].items():
       n=DESIGNS[design][cat]
       selected.append(dict(model=m['model'],variant=m['variant'],backbone='resnet18',epoch=124,training_seed=m['training_seed'],checkpoint_sha256=m['sha256'],design=design,category=cat,protocol=protocol,condition=condition,simulation_seed=seed,calibration_design=cd,p1=p1,p2=p2,study_noise=sn,test_noise=tn,study_items_category=n['study'],memory_items=40,n_trials=n['trials'],correct_pairs=round(acc*n['trials']),accuracy_pct=100*acc))
   write_csv(path,selected);atomic(receipt,dict(spec_sha256=sha(ROOT/'SPEC.json'),csv_sha256=sha(path)))
  assert len(selected)==5*len(PROTOCOLS)*4*len(DESIGNS[design]);rows.extend(selected)
  print('PROGRESS',m['model'],design,start+4-100,'/50',flush=True)
 write_csv(folder/'results.csv',rows);atomic(folder/'DONE.json',dict(spec_sha256=sha(ROOT/'SPEC.json'),csv_sha256=sha(folder/'results.csv'),calibration_sha256=sha(cp),rows=len(rows),seeds_per_cell=50))
def moments(v):
 sd=statistics.stdev(v);return dict(n=len(v),mean_pct=statistics.mean(v),sd_pct=sd,sem_pct=sd/math.sqrt(len(v)))
def aggregate():
 s=spec();rows=[];groups=defaultdict(list);pairs=defaultdict(dict)
 for m,design in evaluation_tasks():
  folder=ROOT/'results'/m['model']/design;d=json.loads((folder/'DONE.json').read_text());assert d['spec_sha256']==sha(ROOT/'SPEC.json') and d['csv_sha256']==sha(folder/'results.csv')
  selected=list(csv.DictReader((folder/'results.csv').open()));assert len(selected)==50*len(PROTOCOLS)*4*len(DESIGNS[design]);rows.extend(selected)
  cd=design if 'faces' in DESIGNS[design] else 'faces_only';cp=ROOT/'calibration'/m['model']/cd/'calibration.json';cal=json.loads(cp.read_text())
  assert sha(cp)==d['calibration_sha256'] and cal['spec_sha256']==sha(ROOT/'SPEC.json')
  for r in selected:
   assert r['checkpoint_sha256']==m['sha256'] and int(r['epoch'])==124 and int(r['training_seed'])==m['training_seed']
   n=DESIGNS[design][r['category']];assert int(r['n_trials'])==n['trials'] and int(r['study_items_category'])==n['study'] and int(r['memory_items'])==40
   assert r['calibration_design']==cd
   p1,p2=float(r['p1']),float(r['p2']);assert (p1,p2)==tuple(cal['fits'][r['protocol']][p] for p in ['p1','p2'])
   assert rates(r['protocol'],r['condition'],p1,p2)==(float(r['study_noise']),float(r['test_noise']))
   acc=float(r['accuracy_pct']);assert 0<=int(r['correct_pairs'])<=n['trials'] and abs(acc-int(r['correct_pairs'])/n['trials']*100)<1e-8
   key=(m['model'],design,r['category'],r['protocol'],r['condition']);groups[key].append(r)
   pk=(*key[:-1],int(r['simulation_seed']));assert r['condition'] not in pairs[pk];pairs[pk][r['condition']]=acc
 assert len(rows)==s['expected_rows'];cells=[];contrast=defaultdict(list)
 for key,cell in groups.items():
  assert sorted(int(r['simulation_seed']) for r in cell)==list(SEEDS)
  cells.append(dict(zip(['model','design','category','protocol','condition'],key),**moments([float(r['accuracy_pct']) for r in cell]),p1=float(cell[0]['p1']),p2=float(cell[0]['p2']),n_trials=int(cell[0]['n_trials'])))
 assert len(cells)==s['expected_rows']//50
 for key,v in pairs.items():
  assert set(v)==set(CONDITIONS);contrast[key[:-1]].append(v['UU']-v['II'])
 contrasts=[dict(zip(['model','design','category','protocol'],key),**moments(v)) for key,v in contrast.items()]
 out=ROOT/'results';write_csv(out/'all_trials.csv',rows)
 atomic(out/'SUMMARY.json',dict(models=s['models'],cells=cells,inversion_effect_UU_minus_II=contrasts,uncertainty='Sample SD and Monte Carlo SEM across 50 simulation seeds, fixed training seed46 and epoch124. Noise fitted on the same50 seeds. Human anchors use12 pairs/category.'))
 lines=['# ResNet-18 Yin experiments: unseen DalHouses','','Accuracy % ± sample SD across50 simulation seeds. SEM retained in SUMMARY.json.','']
 for protocol in PROTOCOLS:
  lines += [f'**{protocol}**','','| Model/design/category | p1,p2 | UU | UI | II | IU | UU–II |','|---|---|---:|---:|---:|---:|---:|']
  for m,design in evaluation_tasks():
   for cat in DESIGNS[design]:
    cs={r['condition']:r for r in cells if (r['model'],r['design'],r['category'],r['protocol'])==(m['model'],design,cat,protocol)}
    co=next(r for r in contrasts if (r['model'],r['design'],r['category'],r['protocol'])==(m['model'],design,cat,protocol))
    values=' | '.join(f"{cs[c]['mean_pct']:.2f} ± {cs[c]['sd_pct']:.2f}" for c in CONDITIONS)
    lines.append(f"| {m['label']}/{design}/{cat} | {cs['UU']['p1']:.2f},{cs['UU']['p2']:.2f} | {values} | {co['mean_pct']:.2f} ± {co['sd_pct']:.2f} |")
  lines.append('')
 (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n');atomic(out/'VALIDATED.json',dict(passed=True,spec_sha256=sha(ROOT/'SPEC.json'),csv_sha256=sha(out/'all_trials.csv'),summary_sha256=sha(out/'SUMMARY.json'),rows=len(rows),cells=len(cells),seeds_per_cell=50))
def pilot():
 checks=[]
 for m,design in evaluation_tasks():
  e=load_exp(m,design)
  for condition in CONDITIONS:
   values,detail=e.measure(101,condition,.25,.30,details=True)
   assert detail['memory_items']==40 and all(math.isfinite(x) for pair in detail['scores'] for x in pair)
   assert set(values)==set(DESIGNS[design]) and all(0<=x<=1 for x in values.values())
   checks.append(dict(model=m['model'],design=design,condition=condition,outcomes=values))
 atomic(ROOT/'PILOT.json',dict(passed=True,spec_sha256=sha(ROOT/'SPEC.json'),checks=checks))
def main():
 mode=sys.argv[1];j=json.loads((ROOT/'JOBS.json').read_text());assert j['registration_complete'] and os.environ['SLURM_JOB_ID'] in j['jobs'][mode] and j['spec_sha256']==sha(ROOT/'SPEC.json');verify()
 import torch
 torch.set_num_threads(1)
 if mode=='smoke':
  subprocess.run([sys.executable,'-m','unittest','test_science','-v'],cwd=ROOT,check=True)
  import pack_houses;pack_houses.main()
  import cache;cache.main();pilot()
  atomic(ROOT/'SMOKE_PASSED.json',dict(passed=True,spec_sha256=sha(ROOT/'SPEC.json'),pilot_sha256=sha(ROOT/'PILOT.json'),house_packing_sha256=sha(ROOT/'PACKED_HOUSES.json'),job_id=os.environ['SLURM_JOB_ID'],environment=dict(python=sys.version,torch=torch.__version__)))
 elif mode=='production':
  smoke=json.loads((ROOT/'SMOKE_PASSED.json').read_text());assert smoke['passed'] and smoke['spec_sha256']==sha(ROOT/'SPEC.json') and smoke['pilot_sha256']==sha(ROOT/'PILOT.json')
  from pack_houses import verify_packed
  verify_packed();assert smoke['house_packing_sha256']==sha(ROOT/'PACKED_HOUSES.json')
  for submode,tasks in [('calibration-task',calibration_tasks()),('evaluation-task',evaluation_tasks())]:
   with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda i:subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),submode,str(i)],check=True),range(len(tasks))))
  aggregate()
  subprocess.run([sys.executable,str(ROOT/'add_human_reference.py'),str(ROOT/'results')],check=True)
 print('DONE',mode,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='calibration-task':calibrate(int(sys.argv[2]))
 elif sys.argv[1]=='evaluation-task':evaluate(int(sys.argv[2]))
 else:main()
