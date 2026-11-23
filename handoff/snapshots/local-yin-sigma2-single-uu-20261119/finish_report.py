import sys,json,csv,hashlib,statistics as st,datetime
from pathlib import Path
from collections import defaultdict
O=Path(__file__).resolve().parent;P=O.parent/'local-yin-two-sigma-tables-20261116'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
s=json.loads((P/'SPEC.json').read_text());spec=json.loads((O/'SPEC.json').read_text());fits=spec['fits']
assert sha(O/'run.py')==spec['runner_sha256']
rows=list(csv.DictReader((O/'all_records.csv').open()));trials=list(csv.DictReader((O/'all_trials.csv').open()));assert len(rows)==1600 and len(trials)==19200
for r in rows:
 r['seed']=int(r['seed']);r['accuracy']=float(r['accuracy'])
 assert float(r['sigma'])==2. and float(r['noise'])==float(r['study_noise'])==float(r['test_noise'])==fits[r['variant']]['noise']
 assert int(r['n_trials'])==12 and r['correct_mask'].count('1')==int(r['correct'])
 assert abs(r['accuracy']-100*int(r['correct'])/12)<1e-12
T=defaultdict(list)
for t in trials:
 assert int(t['correct'])==int(float(t['old_score'])>float(t['new_score']))
 T[t['variant'],t['design'],int(t['seed']),t['condition'],t['category']].append(t)
for r in rows:
 ts=T[r['variant'],r['design'],r['seed'],r['condition'],r['category']]
 assert len(ts)==12 and ''.join(t['correct'] for t in ts)==r['correct_mask']
parent={(r['variant'],r['design'],int(r['simulation_seed']),r['condition'],r['category']):r for r in csv.DictReader((P/'results/all_records.csv').open()) if r['experiment']=='fixed_sigma'}
checks=0
for r in rows:
 if r['condition'] in ['UU','UI']:
  assert r['correct_mask']==parent[r['variant'],r['design'],r['seed'],r['condition'],r['category']]['correct_mask'];checks+=1
assert checks==800
save(O/'REPORT_RECOVERY.json',dict(reason='Original simulation completed and saved all raw rows; reporting failed on an unescaped percent sign. Rebuilt report from unchanged raw results.',source_runner_sha256=sha(O/'run.py'),report_runner_sha256=sha(__file__)))
source=(O/'run.py').read_text();tail=source[source.index('groups=defaultdict(list);paired=defaultdict(dict)'):]
tail=tail.replace('Accuracy %, mean','Accuracy %%, mean')
exec(compile(tail,str(O/'run.py')+' (report recovery)','exec'))
