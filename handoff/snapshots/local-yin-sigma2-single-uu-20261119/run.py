import sys,json,csv,hashlib,statistics as st,datetime
from pathlib import Path
from collections import Counter,defaultdict
B=Path('/Users/darthsid/Documents/ChatGPT/GURU');O=Path(__file__).resolve().parent;P=B/'operations/local-yin-two-sigma-tables-20261116'
sys.path.insert(0,str(P))
import torch
from engine import Experiment
from kde import inputs,score
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
s=json.loads((P/'SPEC.json').read_text())
for n,h in s['files'].items():assert sha(P/n)==h,n
for n,h in s['external_files'].items():assert sha(n)==h,n
torch.set_num_threads(1)
fits={}
for m in s['models']:
 v=m['record']['variant'];f=json.loads((P/'calibration'/f'{v}-fixed_sigma.json').read_text());u=f['UU_fit']
 assert u['condition']=='UU' and u['test_noise_held_at'] is None
 assert min(u['curve'],key=lambda x:(abs(x['accuracy']-u['target']),x['rate']))['rate']==u['selected_noise']==f['p1']
 fits[v]=dict(noise=u['selected_noise'],UU_fit=u,source_sha256=sha(P/'calibration'/f'{v}-fixed_sigma.json'))
assert not (O/'SPEC.json').exists()
spec=dict(sigma=2.,protocol='single_uu',noise_assignment='One rate per model; same rate for every study/test presentation, every condition, both study groups',fits=fits,seeds=s['report_seeds'],designs=s['design_definitions'],models=[m['record'] for m in s['models']],parent_spec_sha256=sha(P/'SPEC.json'),runner_sha256=sha(__file__),reporting='Same 50 fitting/report seeds101-150, not independent validation; fit UU only; II is a prediction')
save(O/'SPEC.json',spec);save(O/'LATEST_STATUS.json',dict(status='RUNNING'))
rows=[];trials=[];plans={};control=list(csv.DictReader((P/'results/all_records.csv').open()));lookup={(r['variant'],r['design'],int(r['simulation_seed']),r['condition'],r['category']):r for r in control if r['experiment']=='fixed_sigma'}
checks=0
for m in s['models']:
 v=m['record']['variant'];pnoise=fits[v]['noise']
 for d in s['designs']:
  e=Experiment(m['cache'],d)
  for seed in s['report_seeds']:
   e.draws.clear()
   for cond in s['conditions']:
    p,mem,q=inputs(e,seed,cond,pnoise,pnoise)
    assert mem.shape==(40,10,256) and q.shape==(48,32,256)
    cats=s['design_definitions'][d]
    assert Counter(c for c,r in p['study_order'])==Counter({c:20 for c in cats})
    assert Counter(t['category'] for t in p['trials'])==Counter({c:12 for c in cats})
    for g in p['groups'].values():assert not set(g['studied'])&set(g['novel']) and set(g['tested_old'])<=set(g['studied'])
    plans[f'{d}:{seed}']=p
    scores=e.familiarity(q,mem,2.).reshape(24,2);ok=(scores[:,0]>scores[:,1]).tolist()
    if seed==101:
     alt,_=score(q,mem,2.);assert torch.equal(alt,scores.flatten())
    for j,t in enumerate(p['trials']):trials.append(dict(variant=v,design=d,seed=seed,condition=cond,category=t['category'],old=t['old'],new=t['new'],old_score=float(scores[j,0]),new_score=float(scores[j,1]),correct=int(ok[j])))
    for cat in cats:
     mask=''.join('1' if ok[j] else '0' for j,t in enumerate(p['trials']) if t['category']==cat)
     if cond in ['UU','UI']:
      assert mask==lookup[v,d,seed,cond,cat]['correct_mask'];checks+=1
     rows.append(dict(variant=v,model=m['record']['model'],checkpoint_sha256=m['record']['sha256'],design=d,seed=seed,condition=cond,category=cat,sigma=2.,noise=pnoise,study_noise=pnoise,test_noise=pnoise,n_trials=12,correct=mask.count('1'),accuracy=100*mask.count('1')/12,correct_mask=mask))
  print('Completed',v,d,'noise',pnoise,flush=True)
assert len(rows)==1600 and len(trials)==19200 and checks==800
for name,data in [('all_records.csv',rows),('all_trials.csv',trials)]:
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
save(O/'STIMULUS_PLANS.json',plans)
groups=defaultdict(list);paired=defaultdict(dict)
for r in rows:
 groups[r['variant'],r['design'],r['category'],r['condition']].append(r)
 paired[r['variant'],r['design'],r['category'],r['seed']][r['condition']]=r['accuracy']
def moments(a):return dict(mean=st.mean(a),sd=st.stdev(a),sem=st.stdev(a)/len(a)**.5)
cells=[]
for key,rs in groups.items():
 assert sorted(x['seed'] for x in rs)==list(range(101,151))
 cells.append(dict(zip(['variant','design','category','condition'],key),**moments([r['accuracy'] for r in rs])))
for v in fits:
 uu=next(c for c in cells if (c['variant'],c['design'],c['category'],c['condition'])==(v,'faces_houses','faces','UU'))
 assert abs(uu['mean']/100-fits[v]['UU_fit']['achieved'])<1e-12
H={(x['category_reference_for'],x['condition']):x for x in json.loads((Path(s['parent_root'])/'results/SUMMARY_WITH_HUMANS.json').read_text())['human_conditions']}
C=s['conditions'];lines=['# Fixed sigma 2, one noise calibrated on face UU','','Accuracy %, mean ± sample SD. 50 simulation seeds 101–150; same fitting/report seeds, not independent validation. LP noise = %.3f; CNN noise = %.3f. Each rate is used at study and test in all orientations and transferred unchanged from faces/houses to faces/objects.'%(fits['lp']['noise'],fits['cnn']['noise']),'','Each study group contains 40 images: 20 per category. Test: 12 old/new pairs per category, comparing only within categories. Original KDE, sigma 2 for both models, 10 study and 32 test fixations, epoch124/trainingseed46 checkpoints. II is not calibrated.']
for d in s['designs']:
 lines+=['','## '+('20 faces + 20 houses' if d=='faces_houses' else '20 faces + 20 objects')]
 for cat in s['design_definitions'][d]:
  lines+=['','### '+cat.capitalize(),'','| System | Noise | UU | UI | IU | II | UU − avg(UI,IU,II) |','|---|---:|---:|---:|---:|---:|---:|']
  h=[H[cat,c] for c in C];hc=h[0]['mean_pct']-sum(x['mean_pct'] for x in h[1:])/3
  label='Human airplanes reference' if cat=='objects' else 'Human'
  lines.append('| '+label+' | — | '+' | '.join('%.2f ± %.2f'%(x['mean_pct'],x['sd_pct']) for x in h)+f' | {hc:.2f} |')
  for v in ['lp','cnn']:
   cs=[next(x for x in cells if (x['variant'],x['design'],x['category'],x['condition'])==(v,d,cat,c)) for c in C]
   contrasts=[]
   for seed in s['report_seeds']:
    a=paired[v,d,cat,seed];contrasts.append(a['UU']-(a['UI']+a['IU']+a['II'])/3)
   delta=moments(contrasts)
   lines.append('| '+v.upper()+f' | {fits[v]["noise"]:.3f} | '+' | '.join('%.2f ± %.2f'%(x['mean'],x['sd']) for x in cs)+' | %.2f ± %.2f |'%(delta['mean'],delta['sd']))
lines+=['','Model SD measures variation across simulation seeds; human SD is across participants. Human contrast SD is unavailable. The historical airplane reference uses different stimuli. The object bank contains 16 categories outside the revised training diet.']
(O/'RESULTS_ACCURACY.md').write_text('\n'.join(lines)+'\n');save(O/'SUMMARY.json',dict(cells=cells))
receipt=dict(passed=True,status='COMPLETED_VERIFIED',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec_sha256=sha(O/'SPEC.json'),records=1600,trial_decisions=19200,seeds_per_cell=50,UU_UI_parent_masks_verified=checks,twenty_study_per_category=True,twelve_test_pairs_per_category=True,single_noise_study_and_test_all_conditions=True,noise_transfer_unchanged=True,records_sha256=sha(O/'all_records.csv'))
save(O/'VERIFICATION.json',receipt);save(O/'LATEST_STATUS.json',receipt);print((O/'RESULTS_ACCURACY.md').read_text())
