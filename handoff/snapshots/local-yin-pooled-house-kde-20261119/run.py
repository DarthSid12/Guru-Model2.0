import sys,json,csv,hashlib,statistics as st,datetime
from pathlib import Path
from collections import defaultdict
import torch
B=Path('/Users/darthsid/Documents/ChatGPT/GURU');O=Path(__file__).resolve().parent
P=B/'operations/local-yin-two-sigma-tables-20261116';S=B/'operations/local-yin-sigma2-single-uu-20261119'
sys.path.insert(0,str(P))
from engine import Experiment
from kde import inputs
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def csvout(p,rs):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
torch.set_num_threads(1)
parent=json.loads((P/'SPEC.json').read_text());single=json.loads((S/'SPEC.json').read_text());verified=json.loads((S/'VERIFICATION.json').read_text())
assert verified['passed'] and verified['spec_sha256']==sha(S/'SPEC.json') and verified['records_sha256']==sha(S/'all_records.csv')
for n,h in parent['files'].items():assert sha(P/n)==h
for n,h in parent['external_files'].items():assert sha(n)==h
assert not (O/'SPEC.json').exists()
spec=dict(parent_single_spec_sha256=sha(S/'SPEC.json'),source_operation=str(S),runner_sha256=sha(__file__),sigma=2.,noise={v:single['fits'][v]['noise'] for v in ['lp','cnn']},noise_rule='Frozen face-UU calibrated single noise, unchanged for every condition and readout; no refit',models=single['models'],design='faces_houses',study='20 faces plus20 houses;10 fixations each',test='12 same-category old/new house pairs;32 fixations each',seeds=single['seeds'],methods=['original_max40','house_max20','house_pool200'],pooled_formula='sum_t log(mean_{i=1..20,j=1..10} exp(-||q_t-m_ij||^2/8)+1e-12)',scope='Only studied house fixation vectors enter the pooled bank; no test vectors or novel images',ties='incorrect',baseline='Preserved unchanged')
save(O/'SPEC.json',spec);save(O/'LATEST_STATUS.json',dict(status='RUNNING'))
ref={(r['variant'],int(r['seed']),r['condition']):r for r in csv.DictReader((S/'all_records.csv').open()) if r['design']=='faces_houses' and r['category']=='houses'}
rows=[];trials=[];plans={};control=0;independent=0;maxdiff=0.
for m in parent['models']:
 v=m['record']['variant'];noise=spec['noise'][v];e=Experiment(m['cache'],'faces_houses')
 for seed in spec['seeds']:
  e.draws.clear()
  for cond in ['UU','UI','IU','II']:
   p,mem,allq=inputs(e,seed,cond,noise,noise);mi=[j for j,(c,r) in enumerate(p['study_order']) if c=='houses'];ti=[j for j,t in enumerate(p['trials']) if t['category']=='houses'];qi=[k for j in ti for k in [2*j,2*j+1]]
   assert len(mi)==20 and len(ti)==12
   assert not set(p['groups']['houses']['studied'])&set(p['groups']['houses']['novel'])
   plans[str(seed)]=p;hm=mem[mi];q=allq[qi]
   original=e.familiarity(allq,mem,2.).reshape(24,2)[ti]
   separate=e.familiarity(q,hm,2.).reshape(12,2)
   mm=hm.reshape(200,256);qq=q.reshape(-1,256)
   distances=(mm.square().sum(1)[:,None]+qq.square().sum(1)[None,:]-2*(mm@qq.T)).clamp_min_(0)
   pooled=torch.log(torch.exp(-distances/8).mean(0).reshape(24,32)+1e-12).double().sum(1).reshape(12,2)
   # Independently express pooling as one memory with 200 fixations in the frozen engine.
   check=e.familiarity(q,hm.reshape(1,200,256),2.).reshape(12,2)
   maxdiff=max(maxdiff,float((check-pooled).abs().max()));assert torch.equal(check,pooled);independent+=24
   assert bool((pooled.max()<=0))
   for method,scores in [('original_max40',original),('house_max20',separate),('house_pool200',pooled)]:
    ok=(scores[:,0]>scores[:,1]).tolist();mask=''.join(str(int(x)) for x in ok)
    if method=='original_max40':assert mask==ref[v,seed,cond]['correct_mask'];control+=1
    rows.append(dict(variant=v,method=method,seed=seed,condition=cond,sigma=2.,noise=noise,n_trials=12,correct=sum(ok),accuracy=100*sum(ok)/12,correct_mask=mask,ties=int((scores[:,0]==scores[:,1]).sum())))
    for j,idx in enumerate(ti):
     t=p['trials'][idx];trials.append(dict(variant=v,method=method,seed=seed,condition=cond,old=t['old'],new=t['new'],old_score=float(scores[j,0]),new_score=float(scores[j,1]),correct=int(ok[j])))
 print('Completed',v,flush=True)
assert len(rows)==1200 and len(trials)==14400 and control==400 and independent==9600
csvout(O/'all_records.csv',rows);csvout(O/'all_trials.csv',trials);save(O/'STIMULUS_PLANS.json',plans)
groups=defaultdict(list)
for r in rows:groups[r['variant'],r['method'],r['condition']].append(r)
def moments(xs):return dict(mean=st.mean(xs),sd=st.stdev(xs),sem=st.stdev(xs)/len(xs)**.5)
cells=[]
for key,rs in groups.items():
 assert sorted(r['seed'] for r in rs)==spec['seeds'];cells.append(dict(zip(['variant','method','condition'],key),**moments([r['accuracy'] for r in rs]),ties=sum(r['ties'] for r in rs)))
contrasts=[]
for v in ['lp','cnn']:
 for c in ['UU','UI','IU','II']:
  old={r['seed']:r['accuracy'] for r in groups[v,'original_max40',c]}
  dif=[r['accuracy']-old[r['seed']] for r in groups[v,'house_pool200',c]]
  contrasts.append(dict(variant=v,condition=c,**moments(dif)))
save(O/'SUMMARY.json',dict(cells=cells,pooled_minus_original=contrasts))
lines=['# Pooled house KDE experiment','','Accuracy % ± sample SD across50 simulation seeds101–150. Fixed sigma2. LPnoise0.373, CNNnoise0.364 atstudy andtest, unchanged from faceUU calibration.20faces+20housesstudied,12housepairs percondition. The existing baseline is unchanged.','','The pooled KDE averages all200 studiedhouse kernels before taking the log for eachtestfixation, then sums over32testfixations. No identity maximization remains. Test/novel fixationvectors never enter memory.','','| Model / readout | UU | UI | IU | II |','|---|---:|---:|---:|---:|','| Human reference |81.42 ±13.33|76.17 ±15.25|71.42 ±12.25|71.50 ±11.33|']
labels={'original_max40':'Current: best of40memories','house_max20':'Best of20housememories','house_pool200':'Pooled200housefixations'}
for v in ['lp','cnn']:
 for method in spec['methods']:
  cs=[next(x for x in cells if (x['variant'],x['method'],x['condition'])==(v,method,c)) for c in ['UU','UI','IU','II']]
  lines.append('|'+v.upper()+' '+labels[method]+'|'+'|'.join('%.2f ± %.2f'%(x['mean'],x['sd']) for x in cs)+'|')
lines+=['','Model SD is across Monte Carlo seeds, human SD across participants. Same fitting/report seeds as the parent experiment, not independent validation. The pooled arm assumes the stimulus category is known to select house memories.']
(O/'RESULTS.md').write_text('\n'.join(lines)+'\n')
receipt=dict(status='COMPLETED_VERIFIED',passed=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec_sha256=sha(O/'SPEC.json'),records=len(rows),trials=len(trials),original_control_category_masks=control,pooled_scores_independently_verified=independent,max_score_difference=maxdiff,all_source_hashes_verified=True,records_sha256=sha(O/'all_records.csv'),trials_sha256=sha(O/'all_trials.csv'))
save(O/'VERIFICATION.json',receipt);save(O/'LATEST_STATUS.json',receipt);print((O/'RESULTS.md').read_text())
