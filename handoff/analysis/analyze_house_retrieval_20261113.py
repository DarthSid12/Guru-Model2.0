import sys,json,csv,math,statistics as st
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
from house_pca_retrieval_config_20261113 import B,P,O,sha,load
sys.path.insert(0,str(P))
from engine import Experiment
from sigma import inputs,distances
from design import plan
s=load();torch.set_num_threads(1)
# Exact raw decisions from prior zero- and face-calibrated-noise sweeps.
lookup={}
for parent,labels in [(P,{'fixed_2':'zero_sigma2','alpha_0.005':'zero_small_alpha'}),(B/'operations/sdsc-kde-alpha-fine-20261111/completed-sdsc',{'fixed_2':'facecal_sigma2'})]:
 for r in csv.DictReader((parent/'results/all_trials.csv').open()):
  if r['split']=='validation' and r['sigma_label'] in labels:
   lookup[r['variant'],labels[r['sigma_label']],int(r['simulation_seed']),r['condition'],r['category']]=r
assert len(lookup)==2400
rows=[];seeds=[];replayed=0
for m in s['models']:
 v=m['record']['variant'];e=Experiment(m['cache'],'faces_houses')
 dbar=json.loads((P/'results'/(v+'-DISTANCES.json')).read_text())['d_bar']
 settings=[('zero_sigma2',0.,2.),('zero_small_alpha',0.,.005*dbar),('facecal_sigma2',m['reference_noise'],2.)]
 for seed in range(151,201):
  e.draws.clear();p=plan('faces_houses',seed);memitems=p['study_order']
  for condition in ['UU','UI','IU','II']:
   # Independent full256D centroid-neighbor geometry diagnostic,not classifier accuracy.
   means=torch.stack([e.probs[c,condition[0]][r,:10].double().mean(0) for c,r in memitems])
   probe_means=torch.stack([e.probs[t['category'],condition[1]][t[w],:32].double().mean(0) for t in p['trials'] for w in ['old','new']])
   rawdist=torch.cdist(means,probe_means).square();rawtop=rawdist.argmin(0)
   for setting,noise,sigma in settings:
    _,memory,probes=inputs(e,seed,condition,noise);d=distances(memory,probes)
    density_logs=torch.logsumexp(-d.double()/(2*sigma*sigma),dim=1)-math.log(10)
    scores=density_logs.sum(-1);best,top=scores.max(0)
    assert scores.shape==(40,48) and torch.isfinite(scores).all()
    tied=(scores==best[None,:]); masks={c:[] for c in ['faces','houses']};local=[]
    for ti,t in enumerate(p['trials']):
     cat=t['category'];old=t['old'];new=t['new'];truth=memitems.index((cat,old));oi=2*ti;ni=oi+1
     wm=int(top[oi]);wn=int(top[ni]);wc,wr=memitems[wm];nc,nr=memitems[wn]
     rawi=int(rawtop[oi]);rc,rr=memitems[rawi]
     samecat=[i for i,(c,r) in enumerate(memitems) if c==cat]
     restricted=scores[samecat].max(0).values
     correct=bool(best[oi]>best[ni]);restricted_correct=bool(restricted[oi]>restricted[ni])
     kind='correct_identity' if (wc,wr)==(cat,old) else ('other_same_category' if wc==cat else 'other_category')
     rawkind='correct_identity' if (rc,rr)==(cat,old) else ('other_same_category' if rc==cat else 'other_category')
     record=dict(variant=v,checkpoint_sha256=m['record']['sha256'],setting=setting,noise_rate=noise,sigma=sigma,simulation_seed=seed,condition=condition,category=cat,old_row=old,new_row=new,old_winning_category=wc,old_winning_row=wr,old_memory_match=kind,old_top_tie_count=int(tied[:,oi].sum()),true_identity_among_top_ties=bool(tied[truth,oi]),new_winning_category=nc,new_winning_row=nr,yin_pair_correct=int(correct),same_category_memory_only_pair_correct=int(restricted_correct),old_self_memory_score=float(scores[truth,oi]),old_best_memory_score=float(best[oi]),new_best_memory_score=float(best[ni]),raw_centroid_old_match=rawkind,raw_centroid_winning_category=rc,raw_centroid_winning_row=rr)
     local.append(record);rows.append(record);masks[cat].append('1' if correct else '0')
    for cat in ['faces','houses']:
     mask=''.join(masks[cat]);prior=lookup[v,setting,seed,condition,cat]
     assert mask==prior['correct_mask'] and mask.count('1')==int(prior['correct_pairs']);replayed+=1
     selected=[r for r in local if r['category']==cat];counts=Counter(r['old_memory_match'] for r in selected);rawcounts=Counter(r['raw_centroid_old_match'] for r in selected)
     seeds.append(dict(variant=v,setting=setting,noise_rate=noise,sigma=sigma,simulation_seed=seed,condition=condition,category=cat,n_pairs=12,full_memory_errors=12-sum(r['yin_pair_correct'] for r in selected),same_category_memory_only_errors=12-sum(r['same_category_memory_only_pair_correct'] for r in selected),n_correct_identity=counts['correct_identity'],n_other_same_category=counts['other_same_category'],n_other_category=counts['other_category'],n_top_tied=sum(r['old_top_tie_count']>1 for r in selected),raw_n_correct_identity=rawcounts['correct_identity'],raw_n_other_same_category=rawcounts['other_same_category'],raw_n_other_category=rawcounts['other_category']))
  if seed%10==0:print('PROGRESS',v,seed-150,'/50',flush=True)
assert len(rows)==28800 and len(seeds)==2400 and replayed==2400
for name,data in [('TRIALS.csv',rows),('PER_SEED.csv',seeds)]:
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def moments(a):
 sd=st.stdev(a);return dict(n=len(a),mean=st.mean(a),sd=sd,sem=sd/math.sqrt(len(a)))
groups=defaultdict(list)
for r in seeds:groups[r['variant'],r['setting'],r['condition'],r['category']].append(r)
cells=[]
for k,vals in groups.items():
 assert sorted(r['simulation_seed'] for r in vals)==list(range(151,201))
 total=600
 cell=dict(zip(['variant','setting','condition','category'],k),noise_rate=vals[0]['noise_rate'],sigma=vals[0]['sigma'],n_old_probes=total)
 for kind in ['correct_identity','other_same_category','other_category']:
  field='n_'+kind;count=sum(r[field] for r in vals)
  cell[kind]=dict(count=count,percent=100*count/total,per_seed_percent=moments([100*r[field]/12 for r in vals]))
  rawfield='raw_n_'+kind;count=sum(r[rawfield] for r in vals)
  cell['raw_centroid_'+kind]=dict(count=count,percent=100*count/total)
 cell['top_tied_probes']=sum(r['n_top_tied'] for r in vals)
 for field in ['full_memory_errors','same_category_memory_only_errors']:cell[field]=moments([r[field] for r in vals])
 cell['paired_error_reduction_removing_other_category']=moments([r['full_memory_errors']-r['same_category_memory_only_errors'] for r in vals])
 cells.append(cell)
summary=dict(source_script_sha256=sha(Path(__file__)),source_spec_sha256=sha(P/'SPEC.json'),model_records=[m['record'] for m in s['models']],readout='Float64 stable unfloored KDE;sum32testfixations then max over40study memories,10fixations each',design='20faces+20houses;12old/new pairs/category;all4conditions',simulation_seeds=list(range(151,201)),cells=cells,verification=dict(passed=True,trial_rows=28800,per_seed_rows=2400,parent_rows_and_complete_decision_masks_reproduced_exactly=replayed),notes=['Old-probe memory identity retrieval is distinct from Yin old/new accuracy.','Correctidentity means exact studied photograph in memory;other_same_category means another house for house probes;other_category means a face.','Novel probes have no correct identity in memory;their winning category is retained in raw records but not counted as identity confusion.','Top ties use original torch max first-index rule;tie counts and true-identity inclusion are retained.','Raw-centroid diagnostic uses unbinarized256D sigmoid vectors:studymean10fixations versus probemean32fixations,Euclidean nearest memory. It is not classifier-head performance.','Restricting search to the same category reuses unchanged scores/codes/plans;only category competition is removed as a diagnostic.','No classifier logits are cached;novel Yin house identities do not have matching training-head class labels.','All computations occur locally from exact frozen caches;no retraining or data changes.'])
(O/'RETRIEVAL_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
lines=['# What does the memory system confuse?','','Current no-generic LP/CNN,epoch124 trainingseed46.50validation simulation seeds151–200,20houses+20faces study,12old/new test pairs/category. All results use the full256D code. The three settings reuse exactly2,400prior category-level counts and full decision masks.','','Identity retrieval below uses only studied(old)house probes. Each cell pools600such probes. Wronghouse and face refer to the top-scoring stored image;this is distinct from whether the studied probe beats the novel probe.','']
for setting in ['zero_sigma2','zero_small_alpha','facecal_sigma2']:
 lines+=['## '+setting,'','| Model | Condition | Correct house | Another house | A face | Tied top | Yin errors,full memory | Yin errors,house-only search |','|---|---|---:|---:|---:|---:|---:|---:|']
 for v in ['lp','cnn']:
  for cond in ['UU','UI','IU','II']:
   c=next(x for x in cells if (x['variant'],x['setting'],x['condition'],x['category'])==(v,setting,cond,'houses'))
   def err(name):x=c[name];return '%.2f ± %.2f'%(x['mean'],x['sd'])
   lines.append('| %s | %s | %.2f%% | %.2f%% | %.2f%% | %d/600 | %s | %s |'%(v.upper(),cond,c['correct_identity']['percent'],c['other_same_category']['percent'],c['other_category']['percent'],c['top_tied_probes'],err('full_memory_errors'),err('same_category_memory_only_errors')))
 lines.append('')
lines+=['## Raw representation nearest-centroid diagnostic','','Unbinarized full256D vectors,not PCA coordinates and not the trained classification head. Studied target query mean uses32fixations;each memory mean uses10. These outcomes do not depend on the noise/sigma setting.','','| Model | Condition | Correct house | Another house | A face |','|---|---|---:|---:|---:|']
for v in ['lp','cnn']:
 for cond in ['UU','UI','IU','II']:
  c=next(x for x in cells if (x['variant'],x['setting'],x['condition'],x['category'])==(v,'zero_sigma2',cond,'houses'))
  lines.append('| %s | %s | %.2f%% | %.2f%% | %.2f%% |'%(v.upper(),cond,c['raw_centroid_correct_identity']['percent'],c['raw_centroid_other_same_category']['percent'],c['raw_centroid_other_category']['percent']))
lines+=['','Error values are errors out of12,mean ± sampleSD across50simulation seeds. Percentages pool600oldprobes;per-seedSD/SEM of percentages are retained in JSON. The PCA pictures discard variance and are not used to calculate these matches.','']
(O/'RETRIEVAL_REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines))
