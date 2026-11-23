from pathlib import Path
import json,math
B=Path('/Users/darthsid/Documents/ChatGPT/GURU')
hum=json.loads((B/'operations/sdsc-yin-no-generic-s46-20261111/completed-sdsc/results/SUMMARY_WITH_HUMANS.json').read_text())
H={(r['category_reference_for'],r['condition']):r['mean_errors_reported'] for r in hum['human_conditions']}
C=['UU','UI','IU','II']
paths={'coarse':B/'operations/sdsc-kde-sigma-sweep-20261111/completed-sdsc/results/SUMMARY.json','fine_noisy':B/'operations/sdsc-kde-alpha-fine-20261111/completed-sdsc/results/SUMMARY.json','fine_zero':B/'operations/local-kde-alpha-zero-20261112-v2/results/SUMMARY.json'}
out=[]
for source,p in paths.items():
 data=json.loads(p.read_text())['cells'];groups={}
 for r in data:
  key=(r['variant'],r['noise_rate'],r['sigma_label'],r['sigma'],r.get('readout','stable_unfloored'),r.get('split','development'))
  groups.setdefault(key,{})[r['category'],r['condition']]=r
 for key,g in groups.items():
  assert len(g)==8
  errors=[g[cat,cond]['mean_errors']-H[cat,cond] for cat in ['faces','houses'] for cond in C]
  rmse=math.sqrt(sum(x*x for x in errors)/8)
  # Pairwise condition order agreement; equal human means would not be scored.
  order={}
  for cat in ['faces','houses']:
   comparisons=[]
   for i,c1 in enumerate(C):
    for c2 in C[i+1:]:
     human=H[cat,c1]-H[cat,c2];model=g[cat,c1]['mean_errors']-g[cat,c2]['mean_errors']
     comparisons.append(int(human*model>0))
   order[cat]=sum(comparisons)
  offset_rmse=math.sqrt(sum(((g[cat,c]['mean_errors']-sum(g[cat,z]['mean_errors'] for z in C)/4)-(H[cat,c]-sum(H[cat,z] for z in C)/4))**2 for cat in ['faces','houses'] for c in C)/8)
  out.append(dict(source=source,variant=key[0],noise=key[1],label=key[2],sigma=key[3],readout=key[4],split=key[5],rmse=rmse,pattern_rmse=offset_rmse,face_order_agreement=order['faces'],house_order_agreement=order['houses'],faces=[g['faces',c]['mean_errors'] for c in C],houses=[g['houses',c]['mean_errors'] for c in C],cells=g))
for v in ['lp','cnn']:
 for source,readout in [('coarse','legacy_floor'),('fine_noisy','stable_unfloored'),('fine_zero','stable_unfloored')]:
  for split in (['development'] if source=='coarse' else ['development','validation']):
   a=sorted([r for r in out if r['variant']==v and r['source']==source and r['readout']==readout and r['split']==split],key=lambda r:r['rmse'])
   print('\n',v,source,readout,split)
   for r in a[:4]:print({k:r[k] for k in ['noise','label','sigma','rmse','pattern_rmse','face_order_agreement','house_order_agreement','faces','houses']})
 for metric in ['rmse','pattern_rmse']:
  a=sorted([r for r in out if r['variant']==v and r['source'].startswith('fine') and r['split']=='development'],key=lambda r:r[metric])
  print('COMBINED DEV',v,metric,[{k:r[k] for k in ['source','label','sigma','rmse','pattern_rmse','face_order_agreement','house_order_agreement']} for r in a[:3]])
