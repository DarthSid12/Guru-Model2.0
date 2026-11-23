import sys,json,hashlib,html
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
B=Path('/Users/darthsid/Documents/ChatGPT/GURU');R=B/'output/house-pca-retrieval-20261113';O=R/'test-probes';O.mkdir(exist_ok=False)
P=B/'operations/local-kde-alpha-zero-20261112-v2';sys.path.insert(0,str(P))
from engine import Experiment
from design import plan
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((P/'SPEC.json').read_text());active=json.loads((B/'operations/ACTIVE_YIN_READOUT.json').read_text());assert active['readout']=='original_KDE' and active['sigma']==2
meta=json.loads((R/'PCA_METADATA.json').read_text());seed=meta['plot_selection_seed'];p=plan('faces_houses',seed);ids=meta['house_rows_zero_based'];five=meta['five_house_rows_zero_based']
assert ids==sorted(p['groups']['houses']['studied'])
# First fixed test pair whose old photo belongs to the existing five-house subset.
ti,t=next((i,t) for i,t in enumerate(p['trials']) if t['category']=='houses' and t['old'] in five)
old,new=t['old'],t['new'];assert old in ids and new not in ids and new in p['groups']['houses']['novel']
torch.set_num_threads(1);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
colors=plt.get_cmap('tab20').colors;allmeta={};score_records=[];plotinfo=[]
for mode in ['sigmoid','binary-zero','binary-facecal']:
 data={}
 for m in s['models']:
  v=m['record']['variant'];cache=Path(m['cache']);done=json.loads((cache/'DONE.json').read_text())
  for name,h in done['files'].items():
   if name.startswith(('houses_dal64','faces_cfdWM64')):assert sha(cache/name)==h
  e=Experiment(cache,'faces_houses');rep='sigmoid' if mode=='sigmoid' else 'binary';basis=np.load(R/(v+'-'+rep+'-PCA.npz'))
  mean,comp=basis['mean'],basis['components'];ev=basis['explained_variance_ratio']
  rate=0. if mode!='binary-facecal' else active['single_noise_context_fits'][v]['faces_houses']['noise_rate']
  views={};save={}
  for ori in ['U','I']:
   if mode=='sigmoid':
    background=basis[ori];oldx=e.probs['houses',ori][old,:32].numpy().astype(np.float64);newx=e.probs['houses',ori][new,:32].numpy().astype(np.float64)
   else:
    study=torch.stack([e.encode(seed,'study','houses',row,ori,10,rate) for row in ids]).numpy().astype(np.float64)
    background=(study-mean)@comp.T
    oldx=e.encode(seed,'probe','houses',old,ori,32,rate).numpy().astype(np.float64)
    newx=e.encode(seed,'probe','houses',new,ori,32,rate).numpy().astype(np.float64)
   oldz=(oldx-mean)@comp.T;newz=(newx-mean)@comp.T
   if mode=='sigmoid':assert np.allclose(oldz,background[ids.index(old)],atol=1e-12)
   views[ori]=dict(background=background,old=oldz,novel=newz)
   save.update({ori+'_background':background,ori+'_old_test':oldz,ori+'_novel_test':newz})
  # Preserve the original reference extent and expand only for added points,uniformly for both views.
  original=np.concatenate([basis['U'].reshape(-1,2),basis['I'].reshape(-1,2)]);origlo=original.min(0);orighi=original.max(0);origpad=(orighi-origlo)*.08
  combined=np.concatenate([original]+[a.reshape(-1,2) for view in views.values() for a in view.values()])
  lo=np.minimum(origlo-origpad,combined.min(0)-.15);hi=np.maximum(orighi+origpad,combined.max(0)+.15)
  data[v]=dict(views=views,ev=ev,lo=lo,hi=hi,rate=rate)
  np.savez_compressed(O/(v+'-'+mode+'-test-projections.npz'),mean=mean,components=comp,house_rows=ids,old_row=old,novel_row=new,axis_lo=lo,axis_hi=hi,**save)
  allmeta[v+'_'+mode]=dict(noise_rate=rate,pca_basis_file=str(R/(v+'-'+rep+'-PCA.npz')),pca_basis_sha256=sha(R/(v+'-'+rep+'-PCA.npz')),explained_variance_ratio=ev.tolist(),axis_limits=[lo.tolist(),hi.tolist()],background_fixations_per_house=32 if mode=='sigmoid' else 10,test_fixations_each=32)
  if mode!='sigmoid':
   for condition in ['UU','UI','IU','II']:
    result,details=e.measure(seed,condition,rate,rate,details=True);scores=details['scores'][ti]
    score_records.append(dict(variant=v,noise_rate=rate,setting=mode,condition=condition,seed=seed,trial_index_zero_based=ti,old_house='H%02d'%(old+1),novel_house='H%02d'%(new+1),old_log_familiarity=scores[0],novel_log_familiarity=scores[1],old_chosen=scores[0]>scores[1],sigma=2))
 for subset,n in [(ids,20),(five,5)]:
  fig,axs=plt.subplots(2,2,figsize=(15,11));fig.patch.set_facecolor('#f8fafc')
  fig.suptitle('%d houses + old/new Yin test fixations'%n,x=.07,ha='left',y=.987,va='top',fontsize=21,fontweight='bold',color='#0f172a')
  mode_label={'sigmoid':'Continuous sigmoid vectors · before binary sampling','binary-zero':'Actual study/test binary codes · XOR noise 0','binary-facecal':'Actual study/test binary codes · LP noise .38 / CNN noise .36'}[mode]
  fig.text(.07,.942,mode_label+' · fixed seed101 · original PCA basis',fontsize=11,color='#475569')
  testhandles=[Line2D([0],[0],marker='^',linestyle='none',color='#111827',markeredgecolor='white',markersize=9,label='OLD test H%02d · 32 fixations'%(old+1)),Line2D([0],[0],marker='x',linestyle='none',color='#d0008f',markersize=9,markeredgewidth=1.7,label='NOVEL test H%02d · 32 fixations'%(new+1))]
  fig.legend(handles=testhandles,loc='upper center',bbox_to_anchor=(.52,.92),ncol=2,frameon=False,fontsize=11,columnspacing=3)
  for ri,v in enumerate(['lp','cnn']):
   d=data[v]
   for ci,ori in enumerate(['U','I']):
    ax=axs[ri,ci];view=d['views'][ori]
    for idx,row in enumerate(ids):
     if row not in subset:continue
     z=view['background'][idx];color=colors[idx]
     ax.scatter(z[:,0],z[:,1],s=18 if n==20 else 25,c=[color],alpha=.27,edgecolors='none',rasterized=True)
     centre=z.mean(0);ax.scatter(*centre,s=55,c=[color],alpha=.65,edgecolors='#64748b',linewidths=.5,zorder=3)
     if n==5:ax.annotate('H%02d'%(row+1),centre,xytext=(4,4),textcoords='offset points',fontsize=8,color='#64748b')
    oz,nz=view['old'],view['novel']
    ax.scatter(oz[:,0],oz[:,1],s=48,marker='^',c='#111827',alpha=.85,edgecolors='white',linewidths=.35,zorder=5,rasterized=True)
    ax.scatter(nz[:,0],nz[:,1],s=60,marker='x',c='#d0008f',alpha=.90,linewidths=1.45,zorder=6,rasterized=True)
    oc,nc=oz.mean(0),nz.mean(0)
    ax.scatter(*oc,s=150,marker='^',c='#111827',edgecolors='white',linewidths=.9,zorder=7)
    ax.scatter(*nc,s=160,marker='X',c='#d0008f',edgecolors='white',linewidths=.9,zorder=8)
    ax.annotate('OLD H%02d'%(old+1),oc,xytext=(7,10),textcoords='offset points',fontsize=9,fontweight='bold',color='#111827',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1),zorder=10)
    ax.annotate('NOVEL H%02d'%(new+1),nc,xytext=(7,-15),textcoords='offset points',fontsize=9,fontweight='bold',color='#b00075',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1),zorder=10)
    ax.set_title(v.upper()+' · '+('Upright (UU)' if ori=='U' else 'Inverted (II)'),loc='left',fontweight='bold')
    ax.set_xlabel('PC1 (%.1f%%)'%(100*d['ev'][0]));ax.set_ylabel('PC2 (%.1f%%)'%(100*d['ev'][1]));ax.set_xlim(d['lo'][0],d['hi'][0]);ax.set_ylim(d['lo'][1],d['hi'][1]);ax.grid(alpha=.15);ax.set_aspect('equal',adjustable='box')
  refs=[Line2D([0],[0],marker='o',linestyle='none',color=colors[ids.index(row)],markersize=6,label='H%02d'%(row+1)) for row in subset]
  fig.legend(handles=refs,loc='lower center',bbox_to_anchor=(.5,.048),ncol=10 if n==20 else 5,frameon=False,columnspacing=1.2,handletextpad=.35)
  foot='PCA coordinates unchanged; old H04 is already studied, novel H13 is absent from all40study memories. Five-house view uses the same limits.'
  if mode=='sigmoid':foot+='\nRaw old-test vectors coincide with H04reference vectors;binary study/test draws are shown in the other views.'
  else:foot+='\nBackground:actual10study fixations/house. Test:32fixations each. PCA was not refitted to include either test cloud.'
  fig.text(.07,.012,foot,fontsize=8,color='#475569',va='bottom')
  fig.subplots_adjust(left=.07,right=.98,top=.865,bottom=.16,wspace=.2,hspace=.28)
  name='houses-%d-%s-with-test'%(n,mode);fig.savefig(O/(name+'.png'),dpi=180,facecolor=fig.get_facecolor());fig.savefig(O/(name+'.svg'),facecolor=fig.get_facecolor());plt.close(fig)
  plotinfo.append(dict(size=n,mode=mode,png=name+'.png',svg=name+'.svg'))
manifest=dict(source_script_sha256=sha(Path(__file__)),active_readout_snapshot=active,parent_pca_metadata_sha256=sha(R/'PCA_METADATA.json'),selection_seed=seed,design='faces_houses',selection='First existing fixed seed101house testpair whose studied image belongs to the already-selected five-house subset;no outcome screening',trial_index_zero_based=ti,old_house_row_zero_based=old,novel_house_row_zero_based=new,old_house_label='H%02d'%(old+1),novel_house_label='H%02d'%(new+1),house_rows_zero_based=ids,five_house_rows_zero_based=five,views=allmeta,plots=plotinfo,notes=['Raw continuous old-test points equal that house reference points because they reuse the same cached image/fixation features.','Binary background uses actual10study-phase samples;testpoints use32probe-phase samples at the labelled noise.','Allnewpoints are projected through the original mean/components;no PCA refitting.','Axis limits include added points and are held the same across20/5house versions andU/Ifor eachmodel/mode.','No models,data,noise fits or active KDEsettings changed.'])
(O/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');(O/'TRIAL_SCORES.json').write_text(json.dumps(score_records,indent=2)+'\n')
print(json.dumps(dict(old_house=manifest['old_house_label'],novel_house=manifest['novel_house_label'],plots=len(plotinfo),trial_scores=score_records),indent=2))
