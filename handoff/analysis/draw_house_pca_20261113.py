import sys,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from house_pca_retrieval_config_20261113 import B,P,O,sha,load
sys.path.insert(0,str(P))
from design import plan
from engine import Experiment
import torch
torch.set_num_threads(1);s=load();p=plan('faces_houses',101)
ids=sorted(p['groups']['houses']['studied']);five=ids[:5]
assert len(ids)==20 and len(set(ids))==20
colors=plt.get_cmap('tab20').colors
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'figure.facecolor':'#f8fafc','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
allmeta={};projections={}
def fit_transform(x):
 mean=x.mean(0);center=x-mean;u,sv,v=np.linalg.svd(center,full_matrices=False)
 comp=v[:2].copy()
 for row in comp:
  if row[np.argmax(np.abs(row))]<0:row*=-1
 z=center@comp.T;ev=(sv[:2]**2)/(sv**2).sum()
 assert np.allclose(comp@comp.T,np.eye(2),atol=1e-10)
 return mean,comp,z,ev
for representation in ['sigmoid','binary']:
 projections[representation]={}
 for m in s['models']:
  v=m['record']['variant'];e=Experiment(m['cache'],'faces_houses');arrays={}
  for ori in ['U','I']:
   if representation=='sigmoid':a=e.probs['houses',ori][ids].numpy().astype(np.float64)
   else:a=torch.stack([e.encode(101,'probe','houses',r,ori,32,0.) for r in ids]).numpy().astype(np.float64)
   assert a.shape==(20,32,256);arrays[ori]=a
  stacked=np.concatenate([arrays['U'].reshape(-1,256),arrays['I'].reshape(-1,256)])
  mean,comp,z,ev=fit_transform(stacked);parts={'U':z[:640].reshape(20,32,2),'I':z[640:].reshape(20,32,2)}
  lo=z.min(0);hi=z.max(0);pad=(hi-lo)*.08
  projections[representation][v]=(parts,ev,lo-pad,hi+pad)
  np.savez_compressed(O/(v+'-'+representation+'-PCA.npz'),mean=mean,components=comp,explained_variance_ratio=ev,house_rows=ids,U=parts['U'],I=parts['I'])
  allmeta[v,representation]=dict(explained_variance_ratio=ev.tolist(),pc1_pc2_total=float(ev.sum()))
 for subset,n in [(ids,20),(five,5)]:
  fig,axs=plt.subplots(2,2,figsize=(14,10),layout=None)
  fig.suptitle('%d houses · %s fixation representations'%(n,'continuous sigmoid' if representation=='sigmoid' else 'binary KDE'),fontsize=20,fontweight='bold',x=.08,ha='left',y=.985,color='#0f172a')
  fig.text(.08,.943,'32 fixation vectors per house · color = house identity · large circle = house centroid',color='#475569',fontsize=11)
  for ri,v in enumerate(['lp','cnn']):
   parts,ev,lo,hi=projections[representation][v]
   for ci,ori in enumerate(['U','I']):
    ax=axs[ri,ci]
    for idx,h in enumerate(ids):
     if h not in subset:continue
     coords=parts[ori][idx];color=colors[idx]
     ax.scatter(coords[:,0],coords[:,1],s=18 if n==20 else 27,c=[color],alpha=.60,edgecolors='none',rasterized=True)
     centroid=coords.mean(0);ax.scatter(*centroid,s=85 if n==20 else 105,c=[color],edgecolors='#1e293b',linewidths=.7,zorder=4)
     ax.annotate('H%02d'%(h+1),centroid,xytext=(4,4),textcoords='offset points',fontsize=7 if n==20 else 9,color='#0f172a',zorder=5)
    ax.set_title(v.upper()+' · '+('Upright' if ori=='U' else 'Inverted'),loc='left',fontweight='bold',color='#0f172a')
    ax.set_xlabel('PC1 (%.1f%% variance)'%(100*ev[0]));ax.set_ylabel('PC2 (%.1f%%)'%(100*ev[1]))
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15,zorder=0)
  handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=colors[ids.index(h)],markeredgecolor='none',markersize=7,label='H%02d'%(h+1)) for h in subset]
  fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035 if n==20 else .055),ncol=10 if n==20 else 5,frameon=False,columnspacing=1.35,handletextpad=.35)
  foot='Centered PCA fit jointly to upright + inverted vectors of the same 20 houses, separately per model; no feature standardization.'
  if n==5:foot+='\nFive-house subset uses the same axes and PCA basis as the 20-house plot.'
  fig.text(.08,.012 if n==5 else .015,foot,fontsize=8,color='#475569',va='bottom')
  fig.subplots_adjust(left=.07,right=.98,top=.90,bottom=.13,wspace=.20,hspace=.28)
  name='houses-%d-%s-pca'%(n,representation)
  fig.savefig(O/(name+'.png'),dpi=180,facecolor=fig.get_facecolor());fig.savefig(O/(name+'.svg'),facecolor=fig.get_facecolor());plt.close(fig)
# A separate category overview keeps face/house competition visible.
fig,axs=plt.subplots(1,2,figsize=(13,6))
catmeta={}
for ax,m in zip(axs,s['models']):
 v=m['record']['variant'];e=Experiment(m['cache'],'faces_houses');samples=[];labels=[]
 for cat in ['faces','houses']:
  for ori in ['U','I']:
   a=e.probs[cat,ori][sorted(p['groups'][cat]['studied'])].numpy().reshape(-1,256).astype(np.float64);samples.append(a);labels.extend([(cat,ori)]*len(a))
 x=np.concatenate(samples);mean,comp,z,ev=fit_transform(x);labels=np.array(labels)
 for cat,color in [('faces','#2563eb'),('houses','#ea580c')]:
  for ori,marker in [('U','o'),('I','^')]:
   mask=(labels[:,0]==cat)&(labels[:,1]==ori)
   ax.scatter(z[mask,0],z[mask,1],s=10,alpha=.25,c=color,marker=marker,label=cat+' '+ori,rasterized=True)
 ax.set_title(v.upper(),loc='left',fontweight='bold');ax.set_xlabel('PC1 (%.1f%%)'%(100*ev[0]));ax.set_ylabel('PC2 (%.1f%%)'%(100*ev[1]));ax.legend(frameon=False);ax.grid(alpha=.15)
 catmeta[v]=dict(explained_variance_ratio=ev.tolist())
fig.suptitle('Do face and house representations overlap?',fontsize=18,fontweight='bold');fig.tight_layout(rect=(0,.02,1,.93));fig.savefig(O/'faces-houses-category-pca.png',dpi=180);fig.savefig(O/'faces-houses-category-pca.svg');plt.close(fig)
meta=dict(source_script_sha256=sha(Path(__file__)),source_spec_sha256=sha(P/'SPEC.json'),plot_selection_seed=101,house_rows_zero_based=ids,five_house_rows_zero_based=five,house_labels=['H%02d'%(h+1) for h in ids],selection='All20studied houses in frozen seed101 plan;five smallest bank-row labels,chosen without representation screening',probability_pca='Centered,unstandardized256D sigmoid representations;sharedU/I fit per model',binary_pca='Same coordinates,probabilities and frozen Bernoulli uniforms from probe phase seed101 at zero XOR;separate PCA basis from probability plot',five_house_projection='Exact same basis and axis limits as20-house projection per model/representation',explained_variance={v+'_'+r:vals for (v,r),vals in allmeta.items()},category_pca=catmeta,notes=['Each point is one fixation,not one whole-house embedding.','PCA2D discards variance;apparent overlap/separation does not establish256D retrieval errors.','Binary draws are one fixed visualization seed;probabilities do not include Bernoulli noise.','House labels are1-based row positions in the frozen Yin64-house bank,not training-class IDs.'])
(O/'PCA_METADATA.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta['explained_variance'],indent=2))
