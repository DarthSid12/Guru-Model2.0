"""Plot recorded histories plus explicitly labelled checkpoint evaluations."""
from pathlib import Path
import csv
import json
import math
import hashlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_vgg16bn_report import MODELS
PAPER=ROOT/'paper'
FIG=PAPER/'figures/vgg16bn_training'
DATA=PAPER/'vgg16bn_report_data'
AUDIT=ROOT/'runs/vgg16bn_training_curve_audit_20260928'
CATS={'faces_vgg2k':'VGGFace2 · 2,048 identities','faces_vgg':'VGGFace2 · old face store',
      'faces':'CelebA','faces_rfwW':'RFW · white','faces_rfwO':'RFW · other',
      'objects':'Objects','houses_zubud137_41':'ZuBuD · building identities',
      'houses':'Generic houses','all_faces':'All face sources pooled'}
COLORS={'train':'#0072B2','valid':'#009E73','test':'#D55E00'}
BOUNDARIES=np.cumsum([2,2,4,4,8,8,16,16,32,32])


def csv_rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def values(rows,k):return np.array([float(r[k]) if r.get(k) else np.nan for r in rows])*100

def save(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def setup(ax):
    ax.set_xlim(0,126);ax.set_ylim(0,103)
    ax.set_xticks([1,20,40,60,80,100,124]);ax.set_yticks([0,20,40,60,80,100]);ax.grid(axis='y',alpha=.15)
    for e in BOUNDARIES[:-1]:ax.axvline(e+.5,color='#aaa',ls=':',lw=.7,zorder=0)
    ax.set_xlabel('Training epoch');ax.set_ylabel('Accuracy (%)')


def overall(ax,h):
    e=[int(r['epoch']) for r in h]
    for name,col,label in [('train_acc','train','Train · recorded during updates'),('valid_acc','valid','Validation · upright'),('test_acc','test','Test split · inverted')]:
        ax.plot(e,values(h,name),color=COLORS[col],lw=1.6,label=label)
    setup(ax);ax.set_title('Overall · recorded at every epoch',fontsize=11)


def category(ax,h,recon,cat,pooled=None):
    e=[int(r['epoch']) for r in h]
    train=[(r['epoch'],r['train_eval'][cat]['accuracy_pct']) for r in recon if cat in r['train_eval']]
    valid=values(h,'valid_acc_'+cat) if pooled is None else pooled['valid']
    test=values(h,'test_acc_'+cat) if pooled is None else pooled['test']
    ax.plot(e,valid,color=COLORS['valid'],lw=1.6,label='Validation · upright (recorded)')
    if np.isfinite(test).any():ax.plot(e,test,color=COLORS['test'],lw=1.6,label='Test split · inverted (recorded)')
    else:
        inv=[(r['epoch'],r['inverted_validation'][cat]['accuracy_pct']) for r in recon if cat in r['inverted_validation']]
        if inv:ax.plot(*zip(*inv),color=COLORS['test'],lw=1.5,ls='--',marker='s',ms=4,label='Validation inverted · checkpoint evaluation')
        ax.text(.02,.04,'No original test split',transform=ax.transAxes,fontsize=8,color='#8e4b27')
    if train:ax.plot(*zip(*train),color=COLORS['train'],lw=1.5,ls='--',marker='o',ms=4,label='Train sample · checkpoint evaluation')
    setup(ax);ax.set_title(CATS[cat],fontsize=11)


def pooled_faces(run,cfg,h,recon,labels):
    facecats=[c for c in cfg['categories'] if c.startswith('faces')]
    # Recorded category accuracies aggregate image counts. Reconstruct their
    # pooled mean using active images in each saved stage and each split.
    weights={}
    for stage in range(1,11):
        active=set(json.loads((run/f'stage{stage}_active_ids.json').read_text())['active_ids'])
        for split in ('valid','test'):
            weights[(stage,split)]={}
            for cat in facecats:
                m=json.loads((ROOT/'fixation_data'/cat/split/'meta.json').read_text());counts=Counter(m['labels'])
                n=sum(count for ci,count in counts.items() if labels.get(cat+'/'+m['classes'][ci]) in active)
                weights[(stage,split)][cat]=n
    pooled={}
    for split in ('valid','test'):
        a=[]
        for row in h:
            w=weights[(int(row['stage']),split)];terms=[(float(row[f'{split}_acc_{c}']),n) for c,n in w.items() if n]
            a.append(100*sum(v*n for v,n in terms)/sum(n for _,n in terms))
        pooled[split]=np.array(a)
    for r in recon:
        included=[v for c,v in r['train_eval'].items() if c in facecats]
        correct=sum(v['correct'] for v in included);n=sum(v['n_photos'] for v in included)
        r['train_eval']['all_faces']=dict(correct=correct,n_photos=n,accuracy_pct=100*correct/n)
    return pooled


def main():
    FIG.mkdir(parents=True,exist_ok=True);DATA.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    model_data={};endpoints=[];records=[];sources={}
    for key,meta in MODELS.items():
        run=next((ROOT/'runs').glob('*_'+meta['tag']));hist=next(run.glob('training_history*.csv'))
        h=csv_rows(hist);cfg=json.loads((run/'config.json').read_text());labels=json.loads((run/'label_map.json').read_text())
        assert len(h)==124 and [int(r['epoch']) for r in h]==list(range(1,125))
        assert json.loads((AUDIT/key/'DONE.json').read_text())['stages']==10
        recon=[json.loads((AUDIT/key/f'stage{i}.json').read_text()) for i in range(1,11)]
        assert [r['epoch'] for r in recon]==list(BOUNDARIES)
        pooled=pooled_faces(run,cfg,h,recon,labels) if key=='mixed' else None
        cats=cfg['categories']+(['all_faces'] if pooled is not None else [])
        model_data[key]=dict(h=h,cfg=cfg,run=run,recon=recon,cats=cats,pooled=pooled)
        sources[str(hist.relative_to(ROOT))]=hashlib.sha256(hist.read_bytes()).hexdigest()
        for row in h:
            for kind in ('train_acc','valid_acc','test_acc'):
                records.append(dict(model=key,model_name=meta['name'],category='Overall',epoch=int(row['epoch']),series=kind,measurement='recorded',accuracy_pct=100*float(row[kind]),n_photos='',source=str(hist.relative_to(ROOT))))
            for cat in cfg['categories']:
                for kind in ('valid','test'):
                    v=row.get(f'{kind}_acc_{cat}')
                    if v:records.append(dict(model=key,model_name=meta['name'],category=cat,epoch=int(row['epoch']),series=kind,measurement='recorded',accuracy_pct=100*float(v),n_photos='',source=str(hist.relative_to(ROOT))))
            if pooled:
                for kind in ('valid','test'):records.append(dict(model=key,model_name=meta['name'],category='all_faces',epoch=int(row['epoch']),series=kind,measurement='pooled from recorded category means and exact active-image counts',accuracy_pct=pooled[kind][int(row['epoch'])-1],n_photos='',source=str(hist.relative_to(ROOT))))
        for r in recon:
            for kind in ('train_eval','inverted_validation'):
                for cat,stats in r[kind].items():records.append(dict(model=key,model_name=meta['name'],category=cat,epoch=r['epoch'],series=kind,measurement='checkpoint evaluation',accuracy_pct=stats['accuracy_pct'],n_photos=stats['n_photos'],source=str((AUDIT/key/f"stage{r['stage']}.json").relative_to(ROOT))))
        endpoints.append(dict(model=meta['name'],train_pct=100*float(h[-1]['train_acc']),validation_pct=100*float(h[-1]['valid_acc']),inverted_test_pct=100*float(h[-1]['test_acc'])))
        fig,ax=plt.subplots(figsize=(9,4.8));overall(ax,h);ax.legend(loc='lower right',fontsize=9,frameon=False);fig.suptitle(meta['name'],fontsize=15);fig.tight_layout();save(fig,key+'_overall')
        for cat in cats:
            fig,ax=plt.subplots(figsize=(9,4.8));category(ax,h,recon,cat,pooled if cat=='all_faces' else None);ax.legend(loc='lower right',fontsize=8,frameon=False)
            fig.suptitle(meta['name']+' · '+CATS[cat],fontsize=14);ax.set_title('Solid lines: recorded each epoch · dashed points: saved checkpoints',fontsize=9)
            fig.tight_layout();save(fig,key+'_'+cat)
        n=1+len(cats);cols=3 if key=='mixed' else 2;rows=math.ceil(n/cols)
        fig,axes=plt.subplots(rows,cols,figsize=(15 if key=='mixed' else 12,rows*3.35))
        overall(axes.flat[0],h)
        for ax,cat in zip(list(axes.flat)[1:],cats):category(ax,h,recon,cat,pooled if cat=='all_faces' else None)
        for ax in list(axes.flat)[n:]:ax.axis('off')
        handles,labels0=axes.flat[0].get_legend_handles_labels()
        from matplotlib.lines import Line2D
        handles.extend([Line2D([0],[0],color=COLORS['train'],ls='--',marker='o',label='Train sample · checkpoint evaluation'),Line2D([0],[0],color=COLORS['test'],ls='--',marker='s',label='ZuBuD inverted validation · checkpoint evaluation')])
        fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.5,.005))
        fig.suptitle(meta['name']+' · overall and category learning curves',fontsize=16,y=.995)
        fig.tight_layout(rect=(0,.085,1,.97));save(fig,key+'_dashboard')
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.2),sharex=True,sharey=True)
    for ax,(key,m) in zip(axes.flat,MODELS.items()):overall(ax,model_data[key]['h']);ax.set_title(m['name'],fontsize=12)
    handles,lab=axes.flat[0].get_legend_handles_labels();fig.legend(handles,lab,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Recorded overall training, validation, and inverted test accuracy',fontsize=16)
    fig.tight_layout(rect=(0,.055,1,.95));save(fig,'all_models_overall')
    with (DATA/'training_curves.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (DATA/'training_source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    lines=['# VGG16-BN training curves','', 'Four models · 124 epochs · overall and individual training categories. [Behavioral report](vgg16bn_results_report.md).', '',
           '## What each curve means','',
           '- **Overall training (solid blue):** the original per-epoch accuracy on augmented fixation crops while parameters were being updated. It was recorded and has not been reconstructed.',
           '- **Validation (solid green):** recorded upright image accuracy, combining logits across 16 fixation crops.',
           '- **Test (solid orange):** recorded image accuracy on the test split after **180° inversion**, again combining 16 fixations. The inversion is verified in the transform and training code.',
           '- **Category training (dashed blue dots):** a new evaluation of the saved stage-end model on a deterministic sample of up to 16 training photos per active class. Generic houses use all stage-eligible photos. This uses the first 16 saved fixation crops, upright presentation, evaluation mode, and no augmentation. It is **not the missing historical per-category training statistic**.',
           '- **ZuBuD inverted validation (dashed orange squares):** a new inverted evaluation of its one validation view per active building. ZuBuD has **no test split**, so this is not labelled as a recorded test curve.', '',
           'The saved checkpoints are at epochs **2, 4, 8, 12, 20, 28, 44, 60, 92, and 124**. Dotted vertical lines mark curriculum changes. Dashed segments connect only the measured checkpoint endpoints; intermediate values are not observed. A category appears when its classes enter the curriculum.', '',
           'Training and evaluation curves use different units and sampling: the overall training curve scores augmented individual crops during learning, whereas validation/test score images after combining crops. Reconstructed category training curves also use a sample of the training images. Their gaps should not be treated as a like-for-like generalization estimate.', '',
           'These are accuracies on the supervised training task. The training categories differ from the held-out behavioral stores: ZuBuD building identities are `houses_zubud137_41`, generic houses are one classifier class, and the Yin house set is `houses_yin64`. The mixed model’s RFW training stores are also distinct from Yin’s RFW WM64 evaluation set. High supervised house accuracy does not imply high held-out Yin memory accuracy.', '',
           '## Test inversion and split audit','',
           '`train.py` sends the test loader through `OnTheFlyTransform("test")`; that transform selects `Rotate(invert=True)`, which is exactly 180°. It reads a separate test store: the curve is not generally validation accuracy with its orientation changed. VGG2k uses the next 12 photos per identity for validation and the last 12 for test; old VGG/CelebA and objects also have different stored pictures. RFW-W, RFW-O and generic houses have matching validation/test image arrays, but **different fixation coordinates** (all arrays checked in the [split audit](vgg16bn_report_data/split_audit.json)). ZuBuD buildings have only train and validation stores.', '',
           'Overall validation includes the ZuBuD category, while overall test omits it. Also, the trainer records overall evaluation as the mean of batch accuracies, while per-category evaluation uses image-level correct/total. Therefore neither the overall valid–test gap nor a simple average of category curves is an exact isolated inversion effect. The Yin behavioral tasks above compare orientations on the same images.', '',
           'The `summary.json` per-category values were computed after reloading the **best** checkpoint. These plots use the per-epoch **history CSV** to avoid substituting best-checkpoint values for the final epoch.', '',
           'Code evidence: [training evaluation and history](../train.py#L1064), [evaluation aggregation](../train.py#L276), [test transform](../salience_trans.py#L270), [180° rotation](../trans.py#L47), [missing-split handling](../datasets.py#L258).', '',
           '## Overall curves','', '![All four models: recorded overall accuracy](figures/vgg16bn_training/all_models_overall.png)', '',
           '| Model | Final recorded train | Final upright validation | Final inverted test |','| --- | ---: | ---: | ---: |']
    for row in endpoints:lines.append(f"| {row['model']} | {row['train_pct']:.2f}% | {row['validation_pct']:.2f}% | {row['inverted_test_pct']:.2f}% |")
    for key,meta in MODELS.items():
        d=model_data[key];lines += ['',f"## {meta['name']}",'',f"![{meta['name']} category dashboard](figures/vgg16bn_training/{key}_dashboard.png)",'',
                              f"[Dashboard PDF](figures/vgg16bn_training/{key}_dashboard.pdf) · [Overall PNG](figures/vgg16bn_training/{key}_overall.png) · [Original history](../{next(d['run'].glob('training_history*.csv')).relative_to(ROOT)})",'',
                              '<details><summary>Individual category images and checkpoint sample sizes</summary>','',
                              '| Category | Individual PNG | PDF | Photos in final reconstructed train evaluation |','| --- | --- | --- | ---: |']
        for cat in d['cats']:
            count=d['recon'][-1]['train_eval'][cat]['n_photos']
            lines.append(f"| {CATS[cat]} | [Image](figures/vgg16bn_training/{key}_{cat}.png) | [PDF](figures/vgg16bn_training/{key}_{cat}.pdf) | {count:,} |")
        lines += ['', '</details>']
    lines += ['', '## Reproducibility','',
              '[All curve coordinates](vgg16bn_report_data/training_curves.csv) · [History hashes](vgg16bn_report_data/training_source_manifest.json) · [Stage evaluation records and photo samples](../runs/vgg16bn_training_curve_audit_20260928/) · [Evaluator](../scripts/evaluate_vgg_training_checkpoints.py) · [Plot generator](../scripts/plot_vgg16bn_training_curves.py)', '',
              'The deterministic sample seed is 20260928. Training samples respect each saved active-class list and the stage-specific generic-house image cap. Stage weights are loaded strictly; no weights are updated. Photo lists, correct/total counts, checkpoint hashes, and evaluation-code hashes are saved. The same VGG2k class/photo samples are used across its three architecture variants. All stages completed for all four models.', '']
    (PAPER/'vgg16bn_training_curves.md').write_text('\n'.join(lines))
    report=PAPER/'vgg16bn_results_report.md';text=report.read_text();a='<!-- TRAINING_CURVES_START -->';b='<!-- TRAINING_CURVES_END -->'
    block='\n\n**The test curve is confirmed to use 180° inverted images.** The overall plots use the original 124-epoch records. Per-category training was never logged; dashed training points evaluate the saved stage checkpoints instead. ZuBuD has no test split, so its dashed orange points evaluate inverted validation images.\n\n![Recorded overall curves](figures/vgg16bn_training/all_models_overall.png)\n\n[Open the training-curve report](vgg16bn_training_curves.md) for all category dashboards, individual PNG/PDF images, sampling details, and the validation/test split audit. The plots use history CSVs, not the best-checkpoint category statistics in `summary.json`.\n\n'
    for key,meta in MODELS.items():block+=f"<details><summary>{meta['name']}: overall and all categories</summary>\n\n![{meta['name']} learning curves](figures/vgg16bn_training/{key}_dashboard.png)\n\n[Individual category figures and interpretation](vgg16bn_training_curves.md)\n\n</details>\n\n"
    report.write_text(text.split(a,1)[0]+a+block+b+text.split(b,1)[1])
    print(f'Plotted {len(list(FIG.glob("*.png")))} PNGs and matching PDFs; exported {len(records)} curve coordinates.')

if __name__=='__main__':main()
