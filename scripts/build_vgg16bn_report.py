"""Compile saved VGG16-BN results, verify seed records, and render comparisons.

No behavioral simulations are run. two_shared_ui is assembled from existing
matching cells, just as in the original noise-matrix report.
"""
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import statistics as st
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
DATA = PAPER / 'vgg16bn_report_data'
FIG = PAPER / 'figures/vgg16bn_comparison'
CONDITIONS = ('UU', 'II', 'UI', 'IU')
PROTOCOLS = ('single_uu', 'two_cross', 'two_shared_ui')
HUMAN = {'Faces': (96.29, 81.88, 84.13, 78.58),
         'Objects': (84.79, 83.96, 86.71, 82.75),
         'Houses': (90.71, 85.75, 88.08, 85.71)}
MODELS = {
    'noaa': dict(tag='r21vgg2k_vgg16_bn_s42', name='VGG2k / no AA', color='#0072B2',
                 description='VGG16-BN without antialiasing; 2,048 VGGFace2 training identities, plus objects and houses.'),
    'aa4': dict(tag='r21vgg2k_vgg16_bn_aa_s42', name='VGG2k / AA4', color='#D55E00',
                description='VGG16-BN with the 4-tap [1,3,3,1] antialias filter; the same VGG2k, object, and house diet.'),
    'bin5': dict(tag='r21vgg2k_vgg16_bn_aa5_s42', name='VGG2k / bin5', color='#009E73',
                 description='VGG16-BN with the 5-tap [1,4,6,4,1] antialias filter; the same VGG2k, object, and house diet.'),
    'mixed': dict(tag='r21dev_vgg16_bn_aa_s42', name='Mixed faces / AA4', color='#CC79A7',
                 description='VGG16-BN with the 4-tap antialias filter; the older mixed face diet (VGGFace2, CelebA, RFW), plus objects and houses.'),
}
FACES = {'RFW WM64':'faces_rfwWM64', 'CFD WM64':'faces_cfdWM64'}
SOURCE_MDS = [
 'paper/yin_experiments_20260925.md',
 'runs/yin_orientation_20260925/README.md',
 'runs/yin_single_noise_20260925/README.md',
 'runs/yin_noise_matrix_20260926/REPORT.md',
 'runs/yin_category_uu_20260926/REPORT.md',
 'runs/yin_noaa_two_cross_20260927/REPORT.md',
 'runs/yin_latest_vgg_20260927/REPORT.md',
 'runs/kanw_latest_vgg_20260927/summary.md',
 'paper/curriculum_results_summary.md', 'paper/yin_replication.tex',
]
SOURCES = set(SOURCE_MDS)


def read_csv(path):
    SOURCES.add(str(path.relative_to(ROOT)))
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def read_json(path):
    SOURCES.add(str(path.relative_to(ROOT)))
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def index(path, column):
    rows = read_csv(path)
    data = {(int(r['seed']), r[column] if column != 'study_test' else r['study']+r['test']):r for r in rows}
    assert len(data) == len(rows), f'Duplicate observations: {path}'
    return data


def group(key, protocol, calibration, category, scope, epoch, mapping, p1, p2=None, fit_status='within_sampled_range'):
    means, sds, samples, sources, cells, hashes = {}, {}, {}, {}, {}, set()
    for condition, (path, label, column) in mapping.items():
        idx = index(path, column)
        selected = [idx[(seed, label)] for seed in range(101,151)]
        assert len({int(r['seed']) for r in selected}) == 50
        assert {int(r['seed']) for r in idx.values() if (r[column] if column!='study_test' else r['study']+r['test']) == label} == set(range(101,151))
        vals = [float(r['accuracy_pct']) for r in selected]
        assert all(0 <= x <= 100 for x in vals)
        means[condition] = st.mean(vals); sds[condition] = st.stdev(vals)
        samples[condition] = vals
        sources[condition] = str(path.relative_to(ROOT)); cells[condition] = label
        hashes.update(r.get('checkpoint_sha256','') for r in selected)
    ref = dict(zip(CONDITIONS, HUMAN[category]))
    err = {c:means[c]-ref[c] for c in CONDITIONS}
    fit_cells = ['UU'] if protocol == 'single_uu' else ['UU','II']
    if scope == 'face_transfer' and category != 'Faces': fit_cells = []
    return dict(model=key, model_name=MODELS[key]['name'], model_tag=MODELS[key]['tag'],
                protocol=protocol, calibrated_on=calibration, category=category, scope=scope,
                epoch=epoch, n_seeds=50, p1=p1, p2=p2, calibration_status=fit_status,
                means=means, sd=sds, samples=samples, errors=err, human=ref,
                fit_cells=fit_cells, mae=st.mean(abs(v) for v in err.values()),
                common_prediction_mae=st.mean(abs(err[c]) for c in ('UI','IU')),
                iu_minus_ii=means['IU']-means['II'],
                human_order=means['UU'] > means['UI'] > means['II'] > means['IU'],
                sources=sources, source_cells=cells, checkpoint_hashes=sorted(hashes))


def collect():
    groups = []
    for key, meta in MODELS.items():
        old = key in ('aa4','mixed'); tag=meta['tag']
        single = ROOT / ('runs/yin_single_noise_20260925/results' if old else 'runs/yin_latest_vgg_20260927/single_noise') / tag
        matrix = ROOT / ('runs/yin_noise_matrix_20260926/results' if old else 'runs/yin_latest_vgg_20260927/transfer/results') / tag
        for face_label, face in FACES.items():
            sf = read_json(single / face / 'calibration.json')
            mf = read_json(matrix / face / 'manifest.json')
            assert sf['p'] == mf['p1']
            assert (single / face / 'DONE.json').exists()
            for cat, folder in [('Faces',face), ('Objects','objects'), ('Houses','houses_yin64')]:
                one = single / face / folder / 'results.csv'
                two = matrix / face / folder / 'results.csv'
                assert read_json(one.parent/'DONE.json')['rows']==200
                assert read_json(two.parent/'DONE.json')['rows']==250
                a=index(one,'condition'); b=index(two,'cell')
                for seed in range(101,151):
                    assert float(a[(seed,'UU')]['accuracy_pct'])==float(b[(seed,'UU')]['accuracy_pct'])
                    assert float(a[(seed,'UI')]['accuracy_pct'])==float(b[(seed,'UI_shared')]['accuracy_pct'])
                for protocol in PROTOCOLS:
                    if protocol=='single_uu': mapping={c:(one,c,'condition') for c in CONDITIONS}
                    else: mapping={c:(two,('UI_cross' if protocol=='two_cross' else 'UI_shared') if c=='UI' else c,'cell') for c in CONDITIONS}
                    groups.append(group(key,protocol,face_label,cat,'face_transfer',124,mapping,sf['p'],mf['p2'] if protocol!='single_uu' else None))
        if old:
            for cat, folder in [('Objects','objects'),('Houses','houses_yin64')]:
                path=ROOT/'runs/yin_category_uu_20260926/results'/tag/folder/'results.csv'
                fit=read_json(path.parent/'calibration.json')
                groups.append(group(key,'single_uu',cat,cat,'category_calibration',124,{c:(path,c,'condition') for c in CONDITIONS},fit['p'],fit_status=fit['status']))
            for face_label,face in FACES.items():
                path=ROOT/'runs/yin_orientation_20260925/results/latest'/tag/face/'results.csv'
                fit=read_json(path.parent/'calibration.json'); manifest=read_json(path.parent/'manifest.json')
                groups.append(group(key,'two_cross',face_label,'Faces','snapshot',manifest['epoch'],{c:(path,c,'study_test') for c in CONDITIONS},fit['U']['p'],fit['I']['p']))
    assert len(groups)==80
    return groups


def rankings(groups):
    ranks=[]
    for key in MODELS:
        for protocol in PROTOCOLS:
            for face in FACES:
                g={r['category']:r for r in groups if (r['model'],r['protocol'],r['calibrated_on'],r['scope'])==(key,protocol,face,'face_transfer')}
                ranks.append(dict(model=key,model_name=MODELS[key]['name'],protocol=protocol,calibrated_on=face,
                                  face_prediction_mae=g['Faces']['common_prediction_mae'],
                                  face_all_cell_mae=g['Faces']['mae'],objects_mae=g['Objects']['mae'],houses_mae=g['Houses']['mae'],
                                  balanced_prediction_mae=st.mean([g['Faces']['common_prediction_mae'],g['Objects']['mae'],g['Houses']['mae']]),
                                  all_cell_mae=st.mean(r['mae'] for r in g.values()),
                                  face_human_order=g['Faces']['human_order']))
    return sorted(ranks,key=lambda r:r['balanced_prediction_mae'])


def collect_kanw():
    rows=[]
    for key in ('noaa','aa4','bin5'):
        tag=MODELS[key]['tag']
        for store,label in [('rfw','RFW WM64'),('cfd','CFD WMk'),('setA','Set A')]:
            path=ROOT/'runs/kanw_latest_vgg_20260927'/store/f'results_{tag}.csv'
            raw=read_csv(path); idx={(int(r['seed']),r['test']):float(r['accuracy_pct']) for r in raw}
            assert len(idx)==40 and len(raw)==40
            u=[idx[(s,'Upright')] for s in range(101,121)];i=[idx[(s,'Inverted')] for s in range(101,121)];d=[a-b for a,b in zip(u,i)]
            manifest=read_json(path.parent/f'manifest_{tag}.json')
            row=dict(model=key,model_name=MODELS[key]['name'],protocol='Kanwisher matching',calibrated_on=label,category=label,
                     n_seeds=20,p=read_json(path.parent/f'noise_{tag}.json')['kanwisher'],epoch=124,
                     upright_mean=st.mean(u),upright_sem=st.stdev(u)/math.sqrt(20),
                     inverted_mean=st.mean(i),inverted_sem=st.stdev(i)/math.sqrt(20),
                     cost_mean=st.mean(d),cost_sem=st.stdev(d)/math.sqrt(20),
                     source=str(path.relative_to(ROOT)),checkpoint_sha256=manifest['checkpoint_sha256'])
            rows.append(row)
    return rows


def export(groups,ranks,kanw):
    wide=[];long=[];trials=[]
    for r in groups:
        base={k:r[k] for k in ('model','model_name','model_tag','scope','epoch','protocol','calibrated_on','category','p1','p2','n_seeds','calibration_status')}
        wide.append(dict(base,**{f'{c}_mean_pct':r['means'][c] for c in CONDITIONS},**{f'{c}_sd_pct':r['sd'][c] for c in CONDITIONS},mae_pp=r['mae'],face_UI_IU_mae_pp=r['common_prediction_mae'],IU_minus_II_pp=r['iu_minus_ii']))
        for c in CONDITIONS:
            long.append(dict(base,condition=c,mean_pct=r['means'][c],sd_pct=r['sd'][c],human_pct=r['human'][c],difference_pp=r['errors'][c],fitted=c in r['fit_cells'],source_csv=r['sources'][c],source_cell=r['source_cells'][c]))
            for s,v in zip(range(101,151),r['samples'][c]):trials.append(dict(base,condition=c,seed=s,accuracy_pct=v,source_csv=r['sources'][c],source_cell=r['source_cells'][c]))
    write_csv(DATA/'yin_results.csv',wide);write_csv(DATA/'yin_conditions.csv',long);write_csv(DATA/'yin_trials.csv',trials)
    write_csv(DATA/'rankings.csv',ranks);write_csv(DATA/'kanwisher_results.csv',kanw)
    write_csv(DATA/'human_yin.csv',[dict(category=cat,**dict(zip(CONDITIONS,vals))) for cat,vals in HUMAN.items()])
    (DATA/'results.json').write_text(json.dumps(dict(models=MODELS,groups=groups,rankings=ranks,kanwisher=kanw),indent=2))


def savefig(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def figures(groups,ranks,kanw):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for protocol in PROTOCOLS:
        fig,axes=plt.subplots(2,3,figsize=(13,7.4),sharex=True,sharey=True)
        for j,face in enumerate(FACES):
            for k,cat in enumerate(HUMAN):
                ax=axes[j,k]
                ax.plot(range(4),HUMAN[cat],color='#222222',lw=2.7,marker='s',ls='--',label='Human')
                for key,meta in MODELS.items():
                    g=next(r for r in groups if (r['model'],r['protocol'],r['calibrated_on'],r['category'],r['scope'])==(key,protocol,face,cat,'face_transfer'))
                    ax.plot(range(4),[g['means'][c] for c in CONDITIONS],color=meta['color'],lw=1.7,marker='o',ms=4,label=meta['name'])
                ax.set_title(f'{cat} · {face} calibration',fontsize=11);ax.set_ylim(45,102);ax.set_xticks(range(4),CONDITIONS);ax.grid(axis='y',alpha=.18)
                if k==0:ax.set_ylabel('Accuracy (%)')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.005))
        fig.suptitle(f'{protocol}: final epoch 124 · 50 seeds per condition',fontsize=16,y=.99)
        fig.tight_layout(rect=(0,.065,1,.96));savefig(fig,protocol)
    fig,ax=plt.subplots(figsize=(10,10))
    ordered=sorted(ranks,key=lambda r:r['balanced_prediction_mae'])
    vals=np.array([[r['face_prediction_mae'],r['objects_mae'],r['houses_mae']] for r in ordered])
    im=ax.imshow(vals,aspect='auto',cmap='YlOrRd',vmin=0,vmax=32)
    ax.set_yticks(range(len(ordered)),[f"{r['model_name']} | {r['protocol']} | {r['calibrated_on'].split()[0]}" for r in ordered],fontsize=9)
    ax.set_xticks(range(3),['Faces: UI + IU','Objects: all four','Houses: all four'])
    ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,length=0)
    for y in range(len(ordered)):
        for x in range(3):ax.text(x,y,f'{vals[y,x]:.1f}',ha='center',va='center',color='white' if vals[y,x]>22 else '#222',fontsize=9)
    ax.set_title('Distance from the human result (percentage points)\nRows sorted by equal-category mean; lower is closer',pad=42)
    fig.colorbar(im,ax=ax,fraction=.035,pad=.025,label='Mean absolute error (pp)')
    fig.tight_layout();savefig(fig,'human_distance')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),sharey=True)
    for ax,label in zip(axes,('RFW WM64','CFD WMk','Set A')):
        for n,key in enumerate(('noaa','aa4','bin5')):
            r=next(r for r in kanw if r['model']==key and r['category']==label)
            ax.errorbar(n,r['cost_mean'],yerr=r['cost_sem'],fmt='o',capsize=4,ms=7,color=MODELS[key]['color'])
        ax.axhline(10.70,ls='--',color='#222',lw=1.7,label='Human: between subjects (10.70)')
        ax.axhline(11.60,ls=':',color='#666',lw=1.7,label='Human: within subject (11.60)')
        ax.set_xticks(range(3),['No AA','AA4','bin5']);ax.set_title(label);ax.set_ylim(0,13);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Upright − inverted accuracy (pp)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Kanwisher: every model has a smaller inversion cost than the human benchmark')
    fig.tight_layout(rect=(0,.08,1,.9));savefig(fig,'kanwisher_costs')


def fcell(r,c):return f"{r['means'][c]:.2f} ± {r['sd'][c]:.2f}"
def table(rows,epoch=False):
    lines=['| Model | Protocol | calibrated_on | Category | '+('Epoch | ' if epoch else '')+'UU | II | UI | IU | MAE vs human |',
           '| --- | --- | --- | --- | '+('---: | ' if epoch else '')+'---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append('| '+' | '.join([r['model_name'],r['protocol'],r['calibrated_on'],r['category']]+([str(r['epoch'])] if epoch else [])+[fcell(r,c) for c in CONDITIONS]+[f"{r['mae']:.2f}"])+' |')
    return '\n'.join(lines)


def markdown(groups,ranks,kanw):
    bestface=min(ranks,key=lambda r:r['face_prediction_mae']);best=ranks[0]
    lines=['# Four VGG16-BN models: behavioral results and training curves','',
           'Compiled 28 September 2026 from saved seed results. **No behavioral simulations were rerun.**', '',
           '## The four models','']
    lines += [f"- **{v['name']}** — {v['description']}" for v in MODELS.values()]
    lines += ['', 'All four use the log-polar front end, a 256-unit bottleneck, training seed 42, and a ten-stage, 124-epoch curriculum. The main comparison uses **final epoch 124**, not each model’s validation-best checkpoint. AA4 and bin5 name the normalized binomial filters; there are three VGG2k models in total, two of them antialiased.', '',
              '## The three Yin protocols','',
              'Each entry below is **study noise / test noise**. Noise flips bits in the sampled representation in both phases. U = upright, I = inverted; the first condition letter is study orientation. `single_noise` in the newer report is the same protocol as `single_uu` here.', '',
              '| Protocol | UU | II | UI | IU | Calibration |', '| --- | --- | --- | --- | --- | --- |',
              '| `single_uu` | p1/p1 | p1/p1 | p1/p1 | p1/p1 | Fit p1 to UU; reuse it everywhere. |',
              '| `two_cross` | p1/p1 | p2/p2 | p1/p2 | p2/p1 | Fit p1 to UU and p2 independently to II. |',
              '| `two_shared_ui` | p1/p1 | p2/p2 | p1/p1 | p2/p1 | Same two fits; UI uses the upright noise rate in both phases. |', '',
              '**`calibrated_on` identifies the stimulus set that chose the noise rates; `category` identifies what was evaluated.** RFW and CFD rates transfer unchanged to objects and houses. Category-specific calibration is a separate setting of `single_uu`, not a fourth protocol. It exists for AA4 VGG2k and mixed-face AA4 only.', '',
              'The newer no-AA/bin5 report listed two protocols, but its saved transfer records already contain every cell needed for `two_shared_ui`. This report assembles those matching records without additional evaluations.', '',
              '## How to read this report','',
              '- Start with the judgment and overview below. Use the [interactive comparison](vgg16bn_results_report.html) to select one protocol, calibration, and category at a time (four model rows).',
              '- Expand a category below for exact values. Yin values are **mean ± SD across 50 simulation seeds (101–150)**, not uncertainty across trained models. Full precision and provenance are in the [80-row result file](vgg16bn_report_data/yin_results.csv) and [condition file](vgg16bn_report_data/yin_conditions.csv).',
              '- Earlier checkpoints and category-specific fits have their own sections. Repeated summaries of the same seed results count once. The latest Kanwisher results appear at the end.', '',
              '## Human reference','',
              '| Category | UU | II | UI | IU | IU − II |','| --- | ---: | ---: | ---: | ---: | ---: |']
    lines += ['| '+cat+' | '+' | '.join(f'{v:.2f}' for v in vals)+f' | {vals[3]-vals[1]:+.2f} |' for cat,vals in HUMAN.items()]
    lines += ['', 'These are the repository’s Yin (1969) reference means. They are **not human measurements on RFW, CFD, or the model’s house/object images**. The repository’s “objects” reference is Yin’s airplanes. Human means come from [the original project tables](yin_replication.tex) and [the saved reference table](../runs/yin_noise_matrix_20260926/human_references.csv). Model seed SD and human participant SD are different quantities; plots show the reference mean only.', '',
              '## Which results look most like humans?','',
              f"**Closest final-checkpoint face predictions numerically: {bestface['model_name']}, `two_cross`, calibrated on {bestface['calibrated_on']}.** Its mean absolute error on the two unfitted face cells, UI and IU, is **{bestface['face_prediction_mae']:.2f} pp**. The accuracies are UI 85.58% and IU 82.75%, versus human 84.13% and 78.58%. IU remains too high and sits above II, whereas the human IU is below II.", '',
              '**My preferred final-model face-pattern candidate is mixed faces / AA4, `two_cross`, RFW calibration.** It is a little less close numerically (UI/IU error 3.23 pp), but reproduces the human ordering **UU > UI > II > IU**: 95.67, 85.92, 83.67, 83.25%. Its IU–II gap is only −0.42 pp versus the human −3.30 pp, so this remains an incomplete match. The `two_shared_ui` version preserves that ordering but moves UI farther from the human value.', '',
              f"**Across all three categories with one face calibration, the lowest balanced error is {best['model_name']}, `{best['protocol']}`, calibrated on {best['calibrated_on']} ({best['balanced_prediction_mae']:.2f} pp).** Its lead over mixed-face AA4 with CFD `two_cross` is only {ranks[1]['balanced_prediction_mae']-best['balanced_prediction_mae']:.2f} pp, so the descriptive ranking does not establish a clear winner. Objects remain too accurate and houses too inaccurate. A face winner should therefore not be presented as the winner across categories.", '',
              '**When objects/houses can choose their own noise rate**, AA4 VGG2k with `single_uu` is the strongest available category-fit candidate: object error on II/UI/IU is 2.78 pp; house error is 4.57 pp. The house II value is 92.67%, above UU 90.58%, while humans show the opposite ordering. No own-category runs are available for no-AA or bin5, so this is not a complete four-model comparison.', '',
              '**The earlier mixed-face AA4 epoch-111 snapshot is the closest saved face UI/IU result** (RFW, `two_cross`, 1.35 pp error). It is an earlier checkpoint selected after seeing these results, and not evidence that the final model achieves that fit. It also puts IU above II.', '',
              '### How the comparisons were scored','',
              'MAE is the average absolute model–human difference in percentage points. For a fair face comparison across protocols, the ranking uses the common unfitted cells **UI and IU**; fitting UU/II close to the human target does not count as independent predictive success. The balanced score averages three errors with equal category weight: face UI/IU MAE, object four-cell MAE, and house four-cell MAE. Object/house cells are all predictions when noise was fitted on faces. The numerical ranking is descriptive: one training seed, shared fitting/evaluation seeds, no uncertainty for the human means, and no correction for selecting among many settings.', '',
              '| Model | Protocol | calibrated_on | Face UI/IU MAE | Object MAE | House MAE | Balanced MAE |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in ranks[:6]:lines.append('| '+' | '.join([r['model_name'],r['protocol'],r['calibrated_on']]+[f"{r[k]:.2f}" for k in ('face_prediction_mae','objects_mae','houses_mae','balanced_prediction_mae')])+' |')
    lines += ['', '![Distance to human data for all 24 model/protocol/calibration combinations](figures/vgg16bn_comparison/human_distance.png)', '',
              '[All ranking scores](vgg16bn_report_data/rankings.csv). Lower is closer; red cells reveal where an apparently good face fit fails to transfer.', '',
              '## Final-checkpoint Yin results','',
              'All **72 combinations** are included: 4 models × 3 protocols × 2 face calibrations × 3 categories. Each expanded table has eight rows. Values are mean ± SD (%); the final column is four-cell MAE in percentage points, including calibration cells where applicable.']
    for protocol in PROTOCOLS:
        lines += ['', f'### {protocol}', '', f'![{protocol} human comparison](figures/vgg16bn_comparison/{protocol}.png)', '']
        for cat in HUMAN:
            selected=[r for r in groups if r['scope']=='face_transfer' and r['protocol']==protocol and r['category']==cat]
            lines += [f'<details><summary>{cat}: all four models × two calibrations</summary>', '',
                      '**Human UU / II / UI / IU:** '+' / '.join(f'{v:.2f}' for v in HUMAN[cat])+'.', '',table(selected), '', '</details>', '']
    lines += ['## Category-specific calibration','',
              'These four additional combinations fit UU on the same category that is tested. The mixed-face model’s house target was outside the sampled calibration range: closest UU 87.25% versus 90.71% human (p = 0.02). This is a calibration limitation, not a successful UU match.', '',
              table([r for r in groups if r['scope']=='category_calibration']), '',
              'No-AA and bin5 category-specific results are **not available**. Their face-calibrated house/object results above must not be mistaken for own-category calibration.', '',
              '## Earlier checkpoint snapshots','',
              'These four results appeared in the 25 September report before training finished. They are retained for completeness and excluded from the final-checkpoint ranking.', '',
              table([r for r in groups if r['scope']=='snapshot'],epoch=True), '',
              '## Calibration rates and coverage','',
              '| Model | calibrated_on | p1 (UU) | p2 (II) | Final epoch |', '| --- | --- | ---: | ---: | ---: |']
    for r in groups:
        if r['scope']=='face_transfer' and r['category']=='Faces' and r['protocol']=='two_cross':lines.append(f"| {r['model_name']} | {r['calibrated_on']} | {r['p1']:.2f} | {r['p2']:.2f} | 124 |")
    lines += ['', 'The same p1 supplies `single_uu`. All RFW/CFD face fits were within the sampled range, with residual error from the finite 0.01 refinement grid. Each evaluation used 40 studied items, 24 old/new pairs, 10 study fixations, 32 test fixations, and KDE width 2. Faces were shuffled per seed; objects/houses used the packed 40/24 split.', '',
              '## Training curves','',
              '<!-- TRAINING_CURVES_START -->',
              'The training-history audit is complete. Curve images and the checkpoint reconstruction are being prepared.',
              '<!-- TRAINING_CURVES_END -->', '',
              '## Sources and reproducibility','',
              'The compilation reads seed CSVs rather than transcribing rounded Markdown. It checks unique seeds, full counts, completed records, and equality of reused UU/UI cells. Exact source paths and checkpoint hashes accompany the [JSON data](vgg16bn_report_data/results.json); [source file hashes](vgg16bn_report_data/source_manifest.json) identify the audited inputs. The [generator](../scripts/build_vgg16bn_report.py) rebuilds the behavioral report and figures.', '',
              '<details><summary>Source reports and supporting records</summary>', '']
    lines += [f'- [{p}]({"../"+p if not p.startswith("paper/") else p[6:]})' for p in SOURCE_MDS]
    lines += ['', '</details>', '', '## Latest Kanwisher / Dobs matching results','',
              'These are the latest final-epoch results: **20 seeds (101–120)**, noise calibrated separately on each model/store to the 87.5% upright target using calibration seed 42. Here uncertainty is **SEM**, matching the Kanwisher source report (the Yin tables above use SD). Inversion cost is computed upright minus inverted within each seed.', '',
              '| Human reference | Upright | Inverted | Inversion cost (pp) |', '| --- | ---: | ---: | ---: |',
              '| Between-subjects benchmark | 87.50 | 76.80 | 10.70 |', '| Within-subject benchmark | 87.50 | 75.90 | 11.60 |', '',
              'Human values are the project’s Dobs et al. (2023) references in [the existing results summary](curriculum_results_summary.md). They are benchmark values, not human data collected on these three model evaluation stores.', '',
              '![Kanwisher inversion costs against both human references](figures/vgg16bn_comparison/kanwisher_costs.png)', '']
    for label in ('RFW WM64','CFD WMk','Set A'):
        lines += [f'### {label}', '', '| Model | Protocol | calibrated_on | Category | p | Upright ± SEM | Inverted ± SEM | Cost ± SEM |','| --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
        for r in kanw:
            if r['category']==label:lines.append('| '+' | '.join([r['model_name'],r['protocol'],r['calibrated_on'],r['category'],f"{r['p']:.2f}"]+[f"{r[k+'_mean']:.2f} ± {r[k+'_sem']:.2f}" for k in ('upright','inverted','cost')])+' |')
        lines.append('')
    lines += ['**None reproduces the human inversion cost.** Bin5 with CFD calibration has the largest, closest cost (6.19 pp), still 4.51 pp below the 10.70 pp benchmark. If judged only on the predicted inverted accuracy, AA4 VGG2k with RFW calibration is closest (82.73%, 5.93 pp above 76.80%); its cost is only 4.46 pp. The preferred candidate therefore depends on whether the criterion is accuracy level or inversion sensitivity.', '',
              'The latest batch contains the three VGG2k variants; **mixed-face AA4 is not present**. RFW samples 40 identities from `faces_rfwWM64` (four photos each); CFD uses 36 identities from **`faces_cfdWMk`**, a different store from Yin’s CFD WM64; Set A uses 40 identities with five photos each. Each trial matches different photos of the same person against another identity.', '',
              '[Latest source report](../runs/kanw_latest_vgg_20260927/summary.md) · [Exact means, SEMs, and provenance](vgg16bn_report_data/kanwisher_results.csv)', '']
    target=PAPER/'vgg16bn_results_report.md'
    if target.exists():
        prior=target.read_text();a='<!-- TRAINING_CURVES_START -->';b='<!-- TRAINING_CURVES_END -->'
        if a in prior and b in prior:
            block=prior.split(a,1)[1].split(b,1)[0]
            text='\n'.join(lines);text=text.split(a)[0]+a+block+b+text.split(b,1)[1]
            target.write_text(text);return
    target.write_text('\n'.join(lines))


def interactive(groups,ranks,kanw):
    payload=json.dumps({'models':MODELS,'groups':[{k:v for k,v in r.items() if k!='samples'} for r in groups],'human':HUMAN,'kanwisher':kanw})
    template=(ROOT/'scripts/vgg16bn_report_template.html').read_text()
    (PAPER/'vgg16bn_results_report.html').write_text(template.replace('__DATA__',payload))


def main():
    DATA.mkdir(exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    groups=collect();ranks=rankings(groups);kanw=collect_kanw()
    export(groups,ranks,kanw);figures(groups,ranks,kanw);markdown(groups,ranks,kanw);interactive(groups,ranks,kanw)
    manifest={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(SOURCES)}
    (DATA/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Compiled {len(groups)} Yin groups, {len(groups)*4} conditions, and {len(kanw)} Kanwisher groups.')
    print('Closest faces:',min(ranks,key=lambda r:r['face_prediction_mae']))
    print('Best balanced:',ranks[0])

if __name__=='__main__':main()
