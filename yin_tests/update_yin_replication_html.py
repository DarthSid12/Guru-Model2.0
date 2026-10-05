"""Add validated, completed training-seed replications to the saved HTML report."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path
import csv
import html
import json
import statistics
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/yin_aa5_seeds43_44_20261004'
START='<!-- YIN_REPLICATION_START -->'
END='<!-- YIN_REPLICATION_END -->'

def update():
    rows=[];status=[]
    old=json.loads((ROOT/'paper/vgg16bn_report_data/results.json').read_text())['groups']
    for seed in (43,44):
        model=f'r21vgg2k_vgg16_bn_aa5_s{seed}'
        for face,label in (('faces_rfwWM64','RFW WM64'),('faces_cfdWM64','CFD WM64')):
            one=OUT/'single_noise'/model/face
            two=OUT/'transfer/results'/model/face
            complete=all((folder/category/'DONE.json').exists() for folder in (one,two) for category in (face,'objects','houses_yin64'))
            status.append(f'<li>Training seed {seed} · {label}: {"complete" if complete else "pending"}</li>')
            if not complete: continue
            hashes=set()
            for category,cat in ((face,'Faces'),('objects','Objects'),('houses_yin64','Houses')):
                for protocol in ('single_uu','two_cross','two_shared_ui'):
                    folder=(one if protocol=='single_uu' else two)/category
                    done=json.loads((folder/'DONE.json').read_text());hashes.add(done['checkpoint_sha256'])
                    with (folder/'results.csv').open() as f: data=list(csv.DictReader(f))
                    assert len(data)==(200 if protocol=='single_uu' else 250)==done['rows']
                    baseline=next(g for g in old if (g['model'],g['scope'],g['protocol'],g['calibrated_on'],g['category'])==('bin5','face_transfer',protocol,label,cat))
                    vals=[];deltas=[]
                    for condition in ('UU','II','UI','IU'):
                        cell=('UI_cross' if protocol=='two_cross' else 'UI_shared') if condition=='UI' and protocol!='single_uu' else condition
                        records=[r for r in data if r['condition' if protocol=='single_uu' else 'cell']==cell]
                        assert len(records)==50 and {int(r['seed']) for r in records}==set(range(101,151))
                        assert {r['checkpoint_sha256'] for r in records}=={done['checkpoint_sha256']}
                        accuracies=[float(r['accuracy_pct']) for r in records]; assert all(0<=v<=100 for v in accuracies)
                        mean=statistics.mean(accuracies);sd=statistics.stdev(accuracies)
                        vals.append(f'{mean:.2f} ± {sd:.2f}')
                        deltas.append(f'{mean-baseline["means"][condition]:+.2f}')
                        rows.append(dict(training_seed=seed,protocol=protocol,calibration=label,category=cat,condition=condition,mean_pct=mean,sd_pct=sd,seed42_mean_pct=baseline['means'][condition],delta_from_seed42_pp=mean-baseline['means'][condition],n_simulation_seeds=50,checkpoint_sha256=done['checkpoint_sha256'],source=str(folder.relative_to(ROOT))))
                    status.append('<tr>'+' '.join(f'<td>{html.escape(str(v))}</td>' for v in (seed,protocol,label,cat,*vals,', '.join(deltas)))+'</tr>')
            assert len(hashes)==1
    timestamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%B %d, %Y at %H:%M %Z')
    lis=''.join(s for s in status if s.startswith('<li>'));trs=''.join(s for s in status if s.startswith('<tr>'))
    block=f'''{START}<section class="panel" id="training-seed-replications"><h2>Bin5: training seeds 43 and 44</h2><p>Updated {timestamp}. Final epoch 124, using the same three Yin protocols, independent RFW/CFD face calibrations, and simulation seeds 101–150 as training seed 42. Rates are fitted separately for each trained checkpoint and transferred to objects and houses.</p><ul>{lis}</ul><p>Only completed calibration sets appear below. Values show mean ± SD across simulation seeds; they do not measure uncertainty across training seeds. The original four-model explorer above remains the seed 42 comparison.</p><div class="tablewrap"><table><thead><tr><th>Training seed</th><th>Protocol</th><th>Calibration</th><th>Category</th><th>UU</th><th>II</th><th>UI</th><th>IU</th><th>Δ from seed 42 (UU, II, UI, IU; pp)</th></tr></thead><tbody>{trs}</tbody></table></div><p><a href="vgg16bn_report_data/yin_training_seed_replications.csv">Download completed replication results and provenance (CSV)</a></p></section>{END}'''
    targets=[ROOT/'paper/vgg16bn_results_report.html',ROOT/'paper/vgg16bn_share_site/dist/index.html',ROOT/'paper/vgg16bn_share_site/dist/full-report.html']
    for target in targets:
        text=target.read_text()
        if START in text: text=text.split(START)[0]+block+text.split(END,1)[1]
        else: text=text.replace('</main>',block+'</main>',1) if '</main>' in text else text.replace('</body>',block+'</body>',1)
        target.write_text(text)
    for directory in (ROOT/'paper/vgg16bn_report_data',ROOT/'paper/vgg16bn_share_site/dist/vgg16bn_report_data'):
        with (directory/'yin_training_seed_replications.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['training_seed','protocol','calibration','category','condition']);writer.writeheader();writer.writerows(rows)
    print(f'Updated 3 HTML documents; {len(rows)} verified condition rows, {len(rows)//4} result groups.')

if __name__=='__main__': update()
