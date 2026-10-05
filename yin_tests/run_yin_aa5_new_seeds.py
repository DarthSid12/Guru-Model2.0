"""Repeat the published final-checkpoint bin5 Yin protocols for seeds 43/44."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import csv
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/yin_aa5_seeds43_44_20261004'
SOURCE = OUT / 'source'
FACES = ('faces_rfwWM64', 'faces_cfdWM64')
MODELS = tuple(f'r21vgg2k_vgg16_bn_aa5_s{s}' for s in (43,44))

def command(script, *args):
    subprocess.run([sys.executable, '-u', str(SOURCE / script), *map(str,args)], check=True)

def prepare(model, face):
    spec = importlib.util.spec_from_file_location('prepare', ROOT/'yin_tests/prepare_latest_vgg_single.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.MODELS = {'aa5': model}
    sys.argv = ['prepare', '--model','aa5','--face',face]
    module.main()

def aggregate():
    rows = []
    for model in MODELS:
        for face in FACES:
            for category in (face,'objects','houses_yin64'):
                one = OUT/'single_noise'/model/face/category
                two = OUT/'transfer/results'/model/face/category
                for folder, expected in ((one,200),(two,250)):
                    if json.loads((folder/'DONE.json').read_text())['rows'] != expected:
                        raise ValueError(f'Incomplete {folder}')
                for protocol in ('single_uu','two_cross','two_shared_ui'):
                    folder = one if protocol == 'single_uu' else two
                    with (folder/'results.csv').open() as handle: data = list(csv.DictReader(handle))
                    for condition in ('UU','II','UI','IU'):
                        cell = ('UI_cross' if protocol == 'two_cross' else 'UI_shared') if condition == 'UI' and protocol != 'single_uu' else condition
                        selected = [r for r in data if r['condition' if protocol == 'single_uu' else 'cell']==cell]
                        assert len(selected)==50 and {int(r['seed']) for r in selected}==set(range(101,151))
                        values=[float(r['accuracy_pct']) for r in selected]
                        rows.append(dict(model=model,protocol=protocol,calibration_face=face,category=category,condition=condition,mean_pct=statistics.mean(values),sd_pct=statistics.stdev(values),n_seeds=50,source=str(folder.relative_to(ROOT))))
    with (OUT/'all_conditions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=['# Bin5 training seeds 43 and 44: published Yin protocols','', 'Final epoch 124; simulation seeds 101–150. Independent RFW/CFD calibration, transferred to faces, objects and houses.','', '| Model | Protocol | Calibration | Category | UU | II | UI | IU |','| --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for offset in range(0,len(rows),4):
        group=rows[offset:offset+4];r=group[0]
        report.append('| '+' | '.join([r['model'],r['protocol'],r['calibration_face'],r['category']]+[f"{v['mean_pct']:.2f} ± {v['sd_pct']:.2f}" for v in group])+' |')
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'STATUS').write_text('running\n')
    sys.path.insert(0,str(SOURCE))
    import run_latest_vgg_two_cross_transfer as transfer
    transfer.MODELS = ('unused_noaa', *MODELS)
    started=time.time()
    try:
        for seed,model in zip((43,44),MODELS):
            run=next((ROOT/'runs').glob(f'*aa5_s{seed}'))
            checkpoint=next(run.glob('final_model_*.pth'))
            for face in FACES:
                print(f'[{time.strftime("%Y-%m-%dT%H:%M:%S%z")}] starting {model} {face}',flush=True)
                command('run_yin_orientation.py','--model',model,'--run-dir',run,'--checkpoint',checkpoint,'--out-dir',OUT/'two_cross'/model/face,'--category',face,'--epoch',124,'--stage','final','--device','cuda:0','--seeds','101-150')
                one=OUT/'single_noise'/model/face
                if not (one/'DONE.json').exists(): prepare(model,face)
                command('run_single_noise.py','--model',model,'--run-dir',run,'--checkpoint',checkpoint,'--out-dir',one,'--face',face,'--epoch',124,'--device','cuda:0','--seeds','101-150')
                inputs=transfer.checked_inputs(model,face,list(range(101,151)))
                dest=OUT/'transfer/results'/model/face
                dest.mkdir(parents=True,exist_ok=True)
                (dest/'manifest.json').write_text(json.dumps(dict(model=model,face=face,p1=inputs['p1'],p2=inputs['p2'],checkpoint_sha256=inputs['checkpoint_sha256'],seeds=list(range(101,151)),epoch=124),indent=2))
                for category in (face,'objects','houses_yin64'):
                    transfer.run_category(model,face,category,list(range(101,151)),inputs,dest/category,'cuda:0')
        aggregate()
        (OUT/'STATUS').write_text('complete\n')
        print(f'Completed in {(time.time()-started)/3600:.2f} hours',flush=True)
    except BaseException:
        (OUT/'STATUS').write_text('failed; see logs/experiments.log\n')
        raise

if __name__=='__main__': main()
