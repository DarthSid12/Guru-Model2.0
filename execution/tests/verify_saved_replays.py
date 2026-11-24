"""Optional full-data regression: compare new runs with saved scientific evidence.

python execution/tests/verify_saved_replays.py --yin /new/yin/output \
    --kanwisher /new/kanwisher/output --old-kanwisher /old/completed-sdsc/results
"""
import argparse,csv,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def verify_summary(path,rows,keys):
    cells=json.loads((path/'SUMMARY.json').read_text())['cells']
    for cell in cells:
        values=[float(r['accuracy']) for r in rows if all(r[k]==cell[k] for k in keys)]
        assert len(values)==50
        sd=statistics.stdev(values)
        for k,v in [('mean',statistics.mean(values)),('sd',sd),('sem',sd/math.sqrt(50))]:assert math.isclose(cell[k],v,abs_tol=1e-10)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--yin',type=Path,required=True);ap.add_argument('--kanwisher',type=Path,required=True);ap.add_argument('--old-kanwisher',type=Path,required=True);a=ap.parse_args()
    original=ROOT/'handoff/snapshots/local-yin-sigma2-single-uu-20261119/all_records.csv'
    old={(r['variant'],r['design'],r['category'],r['seed'],r['condition']):r for r in csv.DictReader(original.open())}
    yin=list(csv.DictReader((a.yin/'all_records.csv').open()));assert len(yin)==1600
    for r in yin:
        x=old[r['model'],r['design'],r['category'],r['seed'],r['condition']]
        assert r['correct_mask']==x['correct_mask'] and float(r['noise'])==float(x['noise']) and int(r['trials'])==12
    verify_summary(a.yin,yin,['model','design','category','condition'])
    old={}
    for model in ['lp','cnn']:
        for r in csv.DictReader((a.old_kanwisher/model/'results.csv').open()):old[model,r['simulation_seed'],'U' if r['condition']=='Upright' else 'I']=r
    kan=list(csv.DictReader((a.kanwisher/'all_records.csv').open()));assert len(kan)==200
    for r in kan:
        x=old[r['model'],r['seed'],r['condition']]
        assert int(r['correct'])==int(x['correct_triplets']) and int(r['trials'])==int(x['n_trials'])==156000 and float(r['noise'])==float(x['noise'])
    verify_summary(a.kanwisher,kan,['model','condition'])
    print(json.dumps(dict(passed=True,yin_records=1600,kanwisher_records=200,decision_counts_match=True,noise_match=True,means_sample_SD_SEM_verified=True),indent=2))
if __name__=='__main__':main()
