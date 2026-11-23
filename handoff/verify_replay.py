"""Compare a portable run with the frozen current single-noise/pooled evidence."""
import argparse,csv,hashlib,json,statistics,math
from pathlib import Path
BASE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
 rows=list(csv.DictReader((a.output/'all_records.csv').open()))
 single=list(csv.DictReader((BASE/'snapshots/local-yin-sigma2-single-uu-20261119/all_records.csv').open()))
 pooled=list(csv.DictReader((BASE/'snapshots/local-yin-pooled-house-kde-20261119/all_records.csv').open()))
 expected={(r['variant'],r['design'],'original_max40',r['category'],int(r['seed']),r['condition']):r for r in single}
 for r in pooled:
  if r['method']!='original_max40':expected[r['variant'],'faces_houses',r['method'],'houses',int(r['seed']),r['condition']]=r
 seen=set()
 for r in rows:
  key=(r['variant'],r['design'],r['method'],r['category'],int(r['seed']),r['condition'])
  assert key not in seen;seen.add(key);ref=expected[key]
  assert r['correct_mask']==ref['correct_mask'] and float(r['sigma'])==2.
  assert float(r['noise'])==float(ref['noise'])
  assert int(r['n_trials'])==12 and int(r['correct'])==r['correct_mask'].count('1')
  assert math.isclose(float(r['accuracy']),100*int(r['correct'])/12,abs_tol=1e-12)
 run=json.loads((a.output/'RUN.json').read_text());n=2400 if run['pooled_houses'] else 1600
 assert len(rows)==n
 summary=json.loads((a.output/'SUMMARY.json').read_text())
 for cell in summary['cells']:
  rs=[r for r in rows if all(r[k]==cell[k] for k in ['variant','design','method','category','condition'])]
  assert sorted(int(r['seed']) for r in rs)==list(range(101,151))
  values=[float(r['accuracy']) for r in rs];sd=statistics.stdev(values)
  for k,x in [('mean',statistics.mean(values)),('sd',sd),('sem',sd/math.sqrt(50))]:assert math.isclose(cell[k],x,abs_tol=1e-12)
 if run['calibrate_uu']:
  for v,target in [('lp',.373),('cnn',.364)]:
   fit=json.loads((a.output/f'{v}-UU-calibration.json').read_text());assert fit['selected_noise']==target and fit['condition']=='UU' and fit['test_noise_held_at'] is None
 print(json.dumps(dict(passed=True,category_records=n,all_decision_masks_match=True,all_summary_statistics_verified=True,noise_refit_verified=run['calibrate_uu'],rows_sha256=hashlib.sha256((a.output/'all_records.csv').read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':main()
