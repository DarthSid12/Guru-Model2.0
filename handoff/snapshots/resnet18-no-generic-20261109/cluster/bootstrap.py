"""I/O-only staging of an exact already-registered source snapshot."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 site=json.loads((ROOT/'site.json').read_text());spec=json.loads((ROOT/'SPEC.json').read_text())
 assert str(ROOT)==site['root'];assert not (ROOT/'registration.json').exists(),'Existing registration must not be overwritten'
 baseline=Path(site['baseline_code']);code=ROOT/'code'
 for name,digest in spec['source_files'].items():
  source=baseline/name;assert sha(source)==digest,'Baseline source changed: '+name
  dest=code/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(dest)==digest
  else:shutil.copyfile(str(source),str(dest))
 expected=Path(site['expected_receipt']);assert sha(expected)==site['expected_receipt_sha256'],'Baseline readiness receipt changed'
 shutil.copyfile(str(expected),str(ROOT/'incoming/BASELINE_EXPECTED.json'))
 (ROOT/'BOOTSTRAPPED.json').write_text(json.dumps({'source_commit':spec['source_commit'],'source_files':spec['source_files'],'baseline_receipt_sha256':sha(expected)},indent=2)+'\n')
 print('Exact source and baseline data receipt staged',len(spec['source_files']),flush=True)
if __name__=='__main__':main()
