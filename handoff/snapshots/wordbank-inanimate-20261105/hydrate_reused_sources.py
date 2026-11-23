"""Read and hash retained immutable photos concurrently to hydrate iCloud files."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'REUSED_PHOTO_SPLITS.json').read_text())['images']
def read(row):
 p=Path(row['reuse_source_root'])/row['normalized_path'];body=p.read_bytes()
 assert hashlib.sha256(body).hexdigest()==row['png_sha256'],str(p)
 return len(body)
total=0
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
 for i,nbytes in enumerate(pool.map(read,rows),1):
  total+=nbytes
  if i%700==0:
   (ROOT/'REUSE_HYDRATION_PROGRESS.json').write_text(json.dumps({'verified_photos':i,'target':len(rows),'bytes':total})+'\n')
   print('Hydrated and hash-verified',i,'/',len(rows),flush=True)
(ROOT/'REUSE_HYDRATION_VERIFIED.json').write_text(json.dumps({'verified_photos':len(rows),'bytes':total,'all_png_hashes_match':True})+'\n')
