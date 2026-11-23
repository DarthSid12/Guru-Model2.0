import json,shutil
from pathlib import Path
from common import sha
ROOT=Path(__file__).resolve().parent.parent
site=json.loads((ROOT/'site.json').read_text());spec=json.loads((ROOT/'SPEC.json').read_text())
assert str(ROOT)==site['root'] and not (ROOT/'registration.json').exists()
old=Path(site['parent_root'])
for name in ['photo_manifest.json','class_order.txt','CURRICULUM.json','VERIFIED.json','PHYSICAL_TOY_OVERRIDES.json','BASELINE_EXPECTED.json','COORDINATOR_PACK_JOB.json']:
 dst=ROOT/'incoming'/name
 if dst.exists():assert sha(dst)==sha(old/'incoming'/name)
 else:shutil.copy2(old/'incoming'/name,dst)
for name,src in [('PACK_EXPORT.json',old/'PACK_EXPORT.json'),('OBJECTS_PACK_READY.json',old/'data/OBJECTS_PACK_READY.json')]:
 if site['site']=='anvil':continue
 shutil.copy2(src,ROOT/'incoming'/name)
for name,digest in spec['source_files'].items():
 src=old/'code'/name;dst=ROOT/'code'/name;assert sha(src)==digest,name
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert sha(dst)==digest,name
 else:shutil.copy2(src,dst)
print('Exact parent source and inputs staged; no training started',flush=True)
