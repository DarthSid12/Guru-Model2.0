import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONDITIONS=('UU','UI','II','IU')
PROTOCOLS=('single_uu','study_only_fitted','two_cross')
SEEDS=tuple(range(101,151))
SOURCES={'faces':'faces_cfdWM64','houses':'houses_dal64','objects':'objects_wordbank64'}
POOL_SIZES={'faces':64,'houses':64,'objects':64}
def counts(pool,study,trials):return dict(pool=pool,study=study,trials=trials)
DESIGNS={
 'faces_only':{'faces':counts(64,40,24)},
 'houses_only':{'houses':counts(64,40,24)},
 'objects_only':{'objects':counts(64,40,24)},
 'faces_houses':{'faces':counts(32,20,12),'houses':counts(32,20,12)},
 'faces_objects':{'faces':counts(32,20,12),'objects':counts(32,20,12)},
 'houses_objects':{'houses':counts(32,20,12),'objects':counts(32,20,12)},
 'faces_houses_objects':{'faces':counts(22,14,8),'houses':counts(21,13,8),'objects':counts(21,13,8)},
}
HUMAN={'faces':{'UU':1-.89/12,'II':1-4.35/12},'houses':{'UU':1-2.23/12,'II':1-3.42/12}}
def rates(protocol,condition,p1,p2):
 assert protocol in PROTOCOLS and condition in CONDITIONS and 0<=p1<=.5 and 0<=p2<=.5
 if protocol=='single_uu':return p1,p1
 return (p1 if condition[0]=='U' else p2),(p1 if protocol=='study_only_fitted' or condition[1]=='U' else p2)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)
def spec():return json.loads((ROOT/'SPEC.json').read_text())
def verify():
 s=spec()
 for n,h in s['files'].items():assert sha(ROOT/n)==h,n
 for n,h in s['external_files'].items():assert sha(n)==h,n
 return s
