"""Dataset, output and provenance contracts shared by all stages."""
import hashlib,json,os,random,re,sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def new_output(path,config):
    path=Path(path)
    if path.exists():raise FileExistsError(f'{path} already exists. Use a new output path; no existing experiment is overwritten.')
    path.mkdir(parents=True)
    code=Path(__file__).resolve().parents[1]
    write_json(path/'RUN.json',dict(config=config,python=sys.version,source_files={str(p.relative_to(code)):sha(p) for p in code.rglob('*.py')}))
    return path

def seeds(text):
    if isinstance(text,list):out=[int(x) for x in text]
    else:
        out=[]
        for group in text.split(','):
            if '-' in group:
                lo,hi=map(int,group.split('-'));out.extend(range(lo,hi+1))
            else:out.append(int(group))
    if not out or len(out)!=len(set(out)) or min(out)<0:raise ValueError('Seeds must be a nonempty, unique list of nonnegative integers.')
    return out

def safe_name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',value) or value in ['.','..']:raise ValueError(f'Invalid name: {value!r}')
    return value

def category_names(c):
    if not c['categories'] or len(set(c['categories']))!=len(c['categories']):raise ValueError('Choose unique, nonempty categories.')
    for x in c['categories']:safe_name(x)
    if not c['splits'] or not set(c['splits'])<=set(['train','valid','test']):raise ValueError('Splits must be train, valid and/or test.')

def rate(p):
    if not isinstance(p,(int,float)) or not 0<=p<=.5:raise ValueError('Noise rates must be in [0, 0.5].')

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def link_or_copy(src,dst):
    import shutil
    try:os.link(src,dst)
    except OSError:shutil.copy2(src,dst)
