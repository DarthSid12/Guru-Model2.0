"""Select sources without silently changing the research dataset."""
import tempfile
from pathlib import Path
from execution.common.io import new_output,write_json,sha,safe_name
from .sources import materialize

def validate(c):
    if not c['datasets'] or len(c['datasets'])!=len(set(c['datasets'])):raise ValueError('Select a nonempty list of unique datasets.')
    for name in c['datasets']:
        safe_name(name)
        if name not in c['sources']:raise ValueError(f'No source configured for {name}')
        s=c['sources'][name]
        if s.get('provider') not in ['directory','archive','manifest'] or not s.get('location'):raise ValueError(f'Invalid source for {name}')

def run(c):
    validate(c)
    # Fail before creating an output if any locally supplied input is absent.
    for name in c['datasets']:
        loc=c['sources'][name]['location']
        if not loc.startswith(('http://','https://')) and not Path(loc.removeprefix('file://')).exists():raise FileNotFoundError(f'{name}: {loc}. Configure the authorized source location in execution/configs/download.json or --config.')
    out=new_output(c['output'],c);receipts={}
    for name in c['datasets']:
        with tempfile.TemporaryDirectory(prefix='guru-source-') as temp:materialize(c['sources'][name],out/name,Path(temp))
        files=[p for p in (out/name).rglob('*') if p.is_file()]
        if not files:raise ValueError(f'Empty dataset: {name}')
        receipts[name]={str(p.relative_to(out/name)):sha(p) for p in files}
    write_json(out/'DOWNLOAD_COMPLETE.json',dict(datasets=receipts))
