from pathlib import Path
import json
from execution.common.io import category_names,new_output,write_json,sha,seed_all,link_or_copy
from .sampling import validate_sampling,sample

def validate(c):category_names(c);validate_sampling(c)

def add_coordinates(directory,c):
    import numpy as np
    import torch
    images=np.load(directory/'images.npy',mmap_mode='r');meta=json.loads((directory/'meta.json').read_text())
    if images.dtype!=np.uint8 or images.ndim!=4 or images.shape[-1]!=3:raise ValueError('Expected uint8 images [N,H,W,3]')
    if len(meta['labels'])!=len(images):raise ValueError('Image/meta row mismatch')
    coords=np.lib.format.open_memmap(directory/'coords.npy',mode='w+',dtype=np.int16,shape=(len(images),c['num_points'],2))
    pipeline=None
    if c['method']=='gabor':
        from training.salience_trans import SaliencePipeline
        pipeline=SaliencePipeline(type='valid',device=c['device'],num_salient_points=c['num_points'],edge_margin=c['edge_margin']).to(c['device'])
    with torch.no_grad():
        for lo in range(0,len(images),c['batch_size']):
            batch=torch.from_numpy(np.array(images[lo:lo+c['batch_size']])).permute(0,3,1,2).float().div(255).to(c['device'])
            pts=sample(batch,c,pipeline);coords[lo:lo+len(batch)]=pts.cpu().numpy().astype(np.int16)
    coords.flush();del coords
    meta.update(num_coords=c['num_points'],edge_margin=c['edge_margin'],salience=dict(c))
    write_json(directory/'meta.json',meta)
    return {name:sha(directory/name) for name in ['images.npy','coords.npy','meta.json']}

def run(c):
    validate(c);jobs=[]
    for cat in c['categories']:
        for split in c['splits']:
            source=Path(c['packed_root'])/cat/split
            for name in ['images.npy','meta.json']:
                if not (source/name).is_file():raise FileNotFoundError(source/name)
            jobs.append((cat,split,source))
    out=new_output(c['output'],c);seed_all(c['seed']);receipts={}
    for cat,split,source in jobs:
        dst=out/cat/split;dst.mkdir(parents=True)
        link_or_copy(source/'images.npy',dst/'images.npy')
        # Metadata will be modified: never hardlink it to a previous experiment.
        (dst/'meta.json').write_bytes((source/'meta.json').read_bytes())
        receipts[f'{cat}/{split}']=add_coordinates(dst,c)
    write_json(out/'COMPLETE.json',receipts)
