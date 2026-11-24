"""One checkpoint-to-probability path for both behavioral experiments."""
import json
from pathlib import Path
from .io import sha,write_json

def load_probabilities(model,banks,c,fixations):
    import numpy as np
    import torch
    root=Path(c['cache_root'])/model['cache_subdir']
    if not root.exists():
        if not model.get('run_dir'):raise FileNotFoundError(f'{root}: provide existing probability caches, or set models[].run_dir to a trained run for inference.')
        build_cache(model,banks,c,fixations,root)
    receipt=root/'DONE.json'
    if not receipt.is_file():raise FileNotFoundError(f'Missing cache completion receipt: {receipt}')
    done=json.loads(receipt.read_text())
    recorded_variant=done.get('variant') or (done.get('model',{}).get('variant') if isinstance(done.get('model'),dict) else None)
    if recorded_variant and recorded_variant!=model['variant']:raise ValueError('Cache model variant differs from the requested variant.')
    for name,digest in done.get('files',{}).items():
        if sha(root/name)!=digest:raise ValueError(f'Cache hash mismatch: {root/name}')
    if model.get('run_dir'):
        checkpoint=Path(model['run_dir'])/'checkpoint_last.pth';expected=done.get('checkpoint_sha256') or done.get('model',{}).get('sha256')
        if not expected or sha(checkpoint)!=expected:raise ValueError('Cache checkpoint provenance does not match the requested run.')
    arrays={}
    for cat,bank in banks.items():
        for orientation in ['U','I']:
            path=root/f'{bank}_{orientation}.npy'
            if not path.exists() and len(banks)==1:path=root/f'{orientation}.npy'
            if path.name not in done.get('files',{}):raise ValueError(f'Cache array is absent from its completion receipt: {path}')
            a=np.load(path)
            if a.dtype!=np.float32 or a.ndim!=3 or a.shape[1]<fixations or a.shape[2]!=256 or not np.isfinite(a).all() or not ((a>=0)&(a<=1)).all():raise ValueError(f'Invalid probabilities: {path}')
            arrays[cat,orientation]=torch.from_numpy(a)
    for cat in banks:
        if arrays[cat,'U'].shape!=arrays[cat,'I'].shape:raise ValueError(f'Upright/inverted cache shapes differ for {cat}')
    return arrays,done

def build_cache(record,banks,c,fixations,out):
    import numpy as np
    import torch
    from training.model import Model
    from training.datasets import _PackedSplit,_crop_at
    from execution.common.transforms import make_transform
    run=Path(record['run_dir']);cfg=json.loads((run/'config.json').read_text());labels=json.loads((run/'label_map.json').read_text());checkpoint=run/'checkpoint_last.pth'
    if cfg['variant']!=record['variant']:raise ValueError('Requested variant differs from training config.')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=Model(size=c['crop_size'],num_classes=len(labels),pretrained=False,T=cfg.get('temperature',2),backbone=cfg['backbone'])
    model.load_state_dict(ck['model'],strict=True);model.to(c['device']).eval();model.stochastic=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out.mkdir(parents=True,exist_ok=False);files={};metadata={}
    with torch.no_grad():
        for cat,bank in banks.items():
            sp=_PackedSplit(c['packed_root'],bank,'valid')
            if sp.num_coords<fixations:raise ValueError(f'{bank} needs at least {fixations} stored coordinates')
            metadata[cat]=dict(labels=sp.labels.tolist() if hasattr(sp.labels,'tolist') else list(sp.labels),classes=sp.classes,packed_files={n:sha(Path(c['packed_root'])/bank/'valid'/n) for n in ['images.npy','coords.npy','meta.json']})
            for orient,phase in [('U','valid'),('I','test')]:
                tf=make_transform(record['variant'],orient,c['device'],c['crop_size']);values=[]
                for row in range(len(sp.labels)):
                    crops=torch.stack([_crop_at(sp.images[row],*sp.coords[row,j],c['crop_size']) for j in range(fixations)]).to(c['device'])
                    _,h,p=model(tf(crops),return_rep=True)
                    if not torch.equal(h,p):raise ValueError('Inference must expose deterministic sigmoid probabilities')
                    values.append(p.cpu().numpy())
                name=f'{bank}_{orient}.npy';np.save(out/name,np.stack(values));files[name]=sha(out/name)
    write_json(out/'METADATA.json',metadata);files['METADATA.json']=sha(out/'METADATA.json')
    write_json(out/'DONE.json',dict(checkpoint_sha256=sha(checkpoint),variant=record['variant'],epoch=ck.get('epoch'),training_seed=cfg.get('seed'),files=files,fixations=fixations,crop_size=c['crop_size'],torch=torch.__version__,device=c['device']))
