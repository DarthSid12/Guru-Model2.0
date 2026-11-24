"""Optional materialization of the complete fixation transform for inspection."""
import json
from execution.common.transforms import make_transform
from execution.common.io import sha

def render(directory,c):
    import numpy as np
    import torch
    from training.datasets import _crop_at
    images=np.load(directory/'images.npy',mmap_mode='r');coords=np.load(directory/'coords.npy',mmap_mode='r');result={}
    n,k,_=coords.shape;size=c['crop_size'];device=c['salience']['device']
    for variant in c['render_variants']:
        for orientation in c['render_orientations']:
            name=f'fixations_{variant}_{orientation}.npy';tf=make_transform(variant,orientation,device,size)
            out=np.lib.format.open_memmap(directory/name,mode='w+',dtype=np.float32,shape=(n,k,3,size,size))
            with torch.no_grad():
                for i in range(n):
                    crops=torch.stack([_crop_at(images[i],*coords[i,j],size) for j in range(k)]).to(device)
                    out[i]=tf(crops).cpu().numpy()
            out.flush();del out;result[name]=sha(directory/name)
    return result
