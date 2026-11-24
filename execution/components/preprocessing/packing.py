"""Pack split/class image trees into the existing training store format."""
import json
from pathlib import Path
from execution.common.io import write_json
EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}

def pack(source,destination,size,limit):
    import numpy as np
    from PIL import Image
    import torchvision.transforms as T
    source=Path(source)
    if not source.is_dir():raise FileNotFoundError(source)
    classes=sorted(p.name for p in source.iterdir() if p.is_dir());items=[]
    for label,name in enumerate(classes):
        files=sorted(p for p in (source/name).iterdir() if p.suffix.lower() in EXTENSIONS)
        if limit:files=files[:limit]
        if not files:raise ValueError(f'No images in {source/name}')
        items.extend((label,p) for p in files)
    if not items:raise ValueError(f'No class folders in {source}')
    destination.mkdir(parents=True)
    images=np.lib.format.open_memmap(destination/'images.npy',mode='w+',dtype=np.uint8,shape=(len(items),size,size,3))
    transform=T.Compose([T.Resize(size),T.CenterCrop(size)])
    for i,(_,path) in enumerate(items):
        with Image.open(path) as image:images[i]=np.asarray(transform(image.convert('RGB')))
    images.flush();del images
    write_json(destination/'meta.json',dict(classes=classes,labels=[i for i,p in items],stems=[p.stem for i,p in items],input_size=size,source_files=[str(p) for i,p in items]))
