"""Source adapters: authorized local banks, HTTP archives, or per-image manifests."""
import json,shutil,tarfile,urllib.request,zipfile
from pathlib import Path,PurePosixPath
from execution.common.io import sha,safe_name

def fetch(location,target,expected=None):
    if location.startswith(('http://','https://')):
        with urllib.request.urlopen(location,timeout=120) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    else:
        src=Path(location.removeprefix('file://'))
        if not src.is_file():raise FileNotFoundError(f'Source file not found: {src}')
        shutil.copyfile(src,target)
    digest=sha(target)
    if expected and digest!=expected:raise ValueError(f'Checksum mismatch: {target}')
    return digest

def member_path(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError(f'Unsafe archive path: {name}')
    return p

def extract(archive,destination):
    destination.mkdir()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for x in z.infolist():
                member_path(x.filename)
                if (x.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive symlinks are not accepted.')
            z.extractall(destination)
    else:
        with tarfile.open(archive) as t:
            for x in t.getmembers():
                member_path(x.name)
                if not (x.isfile() or x.isdir()):raise ValueError('Archive links/devices are not accepted.')
            t.extractall(destination,filter='data')

def materialize(source,target,temporary):
    provider=source['provider'];location=source['location']
    if provider=='directory':
        path=Path(location)
        if not path.is_dir():raise FileNotFoundError(f'{path}: supply the authorized source bank in download config (directory, archive URL, or manifest). No substitute dataset is selected.')
        shutil.copytree(path,target)
    elif provider=='archive':
        archive=temporary/'source.archive';fetch(location,archive,source.get('sha256'))
        extracted=temporary/'extracted';extract(archive,extracted)
        inside=source.get('root_inside','');member_path(inside)
        shutil.copytree(extracted/inside,target)
    elif provider=='manifest':
        manifest=Path(location);data=json.loads(manifest.read_text());seen=set();target.mkdir()
        for item in data['images']:
            split=item['split'];label=safe_name(item['class']);filename=safe_name(item['filename'])
            if split not in ['train','valid','test']:raise ValueError(f'Invalid split {split}')
            key=(split,label,filename)
            if key in seen:raise ValueError(f'Duplicate manifest destination: {key}')
            seen.add(key);dst=target/split/label/filename;dst.parent.mkdir(parents=True,exist_ok=True)
            origin=item.get('url') or item['path']
            if not origin.startswith(('http://','https://','file://')) and not Path(origin).is_absolute():origin=str(manifest.parent/origin)
            fetch(origin,dst,item.get('sha256'))
    else:raise ValueError(f'Unknown source provider: {provider}')
