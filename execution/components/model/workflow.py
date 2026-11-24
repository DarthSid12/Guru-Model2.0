"""Map explicit settings to the established, checkpoint-compatible trainer."""
import json,subprocess,sys
from pathlib import Path
from execution.common.cli import ROOT
from execution.common.io import write_json

def validate(c):
    if not c['categories'] or len(set(c['categories']))!=len(c['categories']):raise ValueError('Choose unique nonempty categories')
    if c['variant'] not in ['lp','cnn','plain']:raise ValueError('variant must be lp/cnn/plain')
    for key in ['lr','batch_size','epochs','num_fixations','temperature','checkpoint_every']:
        if c[key]<=0:raise ValueError(f'{key} must be positive')
    if c['curriculum']:
        if len(c['curriculum_stages'])!=len(c['curriculum_epochs']) or not c['curriculum_stages']:raise ValueError('Each curriculum stage needs one epoch count')
        if sum(c['curriculum_epochs'])!=c['epochs']:raise ValueError('epochs must equal the sum of curriculum_epochs. Set --no-curriculum for flat training.')
        if any(n<1 for n in c['curriculum_epochs']):raise ValueError('Curriculum epoch counts must be positive')
        for stage in c['curriculum_stages']:
            if set(stage)!=set(c['categories']) or any(n<0 for n in stage.values()):raise ValueError('Each stage must provide a nonnegative class count for every category')
        if c['curriculum_image_caps'] and len(c['curriculum_image_caps'])!=len(c['curriculum_stages']):raise ValueError('Image caps must match the stage count')
    if c['resume'] not in [None,'auto'] and not isinstance(c['resume'],str):raise ValueError('resume must be null, "auto", or a checkpoint path')

def command(c):
    args=[sys.executable,str(ROOT/'training/train.py'),'--data-mode','packed']
    skip={'output','curriculum_stages','curriculum_epochs','curriculum_pin','curriculum_image_caps','resume','curriculum','pretrained','random_fixations'}
    for k,v in c.items():
        if k in skip:continue
        flag='--'+k.replace('_','-')
        if isinstance(v,bool):args.append(flag if v else '--no-'+k.replace('_','-'))
        elif isinstance(v,list):
            if v:args.extend([flag,*map(str,v)])
        else:args.extend([flag,str(Path(v).resolve()) if k=='fixation_root' else str(v)])
    args+=['--output-dir',str(Path(c['output']).resolve())]
    if c['pretrained']:args+=['--pretrained']
    if c['random_fixations']:args+=['--random-fixations']
    if c['resume'] is not None:args+=['--resume']+([] if c['resume']=='auto' else [str(Path(c['resume']).resolve())])
    if c['curriculum']:
        args+=['--curriculum','--curriculum-stages',*[','.join(f'{k}={v}' for k,v in stage.items()) for stage in c['curriculum_stages']],'--curriculum-epochs',*map(str,c['curriculum_epochs'])]
        if c['curriculum_image_caps']:args+=['--curriculum-image-caps',*c['curriculum_image_caps']]
        if c['curriculum_pin']:args+=['--curriculum-pin',*[f'{k}=@{Path(v).resolve()}' for k,v in c['curriculum_pin'].items()]]
    return args

def run(c):
    validate(c);out=Path(c['output'])
    if out.exists() and c['resume'] is None:raise FileExistsError(f'{out} exists. Choose a new output or explicitly configure resume.')
    for category in c['categories']:
        for split in ['train','valid','test']:
            for name in ['images.npy','coords.npy','meta.json']:
                p=Path(c['fixation_root'])/category/split/name
                if not p.is_file():raise FileNotFoundError(p)
    if c['curriculum']:
        for name in c['curriculum_pin'].values():
            if not Path(name).is_file():raise FileNotFoundError(name)
    out.mkdir(parents=True,exist_ok=True)
    import datetime
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(out/f'EXECUTION_{stamp}.json',dict(config=c,command=command(c)))
    subprocess.run(command(c),cwd=ROOT,check=True)
