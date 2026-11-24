from pathlib import Path
from execution.common.io import category_names,new_output,seed_all,write_json
from execution.components.salience.sampling import validate_sampling
from execution.components.salience.workflow import add_coordinates
from .packing import pack

def validate(c):
    category_names(c);validate_sampling(c['salience'])
    if c['input_size']<2 or c['limit_per_class']<0:raise ValueError('input_size >=2 and limit_per_class >=0 required')
    if c['crop_size']<2 or not set(c['render_variants'])<=set(['lp','cnn','plain']) or not set(c['render_orientations'])<=set(['U','I']):raise ValueError('Invalid fixation-rendering settings')

def run(c):
    validate(c)
    for cat in c['categories']:
        for split in c['splits']:
            if not (Path(c['raw_root'])/cat/split).is_dir():raise FileNotFoundError(Path(c['raw_root'])/cat/split)
    out=new_output(c['output'],c);seed_all(c['salience']['seed']);receipts={}
    for cat in c['categories']:
        for split in c['splits']:
            dst=out/cat/split;pack(Path(c['raw_root'])/cat/split,dst,c['input_size'],c['limit_per_class'])
            receipts[f'{cat}/{split}']=add_coordinates(dst,c['salience'])
            if c['render_fixations']:
                from .rendering import render
                receipts[f'{cat}/{split}']['rendered']=render(dst,c)
    write_json(out/'COMPLETE.json',receipts)
