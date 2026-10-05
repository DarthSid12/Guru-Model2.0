"""Recover train-set evaluation points, not historical online training accuracy.

At each saved curriculum stage endpoint, evaluate a fixed deterministic sample
of up to 16 training photos per active class, using all 16 evaluation fixations.
Generic houses use all photos allowed by the stage cap. No augmentation,
dropout, model updates, or behavioral retrieval/noise simulation is applied.
For the category lacking a test split, also evaluate inverted validation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from datasets import _PackedSplit, _crop_at
from model import Model
from salience_trans import OnTheFlyTransform
from build_vgg16bn_report import MODELS

OUT=ROOT/'runs/vgg16bn_training_curve_audit_20260928'
SEED=20260928


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');os.replace(p,path)


class PhotoSample(Dataset):
    def __init__(self,cfg,labels,active,stage,split):
        self.splits={};self.items=[];self.photo_records=[]
        caps={}
        if split=='train':
            spec=cfg['curriculum_image_caps'][stage-1]
            if spec!='none':caps={k:int(v) for k,v in (x.split('=') for x in spec.split(','))}
        for cat in cfg['categories']:
            if split!='train' and cat!='houses_zubud137_41':continue
            sp=_PackedSplit(str(ROOT/'fixation_data'),cat,split);self.splits[cat]=sp
            pool=defaultdict(list)
            for i,ci in enumerate(sp.labels):
                key=f'{cat}/{sp.classes[ci]}'
                if key in labels and labels[key] in active:pool[key].append(i)
            for key,indices in sorted(pool.items()):
                if cat in caps:indices=indices[:caps[cat]]
                if split=='train' and cat!='houses':
                    rng=np.random.default_rng(int(hashlib.sha256(f'{SEED}:{key}'.encode()).hexdigest()[:16],16))
                    indices=sorted(rng.permutation(indices)[:16].tolist())
                for i in indices:
                    self.items.append((cat,i,labels[key]));self.photo_records.append(dict(category=cat,class_name=key,image_index=i,stem=sp.stems[i]))
        self.nfix=cfg['num_fixations']
    def __len__(self):return len(self.items)
    def __getitem__(self,index):
        cat,i,label=self.items[index];sp=self.splits[cat]
        crops=torch.stack([_crop_at(sp.images[i],sp.coords[i,j,0],sp.coords[i,j,1],180) for j in range(self.nfix)])
        return crops,label,cat


@torch.inference_mode()
def evaluate(model,ds,transform,active,device,cfg):
    loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=2,pin_memory=True)
    mask=torch.zeros(len(model.fc2.weight),device=device,dtype=torch.bool);mask[list(active)]=True
    counts=defaultdict(lambda:[0,0])
    for crops,labels,cats in loader:
        B,N,C,H,W=crops.shape;crops=transform(crops.to(device,non_blocking=True).reshape(-1,C,H,W))
        if cfg['channels_last']:crops=crops.contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['amp']):logits=model(crops)
        logits=logits.float().reshape(B,N,-1).sum(1).masked_fill(~mask,-float('inf'))
        hits=(logits.argmax(1).cpu()==labels).tolist()
        for cat,hit in zip(cats,hits):counts[cat][0]+=int(hit);counts[cat][1]+=1
    return {cat:dict(correct=correct,n_photos=n,accuracy_pct=100*correct/n) for cat,(correct,n) in counts.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,choices=MODELS);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED)
    run=next((ROOT/'runs').glob('*_'+MODELS[a.model]['tag']));cfg=json.loads((run/'config.json').read_text());labels=json.loads((run/'label_map.json').read_text())
    out=OUT/a.model;out.mkdir(parents=True,exist_ok=True)
    source_hash={f:sha(ROOT/f) for f in ('scripts/evaluate_vgg_training_checkpoints.py','model.py','datasets.py','salience_trans.py','trans.py')}
    signature=dict(model=a.model,tag=MODELS[a.model]['tag'],run_dir=str(run),sample_seed=SEED,photos_per_class=16,generic_houses='all stage-eligible photos',fixations=16,training_transform='upright, no augmentation',model_mode='eval',decision='sum fixation logits, argmax over active classes',source_sha256=source_hash)
    manifest=out/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==signature,'Changed evaluation settings'
    else:atomic(manifest,signature)
    device=torch.device(a.device)
    model=Model(num_classes=len(labels),pretrained=False,T=cfg['temperature'],dropout=cfg['dropout'],backbone=cfg['backbone']).to(device).eval()
    if cfg['channels_last']:model=model.to(memory_format=torch.channels_last)
    transforms={s:OnTheFlyTransform(s,cfg['variant'],device).to(device) for s in ('valid','test')}
    epoch=0
    for stage,epochs in enumerate(cfg['curriculum_epochs'],1):
        epoch+=epochs;path=out/f'stage{stage}.json'
        if path.exists():print(f'Already evaluated {a.model} stage {stage}',flush=True);continue
        t=time.time();active_meta=json.loads((run/f'stage{stage}_active_ids.json').read_text());active=set(active_meta['active_ids'])
        checkpoint=next(run.glob(f'stage{stage}_*cls.pth'));model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        data=PhotoSample(cfg,labels,active,stage,'train');train=evaluate(model,data,transforms['valid'],active,device,cfg)
        inv_data=PhotoSample(cfg,labels,active,stage,'valid');inverted=evaluate(model,inv_data,transforms['test'],active,device,cfg) if len(inv_data) else {}
        result=dict(stage=stage,epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),active_classes=len(active),train_eval=train,inverted_validation=inverted,train_photo_sample=data.photo_records,elapsed_sec=time.time()-t)
        atomic(path,result)
        print(json.dumps({k:v for k,v in result.items() if k!='train_photo_sample'}),flush=True)
    atomic(out/'DONE.json',dict(stages=10,epochs=epoch))

if __name__=='__main__':main()
