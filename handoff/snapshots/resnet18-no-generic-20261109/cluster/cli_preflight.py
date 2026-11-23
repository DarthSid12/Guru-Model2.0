"""Read-only model-CLI/curriculum check; no model or optimizer is instantiated."""
import json
import sys
from pathlib import Path

from recipe import ROOT,SITE,CODE,INCOMING,DATA,CATEGORIES,COUNTS,EPOCHS,SEEDS,arguments
from common import atomic,sha

def locate(name):
    for base in [SITE['baseline_data'],SITE['baseline_persistent_data']]:
        p=Path(base)/name
        if p.exists():return p
    raise FileNotFoundError(name)

def main():
    DATA.mkdir(exist_ok=True)
    for name in ['faces_vgg2k_order.txt','house_identity_order.txt']:
        source=locate(name);target=DATA/name
        if target.is_symlink():assert target.resolve()==source.resolve()
        else:assert not target.exists();target.symlink_to(source)
    classes={}
    for category in [CATEGORIES[0],CATEGORIES[2]]:
        meta=json.loads(locate('fixation_data/'+category+'/train/meta.json').read_text())
        classes[category]=meta['classes']
    classes[CATEGORIES[1]]=sorted((INCOMING/'class_order.txt').read_text().splitlines())
    labels={}
    for c in CATEGORIES:
        for name in classes[c]:labels[c+'/'+name]=len(labels)
    assert len(labels)==2240 and len(classes[CATEGORIES[0]])==2048
    assert len(classes[CATEGORIES[1]])==64 and len(classes[CATEGORIES[2]])==128
    sys.path.insert(0,str(CODE))
    from training import train
    receipts=[]
    for variant in ['lp','cnn']:
        argv=arguments(variant,SEEDS[0]);sys.argv=[argv[0],*argv[1:]]
        a=train.parse_args()
        assert a.categories==CATEGORIES and a.backbone=='resnet18' and not a.pretrained
        assert a.seed==SEEDS[0] and a.variant==variant and a.num_fixations==16
        assert a.lr==.001 and a.weight_decay==.05 and a.batch_size==256
        assert a.steps_per_epoch==2000 and a.curriculum_warmup_steps==500
        stages=train.build_curriculum_stages(a,labels,CATEGORIES)
        assert len(stages)==10 and sum(s['epochs'] for s in stages)==124
        assert all(not s['image_caps'] for s in stages)
        for s,counts,e in zip(stages,COUNTS,EPOCHS):
            assert s['classes_per_category']==dict(zip(CATEGORIES,counts)) and s['epochs']==e
        receipts.append(dict(variant=variant,seed=a.seed,command=argv,stages=stages))
    atomic(ROOT/'CLI_PREFLIGHT.json',dict(passed=True,classes=2240,generic_house_classes=0,
        model_initialized=False,optimizer_steps=0,spec_sha256=sha(ROOT/'SPEC.json'),variants=receipts))
    print('Both real training CLIs and all10 stages verified; zero optimizer steps',flush=True)

if __name__=='__main__':main()
