"""Allocated GPU preflight for the unchanged diet minus generic houses."""
import json,os,subprocess,sys,tarfile
from pathlib import Path
from common import atomic,sha,identity
from recipe import ROOT,SITE,SPEC,CODE,OPS,INCOMING,DATA,PYTHON,SEEDS,CATEGORIES,COUNTS,EPOCHS,arguments
def locate(name):
 for base in [SITE['baseline_data'],SITE['baseline_persistent_data']]:
  p=Path(base)/name
  if p.exists():return p
 raise FileNotFoundError(name)
def main():
 import numpy as np,torch
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
 assert CATEGORIES==['faces_vgg2k','objects_wordbank64','houses_synthetic128']
 DATA.mkdir(exist_ok=True);(DATA/'fixation_data').mkdir(exist_ok=True)
 expected_pack=json.loads((INCOMING/'OBJECTS_PACK_READY.json').read_text())
 exported=json.loads((INCOMING/'PACK_EXPORT.json').read_text())
 assert expected_pack['manifest']==exported['manifest']
 if SITE['object_store']:
  target=DATA/'fixation_data/objects_wordbank64'
  source=Path(SITE['object_store']);assert source.exists()
  if target.is_symlink():assert target.resolve()==source.resolve()
  else:assert not target.exists();target.symlink_to(source,target_is_directory=True)
  pack=expected_pack
  atomic(DATA/'OBJECTS_PACK_READY.json',pack)
 else:
  archive=INCOMING/'object-bank-packed.tar.gz'
  assert sha(archive)==exported['archive_sha256']==SPEC['object_pack_archive_sha256']
  allowed=set(exported['manifest'])|{'OBJECTS_PACK_READY.json'}
  with tarfile.open(archive) as t:
   members=t.getmembers();assert {m.name for m in members}==allowed
   assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
   t.extractall(DATA)
  assert sha(DATA/'OBJECTS_PACK_READY.json')==exported['pack_receipt_sha256']
  pack=json.loads((DATA/'OBJECTS_PACK_READY.json').read_text())
 assert pack==expected_pack
 assert pack['source_photo_manifest_sha256']==SPEC['photo_manifest_sha256']
 assert pack['source_curriculum_sha256']==SPEC['curriculum_sha256']
 assert pack['class_order_sha256']==SPEC['object_order_sha256'] and pack['pack_seed']==SPEC['pack_seed']
 baseline=json.loads((INCOMING/'BASELINE_EXPECTED.json').read_text())['manifest']
 for cat in [CATEGORIES[0],CATEGORIES[2]]:
  target=DATA/'fixation_data'/cat;source=locate('fixation_data/'+cat+'/train/meta.json').parent.parent
  if target.is_symlink():assert target.resolve()==source.resolve()
  else:assert not target.exists();target.symlink_to(source,target_is_directory=True)
 for name in ['faces_vgg2k_order.txt','house_identity_order.txt']:
  source=locate(name);target=DATA/name
  if target.is_symlink():assert target.resolve()==source.resolve()
  else:target.symlink_to(source)
 assert not (DATA/'fixation_data/houses_generic_clean').exists()
 combined={n:h for n,h in baseline.items() if n.startswith(('fixation_data/faces_vgg2k/','fixation_data/houses_synthetic128/')) or n in ['faces_vgg2k_order.txt','house_identity_order.txt']}
 combined.update(pack['manifest']);assert len(combined)==29
 verified={}
 for name,digest in combined.items():
  path=DATA/name;assert sha(path)==digest,'Dataset changed: '+name;verified[name]=identity(path)
  print('VERIFIED',name,flush=True)
 order=(INCOMING/'class_order.txt').read_text().splitlines();assert len(order)==len(set(order))==64
 budgets={'faces_vgg2k':(2048,{'train':200,'valid':12,'test':12}),'objects_wordbank64':(64,{'train':500,'valid':100,'test':100}),'houses_synthetic128':(128,{'train':3,'valid':1,'test':1})}
 for cat in CATEGORIES:
  for split in ['train','valid','test']:
   folder=DATA/'fixation_data'/cat/split;meta=json.loads((folder/'meta.json').read_text());im=np.load(folder/'images.npy',mmap_mode='r');co=np.load(folder/'coords.npy',mmap_mode='r')
   assert im.dtype==np.uint8 and im.shape[1:]==(224,224,3) and co.dtype==np.int16 and co.shape==(len(im),32,2) and co.min()>=0 and co.max()<224
   classes,budget=budgets[cat]
   assert len(meta['classes'])==classes and set(__import__('collections').Counter(meta['labels']).values())=={budget[split]}
   if cat=='objects_wordbank64':assert set(meta['classes'])==set(order)
 sys.path.insert(0,str(CODE));from training import train
 from training.datasets import build_packed_label_map
 labels=build_packed_label_map(str(DATA/'fixation_data'),CATEGORIES,split='train')
 assert len(labels)==2240 and sum(EPOCHS)==124 and not any('generic' in n for n in labels)
 reverse={v:k for k,v in labels.items()}
 for seed in SEEDS:
  for variant in ['lp','cnn']:
   argv=arguments(variant,seed);sys.argv=[argv[0],*argv[1:]];parsed=train.parse_args()
   assert parsed.categories==CATEGORIES and parsed.backbone=='resnet18' and not parsed.pretrained and parsed.seed==seed and parsed.variant==variant and parsed.num_fixations==16
   stages=train.build_curriculum_stages(parsed,labels,CATEGORIES);assert len(stages)==10
   for stage,counts,epochs in zip(stages,COUNTS,EPOCHS):
    assert stage['classes_per_category']==dict(zip(CATEGORIES,counts)) and stage['epochs']==epochs
    names={reverse[i] for i in stage['active_ids']};assert {n.split('/',1)[1] for n in names if n.startswith(CATEGORIES[1]+'/')}==set(order[:counts[1]])
   atomic(ROOT/f'curriculum-{variant}-s{seed}-validated.json',stages)
   subprocess.run([PYTHON,'-u',str(OPS/'smoke.py'),variant,str(seed)],cwd=ROOT,check=True)
 atomic(ROOT/'DATA_VERIFIED.json',{'manifest':combined,'files':verified,'object_pack_receipt_sha256':sha(DATA/'OBJECTS_PACK_READY.json'),'photo_manifest_sha256':SPEC['photo_manifest_sha256'],'class_order_sha256':SPEC['object_order_sha256']})
 atomic(ROOT/'CONFIGURATION_PREFLIGHT.json',{'backbone':'resnet18','seed':SITE['seed'],'pretrained':False,'classes':2240,'epochs':124,'generic_house_images':0,'generic_house_classes':0,'both_optimizer_resume_smokes_passed':True,'updated_object_order_validated':True})
 print('NO-GENERIC-HOUSE DATA, CURRICULUM AND BOTH MODEL SMOKES PASSED',flush=True)
if __name__=='__main__':main()
