"""Verify inherited stimuli/source and completed model, then freeze a new spec."""
import json, shutil
from pathlib import Path
from common import ROOT,sha,atomic,SOURCES,POOL_SIZES,SEEDS,DESIGNS,PROTOCOLS,CONDITIONS
PARENT=Path('/expanse/lustre/projects/sdp159/sagrawal5/yin-unseen-houses-20261103')
def main():
 assert str(ROOT)=='/expanse/lustre/projects/sdp159/sagrawal5/yin-no-generic-s46-20261111' and not (ROOT/'SPEC.json').exists()
 parent=json.loads((ROOT/'PARENT_SPEC.json').read_text())
 assert sha(PARENT/'SPEC.json')==sha(ROOT/'PARENT_SPEC.json')
 assert sha(ROOT/'PARENT_SPEC.json')=='6ad58e5619a697bcfc6a617c78c37190b2d72f37d8506b1c296416818a0483ae'
 reg=json.loads((ROOT/'TRAINING_REGISTRATION.json').read_text())
 assert reg['source_commit']==parent['source_commit'] and reg['seeds']==[46]
 assert reg['backbone']=='resnet18' and reg['fresh_initialization'] and not reg['pretrained']
 # Keep exact completed evaluation source, verified against new training registration.
 (ROOT/'source').mkdir(exist_ok=True)
 for key,digest in parent['files'].items():
  if key.startswith('source/'):
   src=PARENT/key;dst=ROOT/key;assert sha(src)==digest,key
   name=key[len('source/'):];assert reg['frozen']['source_files'][name]==digest,name
   dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 assert sha(ROOT/'STIMULUS_PLANS.json')==parent['files']['STIMULUS_PLANS.json']
 for name in ['engine.py','design.py','test_science.py','common.py']:
  assert sha(ROOT/name)==parent['files'][name],name
 receipt=json.loads((ROOT/'PARENT_PACKED_HOUSES.json').read_text())
 assert receipt['spec_sha256']==sha(ROOT/'PARENT_SPEC.json')
 for category,packed in SOURCES.items():
  for name in ['images.npy','coords.npy','meta.json']:
   key='eval-data/'+packed+'/valid/'+name;src=PARENT/key;dst=ROOT/key
   digest=receipt['files'][key] if category=='houses' else parent['files'][key]
   assert sha(src)==digest,key
   dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
   assert sha(dst)==digest,key
  meta=json.loads((ROOT/'eval-data'/packed/'valid/meta.json').read_text())
  assert len(meta['labels'])==len(set(meta['labels']))==64
 models=[];first_labels=None
 for variant in ['lp','cnn']:
  name='resnet18_'+variant+'_nogeneric_e124_s46';folder=ROOT/'models'/name
  complete=json.loads((folder/'COMPLETE').read_text());summary=json.loads((folder/'summary.json').read_text())
  cfg=json.loads((folder/'config.json').read_text());labels=json.loads((folder/'label_map.json').read_text())
  assert complete['provenance']['registration_sha256']==sha(ROOT/'TRAINING_REGISTRATION.json')
  assert complete['artifacts']['epochs']==summary['results']['epochs_run']==124
  assert sha(folder/'checkpoint_last.pth')==complete['artifacts']['checkpoint_sha256']
  assert sha(folder/'summary.json')==complete['artifacts']['summary_sha256']
  assert cfg['seed']==46 and cfg['variant']==variant and cfg['backbone']=='resnet18' and not cfg['pretrained']
  assert cfg['categories']==['faces_vgg2k','objects_wordbank64','houses_synthetic128']
  assert summary['num_classes']==len(labels)==2240 and not any('generic' in k for k in labels)
  assert complete['artifacts']['checkpoint_sha256']=={'lp':'cf516526e48eed6b6769fef3faf85b94cf3192527b29c38a988d36a3825a05ea','cnn':'e39d54ad778c50df1c0a4820836762916c73f2bac702f13c3ef50321f50b90ad'}[variant]
  if first_labels is None:first_labels=labels
  else:assert labels==first_labels
  models.append(dict(model=name,label='Delta no-generic ResNet18 '+variant.upper(),variant=variant,backbone='resnet18',epoch=124,training_seed=46,run_dir='models/'+name,sha256=complete['artifacts']['checkpoint_sha256'],training_registration_sha256=sha(ROOT/'TRAINING_REGISTRATION.json'),source_run='/projects/bjaa/sagrawal5/guru-resnet18-no-generic-20261109/runs/resnet18_'+variant+'_objects64clean_h128_nogeneric_s46',training_output_classes=2240,generic_house_training_images=0))
 object_meta=json.loads((ROOT/'eval-data/objects_wordbank64/valid/meta.json').read_text())
 trained={k.split('/',1)[1] for k in first_labels if k.startswith('objects_wordbank64/')}
 classes=object_meta['classes']
 atomic(ROOT/'OBJECT_CATEGORY_OVERLAP.json',dict(training_object_categories=sorted(trained),evaluation_classes=classes,evaluation_categories_absent_from_training_names=sorted(set(classes)-trained),note='Prior fixed object bank retained for comparison; lexical category overlap is not image identity overlap.'))
 s=dict(schema=4,remote_root=str(ROOT),models=models,source_commit=reg['source_commit'],external_files={},data_roots={v:str(ROOT/'eval-data') for v in SOURCES.values()},pool_sizes=POOL_SIZES,seeds=list(SEEDS),protocols=list(PROTOCOLS),conditions=list(CONDITIONS),designs=DESIGNS,study_fixations=10,test_fixations=32,sigma=2,expected_rows=14400,checkpoint_selection='Both final epoch124 checkpoint_last files; both best epochs also124',parent_spec_sha256=sha(ROOT/'PARENT_SPEC.json'),house_pool=parent['house_pool'],object_bank_note='Prior fixed evaluation bank retained, including16 categories removed from revised training diet',noise_calibration='Fresh per-model face UU/II fits by face-containing design; face40 transfer to designs without faces',training_generic_house_images=0,training_output_classes=2240)
 excluded={'__pycache__','.git','logs','cache','results','calibration'}
 s['files']={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and not any(n in p.parts for n in excluded) and p.name not in ['SPEC.json','JOBS.json','.submission.lock','stage.tar','full-stage.tar']}
 atomic(ROOT/'SPEC.json',s);print('FROZEN',sha(ROOT/'SPEC.json'),flush=True)
if __name__=='__main__':main()
