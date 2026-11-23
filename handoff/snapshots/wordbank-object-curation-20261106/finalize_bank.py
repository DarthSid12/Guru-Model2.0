"""Create a new source bank, replacing flagged photos and preserving split counts."""
import collections,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'wordbank-inanimate-20261105'
def write(name,obj):(ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def rank(r,prefix):return hashlib.sha256((prefix+'|'+r['image_id']).encode()).digest()
def main():
 plan=json.loads((ROOT/'SOURCE_PLAN.json').read_text());parentfile=PARENT/'PHOTO_SPLITS.json';parent=json.loads(parentfile.read_text())
 assert hashlib.sha256(parentfile.read_bytes()).hexdigest()==plan['parent_manifest_sha256']
 freshfile=ROOT/'FRESH_SOURCES.json';fresh=json.loads(freshfile.read_text());assert fresh['source_plan_sha256']==hashlib.sha256((ROOT/'SOURCE_PLAN.json').read_bytes()).hexdigest()
 rows=fresh['images'];byclass=collections.defaultdict(list)
 for r in rows:byclass[r['class_name']].append(r)
 assert {k:len(v) for k,v in byclass.items()}==plan['new_targets']
 animalflags=json.loads((ROOT/'ANIMAL_SOURCE_FLAGS.json').read_text())['parent_flags'];overrides={r['image_id'] for r in json.loads((ROOT/'PHYSICAL_TOY_OVERRIDES.json').read_text())['images']}
 flagids={r['image_id'] for r in animalflags}-overrides
 quarantine=[{**r,'quarantine_reason':'Broad glass source replaced with boxed drinking-glass instances' if r['class_name']=='glass' else 'Nonhuman-animal source-label flag; conservatively excluded'} for r in parent['images'] if r['class_name']=='glass' or r['image_id'] in flagids]
 kept=[{**r,'file_verified_in_this_operation':False,'inherited_source_manifest_sha256':plan['parent_manifest_sha256']} for r in parent['images'] if r['class_name']!='glass' and r['image_id'] not in flagids]
 needed=collections.Counter((r['class_name'],r['split']) for r in quarantine)
 active=kept[:];reserves=[]
 for cl in sorted(byclass):
  n=sum(needed[(cl,split)] for split in ['train','valid','test'])
  ordered=sorted(byclass[cl],key=lambda r:rank(r,'curation-primary-20261106'))
  mainrows=sorted(ordered[:n],key=lambda r:rank(r,'curation-split-20261106'))
  extra=ordered[n:];assert len(extra)==(100 if cl in json.loads((ROOT/'phase1/SOURCE_PLAN.json').read_text())['new_targets'] else 0)
  start=0
  for split in ['train','valid','test']:
   count=needed[(cl,split)]
   for r in mainrows[start:start+count]:active.append({**r,'split':split,'image_path':str(ROOT/r['normalized_path']),'file_verified_in_this_operation':True,'new_in_curation_pass':True})
   start+=count
  reserves.extend({**r,'image_path':str(ROOT/r['normalized_path']),'reserve_only':True,'split':None} for r in extra)
 assert len(active)==44800 and len(kept)==42355 and len(quarantine)==2445 and len(reserves)==1600
 expected={ROOT/'raw/objects'/r['split']/r['class_name']/(r['image_id']+'.png') for r in active}
 for old in (ROOT/'raw/objects').glob('*/*/*.png'):
  if old not in expected:assert old.is_symlink();old.unlink()
 for r in active:
  path=Path('raw/objects')/r['split']/r['class_name']/(r['image_id']+'.png');link=ROOT/path;source=Path(r['image_path']);link.parent.mkdir(parents=True,exist_ok=True)
  if link.is_symlink():assert link.resolve()==source.resolve()
  else:link.symlink_to(os.path.relpath(source,link.parent))
  r['split_relative_path']=str(path)
 for r in reserves:
  link=ROOT/'raw/reserve'/r['class_name']/(r['image_id']+'.png');link.parent.mkdir(parents=True,exist_ok=True)
  if not link.is_symlink():link.symlink_to(os.path.relpath(r['image_path'],link.parent))
 write('PHOTO_SPLITS.json',{'status':'SOURCE_FILTERED_COUNTS_COMPLETE_REVIEW_NOT_EXHAUSTIVE','parent_manifest_sha256':plan['parent_manifest_sha256'],'word_curriculum_sha256':plan['parent_curriculum_sha256'],'counts_per_class':{'train':500,'valid':100,'test':100},'seed':20261020,'retained_parent_photos':len(kept),'replaced_parent_photos':len(quarantine),'images':active,'training_ready':False})
 write('RESERVE_PHOTOS.json',{'reserve_only':True,'not_in_training_or_evaluation_splits':True,'photos_per_new_replacement_class':100,'images':reserves})
 write('SUPERSEDED_PARENT_PHOTOS.json',{'original_files_preserved':True,'images':quarantine})
 print('New primary bank:',len(active),'images; reserves:',len(reserves),'parent photos superseded:',len(quarantine),flush=True)
if __name__=='__main__':main()
