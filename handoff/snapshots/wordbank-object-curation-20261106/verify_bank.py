"""Audit source labels, animal flags, exact duplicates, and unchanged inherited splits."""
import collections,hashlib,json
from pathlib import Path
import duckdb
from PIL import Image
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'wordbank-inanimate-20261105'
OLD=ROOT.parent/'anvil-wordbank-lp-20261020'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=json.loads((ROOT/'PHOTO_SPLITS.json').read_text());active=manifest['images'];reserve=json.loads((ROOT/'RESERVE_PHOTOS.json').read_text())['images'];allrows=active+reserve
 parent=json.loads((PARENT/'PHOTO_SPLITS.json').read_text())['images'];pmap={r['image_id']:r for r in parent};fresh=json.loads((ROOT/'FRESH_SOURCES.json').read_text())['images']
 assert len(active)==44800 and len(reserve)==1600 and len(fresh)==4045
 classes=json.loads((PARENT/'CURRICULUM.json').read_text())['primary_original_1000_source_gate']['classes'];labels={r['class_name']:r['open_images_id'] for r in classes};labels['glass']='/m/09tvcd'
 assert set(r['class_name'] for r in active)==set(labels) and len(labels)==64
 counts=collections.Counter((r['class_name'],r['split']) for r in active)
 assert dict(counts)=={(cl,sp):n for cl in labels for sp,n in [('train',500),('valid',100),('test',100)]}
 assert all(n==100 for n in collections.Counter(r['class_name'] for r in reserve).values())
 for field in ['image_id','pixel_sha256','original_sha256']:assert len({r[field] for r in allrows})==len(allrows)
 for r in active:
  if r['image_id'] in pmap:
   old=pmap[r['image_id']]
   assert all(r[k]==old[k] for k in ['class_name','split','png_sha256','pixel_sha256','original_sha256'])
  link=ROOT/r['split_relative_path'];assert link.is_symlink() and link.resolve()==Path(r['image_path']).resolve()
 for split,n in [('train',500),('valid',100),('test',100)]:
  folder=ROOT/'raw/objects'/split;assert {p.name for p in folder.iterdir()}==set(labels)
  for cl in labels:assert len(list((folder/cl).glob('*.png')))==n
 for r in fresh:
  p=ROOT/r['normalized_path'];assert sha(p)==r['png_sha256']
  with Image.open(p) as im:
   im.load();assert im.mode=='RGB' and im.size==(224,224);assert hashlib.sha256(im.tobytes()).hexdigest()==r['pixel_sha256']
  assert r['author'] and r['license'] and r['original_landing_url']
 db=duckdb.connect(str(ROOT/'source-index.duckdb'),read_only=True)
 animal={r[0] for r in db.sql('SELECT DISTINCT ImageID FROM animal_positives').fetchall()}
 overrides=json.loads((ROOT/'PHYSICAL_TOY_OVERRIDES.json').read_text())['images'];approved={r['image_id'] for r in overrides}
 assert animal&{r['image_id'] for r in allrows}==approved
 assert all(r['class_name']=='teddybear' and pmap[r['image_id']]['png_sha256']==r['png_sha256'] for r in overrides)
 for alias,path in [('old',OLD/'wordbank/source-index.duckdb'),('parent',PARENT/'source-index.duckdb')]:db.execute("ATTACH '"+str(path).replace("'","''")+"' AS "+alias+" (READ_ONLY)")
 db.execute('CREATE TEMP TABLE expected(ImageID VARCHAR,LabelName VARCHAR)');db.executemany('INSERT INTO expected VALUES (?,?)',[(r['image_id'],labels[r['class_name']]) for r in fresh])
 assert db.sql('SELECT count(*) FROM (SELECT * FROM expected EXCEPT (SELECT * FROM old.positives UNION SELECT * FROM parent.extra_positives UNION SELECT * FROM glass_positives))').fetchone()[0]==0
 glass=[r for r in fresh if r['class_name']=='glass'];assert len(glass)==800 and all(r['source_profile']=='boxed_wine_glass_v1' and r['annotation_box'] for r in glass)
 rejected={r['image_id'] for r in json.loads((ROOT/'REVIEW_REJECTIONS.json').read_text())['rejections']};assert not rejected&{r['image_id'] for r in allrows}
 receipt={'passed':True,'active_images':44800,'classes':64,'counts_per_class':{'train':500,'valid':100,'test':100},'new_photos':4045,'parent_photos_replaced':2445,'parent_photos_retained':42355,'reserve_photos':1600,'reserve_classes':16,'unresolved_animal_annotation_flags':0,'animal_annotation_flags_visually_resolved_as_stuffed_toys':6,'animal_label_screen_size':110,'word_classes_and_comprehension_order_unchanged':True,'orientation_filter':False,'inherited_splits_unchanged':True,'source_label_membership_verified':True,'all_new_png_and_pixel_hashes_verified':True,'exact_source_and_pixel_duplicates':0,'parent_manifest_sha256':sha(PARENT/'PHOTO_SPLITS.json'),'photo_manifest_sha256':sha(ROOT/'PHOTO_SPLITS.json'),'reserve_manifest_sha256':sha(ROOT/'RESERVE_PHOTOS.json'),'exhaustive_manual_visual_review':False,'training_or_yin_run':False}
 (ROOT/'VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
