"""Independently audit the final photo/split inventory against old data and new labels."""
import collections,hashlib,json
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'anvil-wordbank-lp-20261020'
def main():
 c=json.loads((ROOT/'CURRICULUM.json').read_text());classes={r['class_name']:r for r in c['primary_original_1000_source_gate']['classes']}
 result=json.loads((ROOT/'PHOTO_SPLITS.json').read_text());records=result['images'];old=json.loads((BASE/'PHOTO_SPLITS.json').read_text())['images'];oldbyid={r['image_id']:r for r in old}
 assert len(classes)==64 and len(records)==44800
 assert all(r['semantic_type']=='inanimate_physical_object' for r in classes.values())
 assert all(r['word'] not in c['removed_old_animal_words'] for r in classes.values())
 assert len({r['image_id'] for r in records})==len({r['pixel_sha256'] for r in records})==len({r['original_sha256'] for r in records})==44800
 counts=collections.Counter((r['class_name'],r['split']) for r in records)
 expected={(cl,sp):n for cl in classes for sp,n in [('train',500),('valid',100),('test',100)]}
 assert dict(counts)==expected
 rejected={r['image_id'] for r in json.loads((ROOT/'REVIEW_REJECTIONS.json').read_text())['rejections']}
 assert not rejected&{r['image_id'] for r in records}
 reused=[r for r in records if r['reuse_original_split']] if all('reuse_original_split' in r for r in records) else [r for r in records if r.get('reuse_original_split')]
 added=[r for r in records if not r.get('reuse_original_split')]
 assert len(reused)==33600 and len(added)==11200
 assert not {r['image_id'] for r in added}&set(oldbyid)
 for r in reused:
  prior=oldbyid[r['image_id']]
  assert (r['class_name'],r['split'],r['pixel_sha256'],r['png_sha256'])==(prior['class_name'],prior['split'],prior['pixel_sha256'],prior['png_sha256'])
 for split,n in [('train',500),('valid',100),('test',100)]:
  folder=ROOT/'raw/objects'/split
  assert {p.name for p in folder.iterdir()}==set(classes)
  for cl in classes:assert len(list((folder/cl).glob('*.png')))==n
 for r in records:
  assert r.get('reuse_original_split') or r['file_verified_in_this_operation']
  link=ROOT/r['split_relative_path'];assert link.is_symlink() and link.resolve()==Path(r['image_path']).resolve()
  assert r.get('license') and r.get('author') and r.get('original_landing_url')
 db=duckdb.connect(str(ROOT/'source-index.duckdb'),read_only=True)
 db.execute("ATTACH '"+str(BASE/'wordbank/source-index.duckdb').replace("'","''")+"' AS old (READ_ONLY)")
 db.execute('CREATE TEMP TABLE expected_labels(ImageID VARCHAR,LabelName VARCHAR)')
 db.executemany('INSERT INTO expected_labels VALUES (?,?)',[(r['image_id'],classes[r['class_name']]['open_images_id']) for r in added])
 missing=db.sql('SELECT count(*) FROM (SELECT * FROM expected_labels EXCEPT (SELECT * FROM old.positives UNION SELECT * FROM extra_positives))').fetchone()[0]
 assert missing==0
 receipt={'passed':True,'photos':44800,'classes':64,'retained_classes':48,'replacement_classes':16,'retained_photo_splits_unchanged':True,'replacements_disjoint_from_all_prior_photo_ids':True,'exact_source_id_source_bytes_pixel_hash_disjoint':True,'photos_per_class':{'train':500,'valid':100,'test':100},'all_added_photos_have_correct_human_positive_source_label':True,'source_identities_not_certified_instance_disjoint':True,'curriculum_sha256':hashlib.sha256((ROOT/'CURRICULUM.json').read_bytes()).hexdigest(),'photo_splits_sha256':hashlib.sha256((ROOT/'PHOTO_SPLITS.json').read_bytes()).hexdigest(),'new_photo_pixel_and_file_hashes_verified_by_fetcher':True,'retained_photo_hashes_from_pinned_previously_verified_manifest':True,'retained_current_file_bytes_rehashed':False,'training_or_yin_not_launched':True,'image_by_image_manual_visual_review':False}
 (ROOT/'PHOTO_MANIFEST_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
