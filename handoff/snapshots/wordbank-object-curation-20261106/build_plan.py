"""Add reserves and replace the broad glass source with boxed physical wine glasses."""
import collections,gzip,hashlib,json,math
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'wordbank-inanimate-20261105'
OLD=ROOT.parent/'anvil-wordbank-lp-20261020'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parent=json.loads((PARENT/'PHOTO_SPLITS.json').read_text());old=json.loads((OLD/'PHOTO_SPLITS.json').read_text())
 banned={r['image_id'] for r in parent['images']}|{r['image_id'] for r in old['images']}
 rejected={r['image_id'] for r in json.loads((PARENT/'REVIEW_REJECTIONS.json').read_text())['rejections']}
 banned|=rejected
 filter_db=duckdb.connect(str(ROOT/'source-index.duckdb'),read_only=True)
 animal_flagged={r[0] for r in filter_db.sql('SELECT DISTINCT ImageID FROM animal_positives').fetchall()}
 filter_db.close()
 banned|=animal_flagged
 attempted=collections.defaultdict(set)
 for line in (PARENT/'DOWNLOAD_JOURNAL.jsonl').open():
  r=json.loads(line);attempted[r['class_name']].add(r['image_id'])
 prior=json.load(gzip.open(PARENT/'image_candidates.json.gz','rt'))
 candidates={cl:[r for r in rows if r['image_id'] not in banned] for cl,rows in prior.items() if cl!='glass'}
 db=duckdb.connect(str(ROOT/'source-index.duckdb'));db.execute("SET threads=4")
 rows=db.sql("SELECT p.ImageID,b.box FROM glass_positives p JOIN glass_boxes b USING(ImageID,LabelName) WHERE p.LabelName='/m/09tvcd'").fetchall()
 strict=[{'image_id':iid,'box':box,'url':'https://open-images-dataset.s3.amazonaws.com/train/'+iid+'.jpg','verified_positive_label':'/m/09tvcd','verified_positive_label_name':'Wine glass','matching_non_depiction_target_box_required':True} for iid,box in rows if iid not in banned]
 strict.sort(key=lambda r:hashlib.sha256(('object-curation-20261106|'+r['image_id']).encode()).digest())
 candidates['glass']=strict
 # Assign each unused photograph once; protect the scarcer source classes first.
 owners={}
 for cl,rows in candidates.items():
  for r in rows:
   iid=r['image_id'];key=(len(rows),hashlib.sha256(('object-curation-owner-v1|'+iid+'|'+cl).encode()).digest())
   if iid not in owners or key<owners[iid][0]:owners[iid]=(key,cl)
 for cl,rows in candidates.items():
  rows=[r for r in rows if owners[r['image_id']][1]==cl]
  if cl!='glass':rows.sort(key=lambda r:r['image_id'] in attempted[cl]) # stable: untried sources before retrying failures
  candidates[cl]=rows
 wanted=[r['image_id'] for r in candidates['glass']]
 db.execute('CREATE OR REPLACE TEMP TABLE wanted(ImageID VARCHAR PRIMARY KEY)');db.executemany('INSERT INTO wanted VALUES (?)',[(v,) for v in wanted])
 infofile=ROOT.parents[1]/'tmp/oidv6-train-images-with-labels-with-rotation.csv'
 db.execute("CREATE OR REPLACE TABLE strict_glass_image_info AS SELECT i.ImageID,i.OriginalURL,i.Thumbnail300KURL,i.OriginalLandingURL,i.License,i.Author,try_cast(i.Rotation AS DOUBLE) AS Rotation,try_cast(i.OriginalSize AS BIGINT) AS OriginalSize FROM read_csv_auto(?,header=true,all_varchar=true) i INNER JOIN wanted w USING(ImageID)",[str(infofile)])
 info={r[0]:r[1:] for r in db.sql('SELECT * FROM strict_glass_image_info').fetchall()}
 missing_orientation=[]
 kept=[]
 for r in candidates['glass']:
  original,thumb,landing,license,author,rot,nbytes=info[r['image_id']]
  if rot is None or not math.isfinite(rot) or int(rot) not in (0,90,180,270):
   missing_orientation.append(r['image_id']);continue
  r.update(original_url=original,thumbnail_url=thumb,original_landing_url=landing,license=license,author=author,rotation_ccw=int(rot),original_bytes=nbytes)
  kept.append(r)
 candidates['glass']=kept
 with gzip.open(ROOT/'image_candidates.json.gz','wt') as f:json.dump(candidates,f)
 assert len(candidates)==16 and len(set(r['image_id'] for rows in candidates.values() for r in rows))==sum(map(len,candidates.values()))
 plan={'status':'SOURCE_PLAN_READY','parent_manifest':str(PARENT/'PHOTO_SPLITS.json'),'parent_manifest_sha256':sha(PARENT/'PHOTO_SPLITS.json'),'parent_curriculum_sha256':sha(PARENT/'CURRICULUM.json'),'word_classes_and_comprehension_order_unchanged':True,'orientation_filter':False,'photos_per_class_unchanged':{'train':500,'valid':100,'test':100},'new_targets':{cl:800 if cl=='glass' else 100 for cl in candidates},'target_fresh_photos':2300,'glass_replaces_parent_photos':700,'glass_reserve_target':100,'reserve_target_each_other_replacement_class':100,'boxed_glass_source_profile':{'positive_label':'/m/09tvcd','positive_label_name':'Wine glass','positive_required':True,'same_label_instance_box_required':True,'box_min_area':0.04,'box_flags':{'IsGroupOf':0,'IsDepiction':0,'IsInside':0,'IsTruncated':0},'mirror_only_for_box_coordinate_consistency':True,'scope':'Drinking glass represented by wine-glass instances; narrower than generic Drinkware; excludes ceramic mugs. No requirement on physical object orientation.'},'candidate_counts':{cl:len(rows) for cl,rows in candidates.items()},'old_glass_parent_review_indices_inspected':[0,191],'old_glass_parent_action':'Supersede whole class source after systematic mismatches in sampled/reviewed source images; retain all original files and manifest','old_parent_photos_excluded_from_new_candidates':True,'human_positive_nonhuman_animal_images_excluded_from_new_candidates':True,'glass_unusable_orientation_metadata_exclusions':len(missing_orientation),'training_or_yin_requested':False}
 (ROOT/'SOURCE_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan),flush=True)
if __name__=='__main__':main()
