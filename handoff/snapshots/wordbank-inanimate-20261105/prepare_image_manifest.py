"""Freeze non-animal image sources; retain old object splits and ban old animal photos."""
import collections,csv,gzip,hashlib,json,math
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'anvil-wordbank-lp-20261020'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2)+'\n')
def main():
 c=json.loads((ROOT/'CURRICULUM.json').read_text());classes=c['primary_original_1000_source_gate']['classes']
 oldpath=BASE/'PHOTO_SPLITS.json';old=json.loads(oldpath.read_text());words={r['class_name'] for r in classes}
 retained=[{**r,'reuse_source_root':str(BASE),'reuse_original_split':True,'file_verified_in_this_operation':False} for r in old['images'] if r['class_name'] in words]
 retained_classes={r['class_name'] for r in retained}
 oldclasses=json.loads((BASE/'wordbank/CURRICULUM.json').read_text())['classes']
 removed={r['class_name'] for r in oldclasses if r['category']=='animals' and r['word']!='teddybear'}
 assert len(retained)==48*700 and len(retained_classes)==48
 assert not retained_classes&removed
 assert len({r['image_id'] for r in old['images']})==44800
 # Do not relabel any of the removed animal photos as objects.
 excluded_ids={r['image_id'] for r in old['images']}
 write('REUSED_PHOTO_SPLITS.json',{'source_manifest':str(oldpath),'source_manifest_sha256':sha(oldpath),'image_count':len(retained),'classes':len(retained_classes),'images':retained})
 additions=[r for r in classes if r['class_name'] not in retained_classes]
 db=duckdb.connect(str(ROOT/'source-index.duckdb'))
 db.execute("ATTACH '"+str(BASE/'wordbank/source-index.duckdb').replace("'","''")+"' AS old (READ_ONLY)")
 db.execute('CREATE OR REPLACE TEMP VIEW all_positives AS SELECT * FROM old.positives UNION SELECT * FROM extra_positives')
 db.execute('CREATE OR REPLACE TEMP VIEW all_boxes AS SELECT * FROM old.boxes UNION SELECT * FROM extra_boxes')
 counts=dict(db.sql('SELECT LabelName,count(*) FROM all_positives GROUP BY 1').fetchall())
 labels={r['open_images_id'] for r in additions}
 literal=','.join("'"+s+"'" for s in sorted(labels))
 rows=db.execute(f'SELECT ImageID,LabelName FROM all_positives WHERE LabelName IN ({literal})').fetchall()
 owner={}
 for iid,label in rows:
  if iid in excluded_ids:continue
  key=(counts[label],hashlib.sha256(('owner-v1|'+iid+'|'+label).encode()).digest())
  if iid not in owner or key<owner[iid][0]:owner[iid]=(key,label)
 sources={};coverage=[]
 for cl in additions:
  label=cl['open_images_id']
  entries=db.execute('SELECT p.ImageID,b.box FROM all_positives p LEFT JOIN all_boxes b USING(ImageID,LabelName) WHERE p.LabelName=?',[label]).fetchall()
  entries=[(iid,box) for iid,box in entries if iid in owner and owner[iid][1]==label]
  entries.sort(key=lambda v:(v[1] is None,hashlib.sha256(('source-order-20261020|'+v[0]).encode()).digest()))
  candidates=[{'image_id':iid,'box':box,'url':'https://open-images-dataset.s3.amazonaws.com/train/'+iid+'.jpg'} for iid,box in entries[:4000]]
  sources[cl['class_name']]=candidates
  coverage.append({'word':cl['word'],'class_name':cl['class_name'],'human_positive_source_photos':counts[label],'owned_after_excluding_all_old_photo_ids':len(entries),'candidates':len(candidates),'candidates_with_box':sum(x['box'] is not None for x in candidates),'minimum_800_owned_source_reserve_met':len(entries)>=800})
  print(coverage[-1],flush=True)
 wanted=sorted({r['image_id'] for rows in sources.values() for r in rows})
 db.execute('CREATE OR REPLACE TEMP TABLE wanted(ImageID VARCHAR PRIMARY KEY)')
 db.executemany('INSERT INTO wanted VALUES (?)',[(v,) for v in wanted])
 infofile=ROOT.parents[1]/'tmp/oidv6-train-images-with-labels-with-rotation.csv'
 db.execute("""CREATE OR REPLACE TABLE replacement_image_info AS SELECT i.ImageID,i.OriginalURL,i.Thumbnail300KURL,i.OriginalLandingURL,i.License,i.Author,try_cast(i.Rotation AS DOUBLE) AS Rotation,try_cast(i.OriginalSize AS BIGINT) AS OriginalSize FROM read_csv_auto(?,header=true,all_varchar=true) i INNER JOIN wanted w USING(ImageID)""",[str(infofile)])
 info={r[0]:r[1:] for r in db.sql('SELECT * FROM replacement_image_info').fetchall()}
 for rows in sources.values():
  for r in rows:
   v=info.get(r['image_id'])
   if v:
    original,thumb,landing,license,author,rotation,nbytes=v
    r.update(original_url=original,thumbnail_url=thumb,original_landing_url=landing,license=license,author=author,rotation_ccw=int(rotation) if rotation is not None and math.isfinite(rotation) else None,original_bytes=nbytes)
 with gzip.open(ROOT/'image_candidates.json.gz','wt') as f:json.dump(sources,f)
 assert len(sources)==16
 assert len(wanted)==sum(map(len,sources.values()))
 assert not set(wanted)&excluded_ids
 assert len(info)==len(wanted)
 plan={'status':'MANIFEST_READY_REPLACEMENT_DOWNLOADS_PENDING','primary_curriculum_sha256':sha(ROOT/'CURRICULUM.json'),'classes':64,'old_animal_classes_removed':16,'retained_object_classes':48,'replacement_object_classes':16,'reused_original_split_records':len(retained),'replacement_photos_required':16*700,'total_photos_target':64*700,'photo_budget_per_class':{'train':500,'valid':100,'test':100},'source_ownership':'Assign new photos to scarcer selected replacement labels; fixed owner-v1 hash tie-break; exclude all 44800 prior photo IDs','candidate_order':'Bounding boxes first, then original source-order-20261020 SHA256 image-ID ordering','source_annotations':json.loads((BASE/'wordbank/CURRICULUM.json').read_text())['annotations'],'coverage':coverage,'all_replacements_have_800_owned_candidates':all(r['minimum_800_owned_source_reserve_met'] for r in coverage),'replacement_candidates':len(wanted),'official_metadata_records':len(info),'reuse_current_files_verified':False,'replacement_decoded_images':0,'training_ready':False}
 write('IMAGE_PLAN.json',plan)
 print('PLAN',json.dumps({k:v for k,v in plan.items() if k not in ['coverage','source_annotations']}),flush=True)
if __name__=='__main__':main()
