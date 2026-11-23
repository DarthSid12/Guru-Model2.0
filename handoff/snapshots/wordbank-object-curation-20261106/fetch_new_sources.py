"""Fetch independent reserve photos and high-precision boxed drinking glasses."""
import collections,concurrent.futures,gzip,hashlib,importlib.util,json,os,time,urllib.error,threading,email.utils
from urllib.parse import urlsplit
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'wordbank-inanimate-20261105'
spec=importlib.util.spec_from_file_location('frozen_photo_worker',PARENT/'fetch_photos.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker);worker.ROOT=ROOT

ORIGINAL_FETCH=worker.fetch_bytes
HOST_LOCK=threading.Lock()
NEXT_REQUEST={}
BLOCKED_UNTIL={}
def paced_fetch(url):
 host=urlsplit(url).netloc
 if host.endswith('.staticflickr.com'):host='staticflickr.com'
 with HOST_LOCK:
  scheduled=max(time.monotonic(),NEXT_REQUEST.get(host,0))
  NEXT_REQUEST[host]=scheduled+(.15 if host=='open-images-dataset.s3.amazonaws.com' else .35)
 delay=scheduled-time.monotonic()
 if delay>0:time.sleep(delay)
 while True:
  with HOST_LOCK:cooldown=BLOCKED_UNTIL.get(host,0)-time.monotonic()
  if cooldown<=0:break
  time.sleep(cooldown)
 try:return ORIGINAL_FETCH(url)
 except urllib.error.HTTPError as exc:
  if exc.code==429:
   raw=exc.headers.get('Retry-After','30') if exc.headers else '30'
   try:wait=max(30,float(raw))
   except ValueError:
    try:wait=max(30,email.utils.parsedate_to_datetime(raw).timestamp()-time.time())
    except Exception:wait=60
   with HOST_LOCK:
    BLOCKED_UNTIL[host]=max(BLOCKED_UNTIL.get(host,0),time.monotonic()+wait)
    NEXT_REQUEST[host]=max(NEXT_REQUEST.get(host,0),time.monotonic()+wait)
   print('Rate limit: delaying host',host,'for at least',round(wait),'seconds',flush=True)
  raise
worker.fetch_bytes=paced_fetch

def atomic(name,obj):
 p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def boxed_glass(candidate):
 iid=candidate['image_id'];path=ROOT/'raw/objects-unassigned/glass'/f'{iid}.png';path.parent.mkdir(parents=True,exist_ok=True)
 try:
  body=worker.fetch_bytes(candidate['url'])
  image,details=worker.render_image(body,candidate['box'],candidate['rotation_ccw'],True);image.save(path)
  return {'accepted':True,'class_name':'glass','image_id':iid,'url':candidate['url'],'download_kind':'cvdf_mirror','source_profile':'boxed_wine_glass_v1','verified_positive_label':'/m/09tvcd','annotation_box':candidate['box'],'official_rotation_ccw':candidate['rotation_ccw'],'license':candidate['license'],'author':candidate['author'],'original_landing_url':candidate['original_landing_url'],'original_sha256':hashlib.sha256(body).hexdigest(),'png_sha256':worker.sha(path),'normalized_path':str(path.relative_to(ROOT)),**details}
 except Exception as exc:return {'accepted':False,'class_name':'glass','image_id':iid,'reason':str(exc)[:350]}
def run_task(task):
 cl,candidate=task
 if cl=='glass':return boxed_glass(candidate)
 mirror={**candidate,'thumbnail_url':None,'original_url':None}
 return worker.download_one((cl,mirror))
def main():
 plan=json.loads((ROOT/'SOURCE_PLAN.json').read_text());parent_path=Path(plan['parent_manifest']);assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==plan['parent_manifest_sha256']
 old=ROOT.parent/'anvil-wordbank-lp-20261020/PHOTO_SPLITS.json'
 base=json.loads(parent_path.read_text())['images']+json.loads(old.read_text())['images']
 ids={r['image_id'] for r in base};pixels={r['pixel_sha256'] for r in base};sources={r['original_sha256'] for r in base}
 candidates=json.load(gzip.open(ROOT/'image_candidates.json.gz','rt'));accepted=collections.defaultdict(list);attempted=collections.defaultdict(set)
 rejected={r['image_id'] for r in json.loads((ROOT/'REVIEW_REJECTIONS.json').read_text()).get('rejections',[])} if (ROOT/'REVIEW_REJECTIONS.json').exists() else set()
 journal=ROOT/'DOWNLOAD_JOURNAL.jsonl'
 if journal.exists():
  for line in journal.open():
   r=json.loads(line);cl=r['class_name']
   if r['accepted'] or not any(v in r.get('reason','').lower() for v in ['429','timed out','timeout','connection','temporary','network','nodename']):attempted[cl].add(r['image_id'])
   if r['accepted'] and r['image_id'] not in rejected:
    assert worker.sha(ROOT/r['normalized_path'])==r['png_sha256']
    assert r['image_id'] not in ids and r['pixel_sha256'] not in pixels and r['original_sha256'] not in sources
    ids.add(r['image_id']);pixels.add(r['pixel_sha256']);sources.add(r['original_sha256']);accepted[cl].append(r)
 network=worker.connection_check();atomic('NETWORK_CHECK.json',{'hosts':network})
 if not any(v['host']=='open-images-dataset.s3.amazonaws.com' and v['exit_code']==0 for v in network):
  atomic('LATEST_STATUS.json',{'status':'BLOCKED_NETWORK','network':network});return 2
 order=sorted(plan['new_targets'],key=lambda cl:(cl!='glass',list(plan['new_targets']).index(cl)))
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool,journal.open('a',buffering=1) as log:
  for cl in order:
   target=plan['new_targets'][cl];available=[r for r in candidates[cl] if r['image_id'] not in attempted[cl] and r['image_id'] not in rejected]
   while len(accepted[cl])<target and available:
    n=min(8,target-len(accepted[cl]));batch,available=available[:n],available[n:]
    for r in pool.map(run_task,[(cl,r) for r in batch]):
     if r['accepted'] and (r['image_id'] in ids or r['pixel_sha256'] in pixels or r['original_sha256'] in sources):r['accepted']=False;r['reason']='Exact duplicate against parent/original source banks or new sourcing pass'
     if r['accepted']:
      ids.add(r['image_id']);pixels.add(r['pixel_sha256']);sources.add(r['original_sha256']);accepted[cl].append(r)
     log.write(json.dumps(r,allow_nan=False)+'\n')
    atomic('DOWNLOAD_PROGRESS.json',{'target_counts':plan['new_targets'],'accepted_counts':{k:len(accepted[k]) for k in order},'total':sum(map(len,accepted.values())),'training_started':False})
    print(cl,len(accepted[cl]),'/',target,flush=True)
   if len(accepted[cl])<target:
    atomic('LATEST_STATUS.json',{'status':'SOURCE_SHORTFALL','class':cl,'actual':len(accepted[cl]),'target':target});return 3
 atomic('FRESH_SOURCES.json',{'source_plan_sha256':worker.sha(ROOT/'SOURCE_PLAN.json'),'images':[r for cl in order for r in accepted[cl]],'images_are_not_yet_assigned_to_training_or_reserve':True})
 atomic('LATEST_STATUS.json',{'status':'NEW_PHOTOS_DOWNLOADED_REVIEW_PENDING','new_photos':sum(map(len,accepted.values())),'training_started':False});print('Fresh sourcing pass complete',sum(map(len,accepted.values())),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
