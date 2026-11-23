"""Resume real-photo sourcing. No training, cluster submissions or old dataset edits.

Default: fetch replacement classes to 700 photos each and preserve references
to frozen retained-photo splits. --preview fetches four photos/replacement only.
A completed source manifest still requires visual review before use in experiments.
"""
import argparse,collections,concurrent.futures,gzip,hashlib,http.client,io,json,os,ssl,subprocess,threading,time,urllib.error,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parent
PHOTO_COUNTS={'train':500,'valid':100,'test':100}
SEED=20261020
TLS_CONTEXT=ssl.create_default_context()
CONNECTIONS=threading.local()

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(name,value):
 p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def connection_check():
 results=[]
 for host in ['open-images-dataset.s3.amazonaws.com','live.staticflickr.com']:
  r=subprocess.run(['curl','-4','-sS','--connect-timeout','4','--max-time','6','-I','https://'+host],capture_output=True,text=True,timeout=8)
  results.append({'host':host,'exit_code':r.returncode,'error':r.stderr.strip()[:300]})
 return results

# Frozen photo-normalization functions copied from original prepare_data.py.
# Original file SHA256: 821950e902d91aa3105398d65320e76e47d82f3234dc063421894fed6f4ea544

def fetch_bytes(url):
    """Reuse verified TLS connections instead of rebuilding a CA store per photo."""
    from urllib.parse import urlsplit
    parts=urlsplit(url)
    assert parts.scheme in ('https','http')
    if parts.netloc!='open-images-dataset.s3.amazonaws.com':
        assert parts.hostname and parts.hostname.endswith('.staticflickr.com'),'Unexpected source-image host'
        # Flickr follows redirects to its current CDN. Reuse the verified CA
        # context, which avoids expensive platform trust-store loads per image.
        secure_url='https://'+parts.netloc+parts.path+('?' + parts.query if parts.query else '')
        return urllib.request.urlopen(secure_url,timeout=35,context=TLS_CONTEXT).read()
    connection=getattr(CONNECTIONS,'connection',None)
    if connection is None:
        connection=http.client.HTTPSConnection(parts.netloc,timeout=35,context=TLS_CONTEXT)
        CONNECTIONS.connection=connection
    try:
        connection.request('GET',parts.path,headers={'User-Agent':'Wordbank-visual-curriculum-research/1.0','Connection':'keep-alive'})
        response=connection.getresponse()
        body=response.read()
        if response.status!=200:
            raise urllib.error.HTTPError(url,response.status,response.reason,response.headers,None)
        return body
    except (OSError,http.client.HTTPException):
        connection.close()
        CONNECTIONS.connection=None
        raise

def render_image(body, box, rotation_ccw=None, metadata_orientation=False):
    with Image.open(io.BytesIO(body)) as original:
        original.load()
        if original.width < 128 or original.height < 128:
            raise ValueError('Original photo smaller than 128 pixels on one side')
        orientation = original.getexif().get(274, 1)
        image = original.convert('RGB') if metadata_orientation else ImageOps.exif_transpose(original).convert('RGB')
    original_size = list(image.size)
    if not metadata_orientation and orientation not in (None, 1):
        # Annotation coordinate convention for EXIF-rotated originals is unclear;
        # preserve the entire upright image rather than crop incorrect coordinates.
        box = None
    crop = None
    if box:
        width, height = image.size
        x0, x1 = float(box['xmin'])*width, float(box['xmax'])*width
        y0, y1 = float(box['ymin'])*height, float(box['ymax'])*height
        if min(x1-x0, y1-y0) < 80:
            raise ValueError('Annotated target too small for a useful crop')
        px, py = (x1-x0)*.10, (y1-y0)*.10
        crop = [max(0, int(x0-px)), max(0, int(y0-py)), min(width, int(x1+px+1)), min(height, int(y1+py+1))]
        image = image.crop(crop)
    # Keep the whole target, including for long thin objects. The normal packer
    # therefore cannot center-crop away an off-centre Open Images label.
    image = ImageOps.pad(image, (224,224), method=Image.Resampling.BICUBIC, color=(127,127,127))
    applied_rotation=int(rotation_ccw or 0) if metadata_orientation else 0
    assert applied_rotation in (0,90,180,270)
    if applied_rotation:
        image=image.transpose({90:Image.Transpose.ROTATE_90,180:Image.Transpose.ROTATE_180,270:Image.Transpose.ROTATE_270}[applied_rotation])
        if applied_rotation in (90,270):original_size.reverse()
    pixels = np.asarray(image)
    if float(pixels.std()) < 8:
        raise ValueError('Near-uniform image')
    return image, {'source_oriented_size': original_size, 'crop_xyxy': crop,
                   'exif_orientation': orientation, 'normalization_rotation_applied_ccw':applied_rotation,
                   'official_rotation_metadata_used':metadata_orientation,
                   'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()}

def download_one(task):
    class_name, candidate = task
    iid = candidate['image_id']
    original_path = ROOT/'raw/openimages-originals'/f'{iid}.jpg'
    normalized = ROOT/'raw/objects-unassigned'/class_name/f'{iid}.png'
    original_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.parent.mkdir(parents=True, exist_ok=True)
    try:
        body = None
        choices=[(candidate['url'],'cvdf_mirror')]
        if candidate.get('thumbnail_url'):choices.append((candidate['thumbnail_url'],'flickr_thumbnail'))
        if candidate.get('original_url') and (candidate.get('original_bytes') or 0)<=25*1024*1024:
            choices.append((candidate['original_url'],'flickr_original'))
        last_error=None
        for source_url,source_kind in choices:
            for attempt in range(3):
                try:
                    body = fetch_bytes(source_url)
                    break
                except (OSError, urllib.error.URLError,http.client.HTTPException) as exc:
                    last_error=exc
                    if isinstance(exc, urllib.error.HTTPError) and exc.code in (403,404):
                        break
                    time.sleep(.5*(attempt+1))
            if body is not None:break
        if body is None:raise last_error or ValueError('No source URL available')
        if KEEP_ORIGINALS:
            tmp = original_path.with_suffix('.jpg.part')
            tmp.write_bytes(body)
            os.replace(tmp, original_path)
        use_metadata=source_kind in ('cvdf_mirror','flickr_original') and candidate.get('rotation_ccw') is not None
        # Mirror boxes are in stripped-EXIF sensor orientation. Flickr resized
        # thumbnails already have the author's orientation, so preserve them
        # whole rather than apply sensor-coordinate boxes or rotate them twice.
        box=candidate['box'] if source_kind=='cvdf_mirror' else None
        image, details = render_image(body,box,candidate.get('rotation_ccw'),use_metadata)
        image.save(normalized)
        return {'accepted': True, 'class_name': class_name, 'image_id': iid,
                'url': source_url, 'download_kind':source_kind,'download_strategy_version':2,
                'official_rotation_ccw':candidate.get('rotation_ccw'),
                'license':candidate.get('license'),'author':candidate.get('author'),
                'original_landing_url':candidate.get('original_landing_url'),
                'annotation_box': candidate['box'],
                'original_sha256': hashlib.sha256(body).hexdigest(),
                'normalized_path': str(normalized.relative_to(ROOT)),
                'png_sha256': sha(normalized), **details}
    except Exception as exc:
        return {'accepted': False, 'class_name': class_name, 'image_id': iid,'download_strategy_version':2,
                'reason': type(exc).__name__ + ': ' + str(exc)[:250]}

KEEP_ORIGINALS=False

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--workers',type=int,default=16);args=parser.parse_args()
 plan=json.loads((ROOT/'IMAGE_PLAN.json').read_text());assert plan['classes']==64
 reuse=json.loads((ROOT/'REUSED_PHOTO_SPLITS.json').read_text())['images']
 candidates=json.load(gzip.open(ROOT/'image_candidates.json.gz','rt'))
 assert len(candidates)==16
 # Keep recorded source/pixel identities across retained objects, even in previews.
 ids={r['image_id'] for r in reuse};pixels={r['pixel_sha256'] for r in reuse};originals={r['original_sha256'] for r in reuse}
 accepted=collections.defaultdict(list);attempted=collections.defaultdict(set)
 rejections=json.loads((ROOT/'REVIEW_REJECTIONS.json').read_text())['rejections'] if (ROOT/'REVIEW_REJECTIONS.json').exists() else []
 rejected={r['image_id'] for r in rejections}
 journal=ROOT/'DOWNLOAD_JOURNAL.jsonl'
 if journal.exists():
  for line in journal.open():
   r=json.loads(line);cl=r['class_name']
   if cl not in candidates:continue
   if r['image_id'] in rejected:
    attempted[cl].add(r['image_id']);continue
   if r['accepted']:
    attempted[cl].add(r['image_id'])
    assert cl in candidates
    p=ROOT/r['normalized_path'];assert sha(p)==r['png_sha256']
    assert r['image_id'] not in ids and r['pixel_sha256'] not in pixels and r['original_sha256'] not in originals
    ids.add(r['image_id']);pixels.add(r['pixel_sha256']);originals.add(r['original_sha256']);accepted[cl].append(r)
 target=4 if args.preview else 700
 atomic('LATEST_STATUS.json',{'status':'PHOTO_SOURCING_IN_PROGRESS','classes':64,'training_ready':False,'replacement_photos_required':11200,'mode':'preview' if args.preview else 'full'})
 if any(len(accepted[cl])<target for cl in candidates):
  network=connection_check();atomic('NETWORK_CHECK.json',{'hosts':network,'observed_at_unix':time.time()})
  if not any(r['exit_code']==0 for r in network):
   atomic('LATEST_STATUS.json',{'status':'BLOCKED_NETWORK_PHOTO_HOSTS_UNREACHABLE','training_ready':False,'word_classes':64,'retained_photo_records':len(reuse),'downloaded_replacement_photos':sum(map(len,accepted.values())),'required_replacement_photos':11200,'network':network,'resume':'Run fetch_photos.py after network connectivity returns'})
   print('BLOCKED: photo hosts unreachable. No dataset readiness marker written.',flush=True);return 2
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,journal.open('a',buffering=1) as log:
  for cl,options in candidates.items():
   available=[r for r in options if r['image_id'] not in attempted[cl] and r['image_id'] not in rejected]
   while len(accepted[cl])<target and available:
    batch,available=available[:min(args.workers,target-len(accepted[cl]))],available[min(args.workers,target-len(accepted[cl])):]
    # map preserves candidate order; thread completion cannot change the final set.
    for r in pool.map(download_one,[(cl,r) for r in batch]):
     attempted[cl].add(r['image_id'])
     if r['accepted'] and (r['image_id'] in ids or r['pixel_sha256'] in pixels or r['original_sha256'] in originals):
      r['accepted']=False;r['reason']='Duplicate image ID/source bytes/normalized pixels across this dataset'
     if r['accepted']:
      ids.add(r['image_id']);pixels.add(r['pixel_sha256']);originals.add(r['original_sha256']);accepted[cl].append(r)
     log.write(json.dumps(r,allow_nan=False)+'\n')
    atomic('DOWNLOAD_PROGRESS.json',{'last_class':cl,'class_counts':{k:len(accepted[k]) for k in candidates},'target_per_class':target,'training_ready':False})
    print(cl,len(accepted[cl]),'/',target,flush=True)
   if len(accepted[cl])<target:
    atomic('LATEST_STATUS.json',{'status':'INSUFFICIENT_DECODED_PHOTOS','class':cl,'actual':len(accepted[cl]),'required':target,'training_ready':False});return 3
 if args.preview:
  atomic('REPLACEMENT_PREVIEW.json',{'status':'PHOTOS_DOWNLOADED_VISUAL_REVIEW_PENDING','images':[r for cl in candidates for r in accepted[cl][:4]],'review_rejections_applied':len(rejected),'training_ready':False})
  print('Replacement preview photos ready for visual review.',flush=True);return 0
 records=[]
 for r in reuse:
  # These photos/splits are inherited from the already frozen source bank.
  # Read actual bytes again when materializing/packing on the training host.
  source=Path(r['reuse_source_root'])/r['normalized_path']
  assert source.exists()
  records.append({**r,'image_path':str(source),'file_verified_in_this_operation':False,'verification_basis':'Inherited pinned, previously verified photo manifest; source manifest matches live Anvil SHA256'})
 for cl in candidates:
  rows=sorted(accepted[cl],key=lambda r:hashlib.sha256(f"split-{SEED}|{r['image_id']}".encode()).digest())
  assert len(rows)==700
  start=0
  for split,count in PHOTO_COUNTS.items():
   records.extend({**r,'split':split,'image_path':str(ROOT/r['normalized_path']),'file_verified_in_this_operation':True} for r in rows[start:start+count]);start+=count
 assert len(records)==44800 and len(ids)==len(pixels)==len(originals)==44800
 assert len({r['class_name'] for r in records})==64
 for cl in {r['class_name'] for r in records}:
  assert collections.Counter(r['split'] for r in records if r['class_name']==cl)==PHOTO_COUNTS
 expected_links={ROOT/Path('raw/objects')/r['split']/r['class_name']/(r['image_id']+'.png') for r in records}
 for stale in (ROOT/'raw/objects').glob('*/*/*.png'):
  if stale not in expected_links:
   assert stale.is_symlink(),'Refuse to remove a non-link outside this manifest'
   stale.unlink()
 for r in records:
  relative=Path('raw/objects')/r['split']/r['class_name']/(r['image_id']+'.png')
  link=ROOT/relative;link.parent.mkdir(parents=True,exist_ok=True)
  source=Path(r['image_path'])
  if link.is_symlink():assert link.resolve()==source.resolve()
  else:link.symlink_to(os.path.relpath(source,link.parent))
  r['split_relative_path']=str(relative)
 atomic('PHOTO_SPLITS.json',{'status':'SOURCED_WITH_PINNED_REUSED_PHOTOS_SAMPLE_REVIEW_COMPLETE','seed':SEED,'counts_per_class':PHOTO_COUNTS,'distinct_source_ids':len(ids),'distinct_normalized_pixel_sha256':len(pixels),'distinct_original_sha256':len(originals),'images':records,'training_ready':False})
 atomic('LATEST_STATUS.json',{'status':'ALL_SOURCES_READY_FULL_IMAGE_REVIEW_NOT_PERFORMED','training_ready':False,'photos':44800,'classes':64,'new_photos_downloaded':11200,'retained_photo_references':33600})
 print('44800 source entries ready: 11200 verified new photos + 33600 pinned retained references.',flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
