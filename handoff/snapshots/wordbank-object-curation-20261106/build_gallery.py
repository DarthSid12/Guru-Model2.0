"""Publish local review artifacts for the verified source bank; no model results used."""
import copy,hashlib,html,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'wordbank-inanimate-20261105'
OUT=ROOT.parents[1]/'output/wordbank-object-curation-20261106'
PREVIOUS=ROOT.parents[1]/'output/wordbank-inanimate-20261105'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 verified=json.loads((ROOT/'VERIFIED.json').read_text());assert verified['passed']
 rows=json.loads((ROOT/'PHOTO_SPLITS.json').read_text())['images'];byid={r['image_id']:r for r in rows}
 original=json.loads((PARENT/'CURRICULUM.json').read_text());current=copy.deepcopy(original)
 current.update(status='SOURCE_BANK_CURATED_AND_VERIFIED',parent_curriculum_sha256=sha(PARENT/'CURRICULUM.json'),source_curation_operation=str(ROOT),word_classes_and_comprehension_order_unchanged=True)
 for r in current['primary_original_1000_source_gate']['classes']:
  if r['class_name']=='glass':
   r['prior_source_mapping']={'id':r['open_images_id'],'name':r['open_images_name']};r['open_images_id']='/m/09tvcd';r['open_images_name']='Wine glass';r['human_positive_source_photos']=5600;r['source_scope']='Wine-glass instances of the word glass, human positive and matching non-depiction object box; narrower than generic Drinkware'
 (ROOT/'CURRICULUM.json').write_text(json.dumps(current,indent=2)+'\n')
 classes=current['primary_original_1000_source_gate']['classes'];prior={r['class_name']:r for r in json.loads((PREVIOUS/'IMAGE_PREVIEWS.json').read_text())['images']}
 assert [(r['word'],r['introduced_age_months'],r['proportion']) for r in classes]==[(r['word'],r['introduced_age_months'],r['proportion']) for r in original['primary_original_1000_source_gate']['classes']]
 font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',14);title=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',27)
 sheet=Image.new('RGB',(1440,80+8*190),'white');d=ImageDraw.Draw(sheet);d.text((10,8),'Object source bank · 64 classes',font=title,fill='#173d3a');d.text((10,45),'44,800 active photos · 1,600 reserves · 500 train / 100 validation / 100 test per class',font=font,fill='#3d615c')
 cards=[];previews=[];(OUT/'images').mkdir(parents=True,exist_ok=True)
 for i,c in enumerate(classes):
  cl=c['class_name'];preferred=prior.get(cl)
  if preferred and preferred['image_id'] in byid:r=byid[preferred['image_id']];source=Path(preferred['path'])
  else:
   candidates=[r for r in rows if r['class_name']==cl and r['split']=='valid']
   candidates.sort(key=lambda r:(not r.get('new_in_curation_pass',False),hashlib.sha256(('gallery-20261106|'+r['image_id']).encode()).digest()))
   r=candidates[0];source=Path(r['image_path'])
  body=source.read_bytes();assert hashlib.sha256(body).hexdigest()==r['png_sha256'];asset=OUT/'images'/f'{cl}.png';asset.write_bytes(body)
  with Image.open(source) as im:im.load();assert im.size==(224,224);tile=im.convert('RGB').resize((160,160))
  x=i%8*180;y=80+i//8*190;sheet.paste(tile,(x+10,y+4))
  label={'teddybear':'teddy bear (toy)','orange (food)':'orange','dress (object)':'dress','watch (object)':'watch'}.get(c['word'],c['word']);d.text((x+8,y+169),label,font=font,fill='#173d3a')
  cards.append(f'<article><h3>{html.escape(label)}</h3><img width="224" height="224" src="images/{cl}.png" alt="{html.escape(label)}"><p>500 train · 100 validation · 100 test</p><p>Example split: {r["split"]} · source ID {r["image_id"]}</p><p><a href="{html.escape(r["original_landing_url"])}">{html.escape(r["author"])}</a> · <a href="{html.escape(r["license"])}">license</a></p></article>')
  previews.append({'class_name':cl,'image_id':r['image_id'],'split':r['split'],'path':str(asset),'png_sha256':r['png_sha256']})
 sheet.save(OUT/'object-inventory.jpg',quality=94)
 (OUT/'GALLERY_VERIFIED.json').write_text(json.dumps({'classes':64,'all_preview_hashes_match_active_manifest':True,'examples_are_not_a_fixed_yin_pool':True,'images':previews},indent=2)+'\n')
 page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Curated object sources</title><style>body{font:16px/1.5 system-ui;margin:30px;background:#f4f6f3;color:#173d3a}main{max-width:1250px;margin:auto}.note{padding:22px;background:#dfeee7;border-radius:10px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}article{padding:18px;background:white;border:1px solid #d5dfd9;border-radius:8px}article p{font-size:12px}h3{margin:0 0 12px}a{color:#13675c}nav{margin:20px 0}</style><main><h1>64 object classes · updated photo sources</h1><div class="note"><strong>44,800 active photos + 1,600 reserves.</strong><br>4,045 fresh downloads; 2,445 parent photos superseded. Every class keeps 500 train / 100 validation / 100 test images. Classes and Wordbank comprehension order are unchanged.</div><nav><a href="README.md">Method and verification</a> · <a href="VERIFIED.json">Audit receipt</a> · <a href="object-inventory.jpg">Contact sheet</a></nav><p>New candidates were screened against 110 nonhuman-animal labels. Six flagged pictures were visually confirmed as stuffed toys and retained. Glass now uses boxed wine-glass instances instead of broad Drinkware. No orientation eligibility filter is applied.</p><p>These are source-photo examples. The 1,600 reserves are outside the active splits. No new model training or Yin run has started. Full image-by-image manual review was not performed.</p><section>'''+''.join(cards)+'</section></main></html>'
 (OUT/'index.html').write_text(page)
 for name in ['VERIFIED.json','PHYSICAL_TOY_OVERRIDES.json','SOURCE_PLAN.json','CURRICULUM.json']:(OUT/name).write_bytes((ROOT/name).read_bytes())
 readme='''# Object source continuation — 2026-11-06

The active bank has 64 object classes and 44,800 images, exactly 500 train + 100 validation + 100 test per class. This pass downloaded 4,045 new source photos, superseded 2,445 parent photos and added 1,600 unused reserves (100 for each of the 16 replacement classes). The other 42,355 parent photos retain their class and split assignments.

Class names, Wordbank comprehension numerators/denominators, introduction ages, stage counts and ordering are unchanged. Physical object orientation is not an eligibility criterion. Glass source metadata is corrected from generic Drinkware to human-positive Wine glass with a matching non-depiction object box. This represents glass with wine-glass instances, so it is narrower than the earlier broad vessel source.

Human-positive nonhuman-animal labels across 110 hierarchy entries were screened. This flagged 1,760 parent photos, including false matches to stuffed toys. Six unmistakable manufactured teddy-bear toys were visually confirmed and retained; the other flags were excluded from the updated bank. All 700 parent glass images were superseded because the broad source also contained mugs, scenes and an animal photo. There are zero unresolved animal-annotation flags in the active bank/reserves; this annotation-based screen does not certify every pixel is animal-free.

All 4,045 new PNG files and normalized pixel hashes were checked. Active and reserve source IDs, source-byte hashes and pixel hashes are disjoint. Human-positive source label membership, exact split counts and unchanged inherited assignments passed independent checks. Inherited photos reference the pinned, previously verified parent bank; their full bytes were not all rehashed during this pass.

Source photos can include people/background context. Visual checks cover parent-source review sheets, corrected-glass examples, six toy adjudications and the 64 gallery examples; there was no exhaustive manual review of all 44,800 images. No new training, calibration, fixed Yin stimulus selection or Yin run was performed. Existing model runs and historical results are unchanged.

Files: [gallery](index.html), [contact sheet](object-inventory.jpg), [verification receipt](VERIFIED.json), [source plan](SOURCE_PLAN.json), [toy decisions](PHYSICAL_TOY_OVERRIDES.json).

Authoritative data manifests and code are in ../../operations/wordbank-object-curation-20261106/. PHOTO_SPLITS.json is the active bank; RESERVE_PHOTOS.json stays outside training/evaluation; SUPERSEDED_PARENT_PHOTOS.json preserves the audit trail. raw/objects is the active split tree; raw/objects-unassigned is a download/audit cache and must not be packed directly.
'''
 (OUT/'README.md').write_text(readme);(ROOT/'README.md').write_text(readme)
 status={'status':'SOURCE_BANK_UPDATED_VERIFIED','classes':64,'active_images':44800,'new_photos_downloaded':4045,'parent_photos_superseded':2445,'reserve_photos':1600,'counts_per_class':{'train':500,'valid':100,'test':100},'unresolved_animal_annotation_flags':0,'physical_toy_overrides':6,'orientation_filter':False,'exhaustive_manual_review':False,'training_or_yin_run':False,'gallery':str(OUT/'index.html'),'verification':str(ROOT/'VERIFIED.json')}
 (ROOT/'LATEST_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
 pointer={'manifest':str(ROOT/'PHOTO_SPLITS.json'),'manifest_sha256':sha(ROOT/'PHOTO_SPLITS.json'),'curriculum':str(ROOT/'CURRICULUM.json'),'curriculum_sha256':sha(ROOT/'CURRICULUM.json'),'reserves':str(ROOT/'RESERVE_PHOTOS.json'),'verification':str(ROOT/'VERIFIED.json'),'gallery':str(OUT/'index.html'),'does_not_modify_existing_experiment_registrations':True}
 (ROOT.parent/'wordbank-objects-current.json').write_text(json.dumps(pointer,indent=2)+'\n')
 print(json.dumps(status),flush=True)
if __name__=='__main__':main()
