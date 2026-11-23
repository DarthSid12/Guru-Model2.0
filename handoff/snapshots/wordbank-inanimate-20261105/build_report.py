"""Build a transparent word/source review artifact; pending photos stay explicit."""
import collections,csv,hashlib,html,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'anvil-wordbank-lp-20261020'
OUT=ROOT.parents[1]/'output/wordbank-inanimate-20261105'
OUT.mkdir(parents=True,exist_ok=True)

def main():
 c=json.loads((ROOT/'CURRICULUM.json').read_text());p=json.loads((ROOT/'IMAGE_PLAN.json').read_text())
 rows=c['primary_original_1000_source_gate']['classes'];old=set(c['retained_old_words']);new=set(c['new_words'])
 stats={(r['word'],r['age']):r for r in json.loads((ROOT/'word_proportions.json').read_text())}
 source_reuse=json.loads((ROOT/'REUSED_PHOTO_SPLITS.json').read_text())['images']
 preview=json.loads((ROOT/'REPLACEMENT_PREVIEW.json').read_text())['images'] if (ROOT/'REPLACEMENT_PREVIEW.json').exists() else []
 try:prior_yin=json.loads((ROOT.parent/'sdsc-wordbank-objects-20261026/SELECTION.json').read_text())['images']
 except FileNotFoundError:prior_yin=[]
 chosen={r['class_name']:r for r in prior_yin if r['class_name'] in {r['class_name'] for r in rows}}
 for r in source_reuse:
  if r['class_name'] not in chosen and r['split']=='valid':chosen[r['class_name']]=r
 for r in preview:chosen.setdefault(r['class_name'],r)
 images=[];cards=[]
 sourced=(ROOT/'PHOTO_MANIFEST_VERIFIED.json').exists()
 font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',14)
 sheet=Image.new('RGB',(8*180,80+8*190),'white');draw=ImageDraw.Draw(sheet)
 titlefont=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',29)
 draw.text((10,9),'64 Wordbank object classes',font=titlefont,fill='#142b31')
 draw.text((10,47),'16 replacements outlined in teal • one source-photo example per class',font=font,fill='#42635f')
 for i,r in enumerate(rows):
  cl=r['class_name'];word=r['word'];asset=None;record=chosen.get(cl)
  if record:
   is_new=word in new
   paths=[ROOT/record['normalized_path']] if is_new else [ROOT.parents[1]/'output/objects-curriculum-20261030/assets'/record['normalized_path'],BASE/record['normalized_path']]
   source=next((v for v in paths if v.exists()),None)
   if source:
    body=source.read_bytes();assert hashlib.sha256(body).hexdigest()==record['png_sha256']
    with Image.open(source) as im:
     im.load();assert im.size==(224,224);im=im.convert('RGB')
     assert hashlib.sha256(im.tobytes()).hexdigest()==record['pixel_sha256']
     asset=OUT/'images'/f'{cl}.png';asset.parent.mkdir(exist_ok=True);asset.write_bytes(body)
     tile=im.resize((160,160));sheet.paste(tile,((i%8)*180+10,80+(i//8)*190+6))
    images.append({'word':word,'class_name':cl,'image_id':record['image_id'],'path':str(asset),'png_sha256':record['png_sha256'],'original_landing_url':record.get('original_landing_url'),'author':record.get('author'),'license':record.get('license'),'kind':'replacement_preview' if is_new else 'retained_object_validation_example','new_yin_pool_selection':False})
  x=(i%8)*180;y=80+(i//8)*190
  display={'teddybear':'teddy bear (toy)','orange (food)':'orange','watch (object)':'watch','dress (object)':'dress'}.get(word,word)
  draw.text((x+8,y+170),display+(' [new]' if word in new else ''),font=font,fill='#136b65' if word in new else '#142b31')
  if word in new:draw.rectangle((x+8,y+4,x+172,y+168),outline='#168078',width=3)
  status='replacement photo pending' if asset is None else ('replacement object example' if word in new else 'retained object example')
  # Source status is kept in the gallery rather than repeated on the contact sheet.
  if asset is None:
   draw.rectangle((x+10,y+6,x+170,y+166),fill='#f6e9cf');draw.text((x+24,y+70),'Download pending',font=font,fill='#965000')
  photo=f'<img src="images/{cl}.png" width="160" height="160" alt="{html.escape(word)}">' if asset else '<div class="pending">Replacement photo download pending</div>'
  credits=''
  if asset:
   credits=f'<p><a href="{html.escape(record.get("original_landing_url") or record.get("url") or "")}">Source photo</a> · <a href="{html.escape(record.get("license") or "")}">license</a> · {html.escape(record.get("author") or "unknown author")}</p>'
  cards.append(f'<article><h3>{i+1}. {html.escape(word)}</h3>{photo}<p>{status}</p><p>Introduced: {r["introduced_age_months"]} months · comprehension {100*r["proportion"]:.1f}% ({r["n_understands"]}/{r["n_responses"]})</p><p>{r["human_positive_source_photos"]:,} human-positive source photos · {html.escape(r["open_images_name"])}</p>{credits}</article>')
 sheet.save(OUT/'object-inventory.jpg',quality=93)
 expected={r['class_name'] for r in rows}
 for stale in (OUT/'images').glob('*.png'):
  if stale.stem not in expected:stale.unlink()
 (OUT/'IMAGE_PREVIEWS.json').write_text(json.dumps({'verified_preview_images':len(images),'replacement_photos_pending':16-len({r['class_name'] for r in images if r['kind']=='replacement_preview'}),'images':images,'source_manifest_complete':sourced,'new_yin_pool_selected':False},indent=2)+'\n')
 for name in ['OBJECT_WORDS.csv','CURRICULUM.json','ELIGIBILITY_AUDIT.json','IMAGE_PLAN.json','EXTRACTION_VERIFIED.json']:(OUT/name).write_bytes((ROOT/name).read_bytes())
 with (OUT/'WORD_COMPREHENSION.csv').open('w') as f:
  columns=['word','class_name','introduced_age_months','new_replacement']
  for age in [8,10,12,15,18]:columns.extend([f'age{age}_n_understands',f'age{age}_n_responses',f'age{age}_proportion'])
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
  for r in rows:
   row={k:r[k] for k in ['word','class_name','introduced_age_months']};row['new_replacement']=r['word'] in new
   for age in [8,10,12,15,18]:
    s=stats[(r['word'],age)]
    for key in ['n_understands','n_responses','proportion']:row[f'age{age}_{key}']=s[key]
   w.writerow(row)
 lines=['# Non-animal Wordbank object inventory','',f'64 object classes: 48 retained, 16 replacements. The word extraction and source-image manifest are complete. Downloaded photographs shown here are previews; full sourcing and visual review remain pending. This dataset is not ready for training or Yin evaluation.','', '## Selection','', 'American English Words & Gestures (WG), parent-reported comprehension (understands). Same pinned public Wordbank source, same 8/10/12/15/18 month checkpoints, same cumulative 4/8/16/32/64 class targets, same age-specific comprehension ranking and alphabetical tie-break. Re-extraction of all 4,356 word/month rows exactly matches the prior analysis. No orientation eligibility filter is applied.','', 'Keep concrete physical non-animal objects, including clothing, furniture, vehicles, toys and foods. Exclude animal concepts, people/body parts, scenes and ambiguous/broad meanings under the original rules. Wordbank categorizes teddybear under animals, but the mapped Teddy bear class is a stuffed toy and is retained. This removes 16 actual animal classes.','', '## Replacement words','',', '.join(c['new_words'])+'.','', '## Removed animal words','',', '.join(c['removed_old_animal_words'])+'.','','## Rebuilt cumulative curriculum','','| Age (months) | Total classes | Added words |','|---:|---:|---|']
 for s in c['primary_original_1000_source_gate']['stages']:lines.append(f'| {s["age_months"]} | {s["actual_classes"]} | '+', '.join(s['new_words'])+' |')
 lines+=['','## Image sourcing','','The original source reserve is 1,000 distinct human-positive Open Images photographs/class. One documented exception is noodles: 919 human-positive source photos (891 after ownership and old-photo exclusion), replacing the unreliable button mapping. This changes the source reserve, while preserving the target of 700 photos/class. New source aliases used in the selected inventory: boots → Boot; purse → Handbag; noodles → Noodle. The button source includes interface screenshots and tiny clothing buttons in portraits; this source match is excluded. Synonyms and all exclusions are recorded in ELIGIBILITY_AUDIT.json. Generic room/scene labels and animal-food sense mismatches remain excluded.','', f'Retain 33,600 existing object-photo records (48 × 700), preserving their original train/valid/test assignments. Fetch 11,200 replacement photos (16 × 700), targeting the original 500 train + 100 valid + 100 test pictures/class, 44,800 total. The {p['replacement_candidates']:,} replacement candidates have official URL, author, license and rotation metadata. All 44,800 old image IDs are excluded from replacement candidates, including removed animal photos. Every replacement has at least 800 uniquely assigned candidates after this exclusion.','', 'Object boxes, when available, follow the original quality filters and 10% context padding. Preserve the whole target in a 224 × 224 padded RGB image; correct source orientation from official metadata. This normalization does not impose an orientation eligibility filter. Exact source IDs, source-byte hashes and normalized-pixel hashes must be unique across the new dataset.','',f'The review page currently displays {len(images)} hash-verified retained/replacement examples. These are source previews, not a new fixed Yin stimulus pool. Full sourcing/decoding, duplicate checks and visual review remain pending. No model training or experiment has been launched.','', '## Files','','- [Object words](OBJECT_WORDS.csv): word, entry-age comprehension numerator/denominator, mapping, source count.','- [Comprehension at all five ages](WORD_COMPREHENSION.csv).','- [Review gallery](index.html) and [contact sheet](object-inventory.jpg).','- [Source plan](IMAGE_PLAN.json); [complete word audit](ELIGIBILITY_AUDIT.json).','','Source code and resumable photo downloader: '+str(ROOT)+'.','', 'Source: pinned Wordbank commit e9401fc8f54579f5283bde391cc70cd5569f0d8f, WG SHA256 8fa08d7b716a06e1ddad3ccb4a72ef98235dddf146f605df07c0f2fd7e50a8f5. Parent reports measure word comprehension, not visual recognition age. Open Images photo-disjoint splits do not certify physical-object-instance disjointness.','']
 if sourced:
  lines=[v.replace('full sourcing and visual review remain pending. This dataset is not ready for training or Yin evaluation.','the source manifest is complete: 33,600 retained photo references and 11,200 newly downloaded photos. New photos were decoded and hash-checked; retained photos inherit the pinned previously verified manifest. Full image-by-image visual review was not performed. Training packing and selection of a new Yin stimulus pool are separate steps.').replace('Full sourcing/decoding, duplicate checks and visual review remain pending.','All 44,800 image entries pass the manifest, source-label, split and exact-duplicate audit. Visual checks cover the gallery examples and recorded source-review samples; every full-resolution photo has not been manually reviewed.') for v in lines]
 (OUT/'README.md').write_text('\n'.join(lines))
 page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Non-animal object inventory</title><style>body{font:16px/1.5 system-ui;background:#f5f6f2;color:#172f36;margin:30px}main{max-width:1250px;margin:auto}.note{padding:20px;background:#fff1d3;border-radius:10px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:16px}article{background:white;border:1px solid #d6ded9;padding:18px;border-radius:10px}article p{font-size:13px}h3{margin-top:0}.pending{width:160px;height:160px;display:grid;place-content:center;text-align:center;background:#f6e9cf;color:#965000}a{color:#166247}img{object-fit:contain}nav{margin:22px 0}</style><main><h1>64 non-animal object classes</h1><p>Same Wordbank comprehension method · orientation filtering removed.</p><div class="note"><strong>64 object classes: downloaded photo previews</strong><br>48 retained classes + 16 replacements. Target: 500 train / 100 validation / 100 test photographs per class. Full 700-photo/class sourcing and visual review remain pending. No new training or Yin run has started. New replacement classes are marked [new] on the contact sheet.</div><nav><a href="README.md">Method and status</a> · <a href="WORD_COMPREHENSION.csv">Wordbank rates at each age</a> · <a href="OBJECT_WORDS.csv">Word and image-source list</a></nav><p>Teddy bear is retained as an inanimate toy, despite its Wordbank animal-category tag. Retained photos shown here are existing validation examples; this page does not designate a new Yin pool.</p><section>'+''.join(cards)+'</section></main></html>'
 if sourced:page=page.replace('Full 700-photo/class sourcing and visual review remain pending.','Source manifest complete: 44,800 image entries (33,600 retained photo references + 11,200 new photos), 500 / 100 / 100 per class. Sample previews were visually checked; a full image-by-image manual audit was not performed.')
 (OUT/'index.html').write_text(page)
 print(json.dumps({'gallery':str(OUT/'index.html'),'classes':len(rows),'verified_examples':len(images),'pending_replacement_classes':16-len(preview)//4}))
if __name__=='__main__':main()
