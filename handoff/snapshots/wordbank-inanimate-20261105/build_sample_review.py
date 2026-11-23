"""Deterministic random photo QA samples, independent of model/Yin results."""
import hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'output/wordbank-inanimate-20261105/source-quality-review'
OUT.mkdir(parents=True,exist_ok=True)
c=json.loads((ROOT/'CURRICULUM.json').read_text());new={r['class_name'] for r in c['primary_original_1000_source_gate']['classes'] if r['word'] in c['new_words']}
rows=json.loads((ROOT/'PHOTO_SPLITS.json').read_text())['images'];font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',13)
samples=[]
for cl in sorted(new):
 selected=sorted([r for r in rows if r['class_name']==cl],key=lambda r:hashlib.sha256(('source-qc-20261105|'+r['image_id']).encode()).digest())[:8]
 samples.extend(selected)
for page in range(2):
 im=Image.new('RGB',(8*160,8*185),'white');d=ImageDraw.Draw(im)
 for j,r in enumerate(samples[page*64:(page+1)*64]):
  x=j%8*160;y=j//8*185
  with Image.open(r['image_path']) as photo:photo.load();im.paste(photo.resize((148,148)),(x+6,y+3))
  d.text((x+4,y+153),r['class_name'],font=font,fill='black');d.text((x+4,y+169),r['image_id'][:13],font=font,fill='black')
 im.save(OUT/f'sample-{page+1}.jpg',quality=95)
(OUT/'SAMPLES.json').write_text(json.dumps({'sampling':'8 per new class, smallest SHA256(source-qc-20261105|image_id), independent of model performance; full dataset not exhaustively manually reviewed','images':samples},indent=2)+'\n')
print('128 sample photos in',OUT)
