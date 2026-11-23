"""Render animal-flagged teddy-bear photos for physical-toy visual adjudication."""
import concurrent.futures,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'output/wordbank-object-curation-20261106/toy-review';OUT.mkdir(parents=True,exist_ok=True)
rows=sorted([r for r in json.loads((ROOT/'ANIMAL_SOURCE_FLAGS.json').read_text())['parent_flags'] if r['class_name']=='teddybear'],key=lambda r:r['image_id'])
(ROOT/'TOY_REVIEW_INDEX.json').write_text(json.dumps({'images':[{'review_index':i,**r} for i,r in enumerate(rows)]},indent=2)+'\n')
def load(r):
 p=Path(r['image_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['png_sha256']
 with Image.open(p) as im:im.load();return im.convert('RGB').resize((140,140))
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',13)
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
 for start in range(0,len(rows),64):
  group=rows[start:start+64];images=list(pool.map(load,group));sheet=Image.new('RGB',(8*150,8*170),'white');d=ImageDraw.Draw(sheet)
  for i,(r,im) in enumerate(zip(group,images)):
   x=i%8*150;y=i//8*170;sheet.paste(im,(x+5,y));d.text((x+4,y+143),str(start+i)+' '+r['image_id'][:10],font=font,fill='black')
  sheet.save(OUT/f'{start//64+1:02d}.jpg',quality=95);print('Toy review sheet',start//64+1,'ready',flush=True)
