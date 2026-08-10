from pathlib import Path
import shutil
import re

src = Path("~/Downloads/houses_unfoldered").expanduser()
dst = Path("~/Downloads/houses_unprocessed").expanduser()
dst.mkdir(exist_ok=True)

# matches things like:
# 1_frontal.jpg, 123_frontal.jpg
pattern = re.compile(r"(\d+)_frontal\.jpg")

for img in src.glob("*_frontal.jpg"):
    m = pattern.match(img.name)
    if not m:
        continue

    print("matched!")
    house_id = int(m.group(1))

    house_dir = dst / f"house{house_id:04d}"
    house_dir.mkdir(exist_ok=True)

    shutil.copy2(img, house_dir / "frontal.jpg")

print("Done!")
