#!/usr/bin/env bash
# Chain for model 4: wait for make_faces_vgg2k.py -> pack -> verify -> launch.
set -euo pipefail
cd /home/siagrawal/combined_lpnet
echo "[$(date +%T)] waiting for store build"
until grep -qE "^done:|Traceback|wrote .* of" logs/make_faces_vgg2k.log; do sleep 10; done
grep -q "^done:" logs/make_faces_vgg2k.log || { echo "BUILD FAILED"; exit 1; }
echo "[$(date +%T)] build done; packing"
python data_preparation/preprocess_fixations.py --categories faces_vgg2k --raw-root faces_vgg2k=data/faces_vgg2k \
  --splits train valid test --num-coords 32 --edge-margin 0.15 \
  --devices cuda:0 cuda:2 --batch-size 64 --num-workers 8
echo "[$(date +%T)] packed; verifying"
python - <<'PY'
import json, collections, numpy as np, sys
order=[l.strip() for l in open("data/vggface2_raw/faces_vgg2k_order.txt")]
ok=True
for sp,want in (("train",200),("valid",12),("test",12)):
    m=json.load(open(f"fixation_data/faces_vgg2k/{sp}/meta.json"))
    c=collections.Counter(m["labels"]); cnt=sorted(c.values())
    co=np.load(f"fixation_data/faces_vgg2k/{sp}/coords.npy",mmap_mode="r")
    x,y=np.asarray(co[...,0]),np.asarray(co[...,1])
    out=int(((x<33.6)|(x>190.4)|(y<33.6)|(y>190.4)).sum())
    print(f"{sp}: {len(m['classes'])} classes, {len(m['labels'])} imgs, per-class {cnt[0]}-{cnt[-1]}, "
          f"edge_margin {m.get('edge_margin')}, coords outside central 70%: {out}")
    ok &= len(m["classes"])==2048 and set(m["classes"])==set(order) and m.get("edge_margin")==0.15 and out==0
sys.exit(0 if ok else "VERIFY FAILED")
PY
echo "[$(date +%T)] verified; launching model 4 (vgg16_bn_aa) on cuda:0"
setsid nohup bash training/train_r21_vgg2k.sh cuda:0 42 vgg16_bn_aa \
  > logs/r21vgg2k_vgg16bnaa_s42.log 2>&1 < /dev/null & disown
echo "[$(date +%T)] launched"
