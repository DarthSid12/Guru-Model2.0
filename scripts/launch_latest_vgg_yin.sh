#!/usr/bin/env bash
# Run only missing Yin cells for the final no-AA and bin5 VGG2k checkpoints.
set -euo pipefail

ROOT=/home/siagrawal/combined_lpnet
OUT="$ROOT/runs/yin_latest_vgg_20260927"
PYTHON=/home/siagrawal/miniconda3/envs/themodel2/bin/python
SOURCE="$OUT/source"
AA5_RUN="$ROOT/runs/faces_vgg2k_objects_houses_zubud137_41_houses_lp_16fix_lr0.001_vgg16_bn_aa5_r21vgg2k_vgg16_bn_aa5_s42"
NOAA_RUN="$ROOT/runs/faces_vgg2k_objects_houses_zubud137_41_houses_lp_16fix_lr0.001_vgg16_bn_r21vgg2k_vgg16_bn_s42"
AA5_CK="$AA5_RUN/final_model_20260927_145006.pth"
NOAA_CK="$ROOT/runs/yin_noaa_two_cross_20260927/checkpoint/weights.pth"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/logs"

single() {
    local model="$1" run_dir="$2" checkpoint="$3" gpu="$4" face
    for face in faces_rfwWM64 faces_cfdWM64; do
        if [[ -f "$OUT/single_noise/$model/$face/DONE.json" ]]; then
            echo "[$(date -Is)] already complete: single $model $face"
            continue
        fi
        echo "[$(date -Is)] starting single $model $face on GPU $gpu"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$SOURCE/run_single_noise.py" \
            --model "$model" --run-dir "$run_dir" --checkpoint "$checkpoint" \
            --epoch 124 --face "$face" --device cuda:0 \
            --out-dir "$OUT/single_noise/$model/$face"
    done
}

single_face() {
    local model="$1" run_dir="$2" checkpoint="$3" gpu="$4" face="$5"
    if [[ -f "$OUT/single_noise/$model/$face/DONE.json" ]]; then
        echo "[$(date -Is)] already complete: single $model $face"
        return
    fi
    echo "[$(date -Is)] starting single $model $face on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$SOURCE/run_single_noise.py" \
        --model "$model" --run-dir "$run_dir" --checkpoint "$checkpoint" \
        --epoch 124 --face "$face" --device cuda:0 \
        --out-dir "$OUT/single_noise/$model/$face"
}

cross() {
    local face="$1" gpu="$2"
    if [[ -f "$OUT/two_cross/r21vgg2k_vgg16_bn_aa5_s42/$face/DONE.json" ]]; then
        echo "[$(date -Is)] already complete: two_cross bin5 $face"
        return
    fi
    echo "[$(date -Is)] starting two_cross bin5 $face on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$SOURCE/run_yin_orientation.py" \
        --run-dir "$AA5_RUN" --checkpoint "$AA5_CK" \
        --out-dir "$OUT/two_cross/r21vgg2k_vgg16_bn_aa5_s42/$face" \
        --category "$face" --model r21vgg2k_vgg16_bn_aa5_s42 \
        --epoch 124 --stage final --device cuda:0 --seeds 101-150
}

case "${1:?worker required}" in
    noaa_single)
        single r21vgg2k_vgg16_bn_s42 "$NOAA_RUN" "$NOAA_CK" 1 ;;
    aa5_cross_rfw)
        cross faces_rfwWM64 2
        "$PYTHON" "$ROOT/scripts/prepare_latest_vgg_single.py" --model aa5 --face faces_rfwWM64
        single_face r21vgg2k_vgg16_bn_aa5_s42 "$AA5_RUN" "$AA5_CK" 2 faces_rfwWM64 ;;
    aa5_cross_cfd)
        cross faces_cfdWM64 7
        "$PYTHON" "$ROOT/scripts/prepare_latest_vgg_single.py" --model aa5 --face faces_cfdWM64
        single_face r21vgg2k_vgg16_bn_aa5_s42 "$AA5_RUN" "$AA5_CK" 7 faces_cfdWM64 ;;
    *) echo "Unknown worker $1" >&2; exit 2 ;;
esac
