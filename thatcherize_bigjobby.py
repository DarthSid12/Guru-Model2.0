import cv2
import numpy as np
import argparse
import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# Load MediaPipe Landmarker
# ============================================================

base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# ============================================================
# PARAMETERS (BigJobby-style fixed values)
# ============================================================

EYE_PAD = 0.45
MOUTH_PAD = 0.40

FEATHER_FRAC = 0.15
OPACITY = 1.0


# ============================================================
# MediaPipe landmark groups
# ============================================================

LEFT_EYE = [
    33, 133, 160, 159, 158,
    157, 173, 246, 163,
    144, 145, 153, 154
]

RIGHT_EYE = [
    362, 263, 387, 386,
    385, 384, 398,
    373, 374, 380,
    381, 382
]

MOUTH = [
    61, 146, 91, 181,
    84, 17, 314,
    405, 321, 375,
    291, 409, 270,
    269, 267
]


# ============================================================
# Bounding box from landmarks
# ============================================================

def get_region_bbox(
        landmarks,
        indices,
        W,
        H,
        pad_frac):

    pts = []

    for i in indices:
        lm = landmarks[i]

        pts.append([
            lm.x * W,
            lm.y * H
        ])

    pts = np.array(pts)

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)


    pw = (max_x-min_x)*pad_frac
    ph = (max_y-min_y)*pad_frac


    return (
        max(0,int(min_x-pw)),
        max(0,int(min_y-ph)),
        min(W-1,int(max_x+pw)),
        min(H-1,int(max_y+ph))
    )


# ============================================================
# Eye centre
# ============================================================

def landmark_center(
        landmarks,
        indices,
        W,
        H):

    xs=[]
    ys=[]

    for i in indices:
        xs.append(
            landmarks[i].x*W
        )
        ys.append(
            landmarks[i].y*H
        )

    return (
        np.mean(xs),
        np.mean(ys)
    )


# ============================================================
# Rotate image
# ============================================================

def rotate_image(img, angle):

    H, W = img.shape[:2]

    rad = math.radians(angle)

    cos = abs(math.cos(rad))
    sin = abs(math.sin(rad))

    newW = int(W*cos + H*sin)
    newH = int(W*sin + H*cos)

    M = np.array([
        [math.cos(rad), -math.sin(rad), 0],
        [math.sin(rad),  math.cos(rad), 0]
    ], dtype=np.float32)


    # move old center to new center

    M[0,2] = newW/2 - (
        M[0,0]*W/2 +
        M[0,1]*H/2
    )

    M[1,2] = newH/2 - (
        M[1,0]*W/2 +
        M[1,1]*H/2
    )


    rotated=cv2.warpAffine(
        img,
        M,
        (newW,newH)
    )

    return rotated,M

# ============================================================
# Transform landmarks after rotation
# ============================================================

def transform_landmarks(
        landmarks,
        M,
        oldW,
        oldH,
        newW,
        newH):

    output=[]

    for lm in landmarks:

        p=np.array([
            lm.x*oldW,
            lm.y*oldH,
            1
        ])

        q=M @ p


        class LM:
            pass

        n=LM()
        n.x=q[0]/newW
        n.y=q[1]/newH

        output.append(n)


    return output



# ============================================================
# Feathered flipped patch
# ============================================================

def flip_region(
        img,
        bbox,
        feather):

    x1,y1,x2,y2=bbox


    patch=img[
        y1:y2,
        x1:x2
    ].copy()


    if patch.size==0:
        return


    flipped=cv2.flip(
        patch,
        0
    )


    h,w=flipped.shape[:2]


    mask=np.zeros(
        (h,w),
        dtype=np.float32
    )


    cv2.ellipse(
        mask,
        (w//2,h//2),
        (w//2,h//2),
        0,
        0,
        360,
        1,
        -1
    )


    # feather edge

    blur=int(
        min(w,h)*feather
    )

    if blur>0:
        mask=cv2.GaussianBlur(
            mask,
            (blur|1,blur|1),
            0
        )


    mask=mask[:,:,None]


    target=img[
        y1:y2,
        x1:x2
    ]


    blended=(
        flipped*mask +
        target*(1-mask)
    )


    img[
        y1:y2,
        x1:x2
    ]=blended.astype(
        np.uint8
    )



# ============================================================
# Thatcher transform
# ============================================================

def thatcherize(img, face_landmarks):


    H,W=img.shape[:2]


    # ------------------------------
    # roll angle
    # ------------------------------

    left=landmark_center(
        face_landmarks,
        LEFT_EYE,
        W,H
    )

    right=landmark_center(
        face_landmarks,
        RIGHT_EYE,
        W,H
    )


    angle=math.atan2(
        right[1]-left[1],
        right[0]-left[0]
    )


    print(
        "roll:",
        angle
    )


    # ------------------------------
    # rotate upright
    # ------------------------------

    upright,M=rotate_image(
        img,
        -angle
    )


    h2,w2=upright.shape[:2]


    lm2=transform_landmarks(
        face_landmarks,
        M,
        W,H,
        w2,h2
    )


    # ------------------------------
    # regions
    # ------------------------------

    eyeL=get_region_bbox(
        lm2,
        LEFT_EYE,
        w2,h2,
        EYE_PAD
    )


    eyeR=get_region_bbox(
        lm2,
        RIGHT_EYE,
        w2,h2,
        EYE_PAD
    )


    mouth=get_region_bbox(
        lm2,
        MOUTH,
        w2,h2,
        MOUTH_PAD
    )


    debug=upright.copy()


    for box in [eyeL,eyeR,mouth]:

        x1,y1,x2,y2=box

        cv2.rectangle(
            debug,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


    cv2.imwrite(
        "debug_boxes.png",
        debug
    )


    # ------------------------------
    # flip regions
    # ------------------------------

    result=upright.copy()


    flip_region(
        result,
        eyeL,
        FEATHER_FRAC
    )

    flip_region(
        result,
        eyeR,
        FEATHER_FRAC
    )

    flip_region(
        result,
        mouth,
        FEATHER_FRAC
    )


    # ------------------------------
    # rotate back
    # ------------------------------

    restored,_=rotate_image(
        result,
        angle
    )


    return restored



# ============================================================
# MAIN
# ============================================================

def main():

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    args=parser.parse_args()


    img=cv2.imread(
        args.input
    )


    rgb=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img
    )

    result = landmarker.detect(mp_image)

    landmarks = result.face_landmarks[0]

    output=thatcherize(
        img,
        landmarks
    )


    cv2.imwrite(
        "thatcher_output.png",
        output
    )


    # inverted version

    inverted=cv2.rotate(
        output,
        cv2.ROTATE_180
    )

    cv2.imwrite(
        "thatcher_inverted.png",
        inverted
    )


    print(
        "done"
    )



if __name__=="__main__":
    main()