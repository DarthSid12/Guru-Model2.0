import cv2
import numpy as np
import argparse
import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

"""
Credit to https://bigjobby.com/optical/Thatcher/ for inspiring this script!
"""

from PIL import Image

# ============================================================
# Load MediaPipe Face Landmarker
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

landmarker = vision.FaceLandmarker.create_from_options(
    options
)

# ============================================================
# Params
# ============================================================

EYE_PAD = 0.45
MOUTH_PAD = 0.30

FEATHER_FRAC = 0.15
OPACITY = 1.0

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

def bigjobby_rotate(img, angle):

    H, W = img.shape[:2]

    cosA = abs(math.cos(angle))
    sinA = abs(math.sin(angle))

    dW = math.ceil(W*cosA + H*sinA)
    dH = math.ceil(W*sinA + H*cosA)

    output = np.zeros(
        (dH,dW,3),
        dtype=np.uint8
    )

    cx_old = W/2
    cy_old = H/2

    cx_new = dW/2
    cy_new = dH/2


    for y in range(dH):
        for x in range(dW):

            # new canvas -> old coordinates

            dx = x - cx_new
            dy = y - cy_new

            src_x = (
                dx*math.cos(angle)
                +
                dy*math.sin(angle)
                +
                cx_old
            )

            src_y = (
                -dx*math.sin(angle)
                +
                dy*math.cos(angle)
                +
                cy_old
            )


            ix=int(round(src_x))
            iy=int(round(src_y))

            if (
                0 <= ix < W
                and
                0 <= iy < H
            ):
                output[y,x]=img[iy,ix]


    return output

def composite_rotated(dst, patch, angle):

    H, W = dst.shape[:2]
    h, w = patch.shape[:2]


    cosA = math.cos(angle)
    sinA = math.sin(angle)


    M = np.array([
        [
            cosA,
            -sinA,
            W/2 - cosA*w/2 + sinA*h/2
        ],
        [
            sinA,
            cosA,
            H/2 - sinA*w/2 - cosA*h/2
        ]
    ], dtype=np.float32)


    rotated = cv2.warpAffine(
        patch,
        M,
        (W,H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0,0)
    )


    # rotated is now BGRA

    alpha = rotated[:,:,3].astype(np.float32) / 255.0

    rgb = rotated[:,:,:3].astype(np.float32)
    dst_rgb = dst.astype(np.float32)

    alpha3 = alpha[:,:,None]

    out = (
        rgb * alpha3 +
        dst_rgb * (1-alpha3)
    )

    dst[:] = np.clip(out,0,255).astype(np.uint8)

def composite_rotated_old(
        dst,
        patch,
        angle):

    H,W = dst.shape[:2]

    h,w = patch.shape[:2]


    cosA = math.cos(angle)
    sinA = math.sin(angle)


    cx_old = w/2
    cy_old = h/2

    cx_new = W/2
    cy_new = H/2


    rotated = np.zeros_like(dst)


    for y in range(h):

        for x in range(w):

            dx = x-cx_old
            dy = y-cy_old


            new_x = (
                dx*cosA
                -
                dy*sinA
                +
                cx_new
            )


            new_y = (
                dx*sinA
                +
                dy*cosA
                +
                cy_new
            )


            ix=int(round(new_x))
            iy=int(round(new_y))


            if (
                0 <= ix < W
                and
                0 <= iy < H
            ):
                rotated[iy,ix]=patch[y,x]


    mask = rotated.sum(axis=2)>0

    dst[mask]=rotated[mask]

def transform_landmarks(
        landmarks,
        angle,
        oldW,
        oldH,
        newW,
        newH):

    cosA = math.cos(angle)
    sinA = math.sin(angle)


    transformed=[]


    for lm in landmarks:

        # original pixel coordinates
        x = (
            lm.x * oldW
            -
            oldW/2
        )

        y = (
            lm.y * oldH
            -
            oldH/2
        )


        class LM:
            pass


        out = LM()


        out.x = (
            x*cosA
            -
            y*sinA
            +
            newW/2
        ) / newW


        out.y = (
            x*sinA
            +
            y*cosA
            +
            newH/2
        ) / newH


        transformed.append(out)

    return transformed

def rotate_image(img, angle_deg):

    H, W = img.shape[:2]

    angle = math.radians(angle_deg)

    cosA = math.cos(angle)
    sinA = math.sin(angle)

    newW = math.ceil(
        abs(W*cosA) +
        abs(H*sinA)
    )

    newH = math.ceil(
        abs(W*sinA) +
        abs(H*cosA)
    )


    # Canvas transform matrix
    M = np.array([
        [
            cosA,
            -sinA,
            newW/2 - cosA*W/2 + sinA*H/2
        ],
        [
            sinA,
            cosA,
            newH/2 - sinA*W/2 - cosA*H/2
        ]
    ], dtype=np.float32)


    rotated = cv2.warpAffine(
        img,
        M,
        (newW,newH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    
    )


    return rotated

def rotate_image_old(img, angle_deg):

    H, W = img.shape[:2]

    angle = math.radians(angle_deg)

    cosA = math.cos(angle)
    sinA = math.sin(angle)

    dW = math.ceil(
        abs(W*cosA) +
        abs(H*sinA)
    )

    dH = math.ceil(
        abs(W*sinA) +
        abs(H*cosA)
    )

    output = np.zeros(
        (dH,dW,3),
        dtype=np.uint8
    )


    oldCX = W / 2
    oldCY = H / 2

    newCX = dW / 2
    newCY = dH / 2


    for y in range(dH):

        for x in range(dW):

            # coordinates relative to new centre

            dx = x - newCX
            dy = y - newCY


            # inverse rotation

            src_x = (
                dx*cosA +
                dy*sinA +
                oldCX
            )

            src_y = (
                -dx*sinA +
                dy*cosA +
                oldCY
            )


            ix = int(round(src_x))
            iy = int(round(src_y))


            if (
                0 <= ix < W
                and
                0 <= iy < H
            ):
                output[y,x] = img[iy,ix]


    return output

def landmark_center(landmarks, indices, W, H):

    sx = 0
    sy = 0

    for i in indices:
        lm = landmarks[i]

        sx += lm.x * W
        sy += lm.y * H

    return (
        sx / len(indices),
        sy / len(indices)
    )


def get_region_bbox(landmarks, indices, imgW, imgH, padFrac):
    minX = float("inf")
    minY = float("inf")
    maxX = float("-inf")
    maxY = float("-inf")

    for i in indices:
        lm = landmarks[i]

        x = lm.x * imgW
        y = lm.y * imgH

        if x < minX:
            minX = x

        if y < minY:
            minY = y

        if x > maxX:
            maxX = x

        if y > maxY:
            maxY = y

    pw = (maxX - minX) * padFrac
    ph = (maxY - minY) * padFrac

    bbox = {
        "x": max(0, math.floor(minX - pw)),
        "y": max(0, math.floor(minY - ph)),
        "x2": min(imgW - 1, math.ceil(maxX + pw)),
        "y2": min(imgH - 1, math.ceil(maxY + ph)),
    }

    bbox["w"] = bbox["x2"] - bbox["x"]
    bbox["h"] = bbox["y2"] - bbox["y"]

    return bbox



def blit_flipped_region(src_img, dst_img, bbox, featherFrac, opacity):
    """
    src_img : source image (numpy array)
    dst_img : destination image (modified in-place)
    bbox    : dictionary returned by get_region_bbox()
    """

    x = bbox["x"]
    y = bbox["y"]
    w = bbox["w"]
    h = bbox["h"]

    if w <= 0 or h <= 0:
        return

    # -------------------------------------------------------
    # Grab source pixels
    # -------------------------------------------------------

    tmp = src_img[y:y+h, x:x+w].copy()

    cv2.imwrite(
        "debug_patch_original.png",
        tmp
    )
 
    # -------------------------------------------------------
    # Flip vertically
    # JS:
    # translate(0,h)
    # scale(1,-1)
    # -------------------------------------------------------

    flipped = cv2.warpAffine(
        tmp,
        np.float32([
            [1, 0, 0],
            [0,-1,h]
        ]),
        (w,h)
    )

    cv2.imwrite(
        "debug_patch_flipped.png",
        flipped
    )

    # -------------------------------------------------------
    # Build elliptical feather mask
    # -------------------------------------------------------

    cx = w / 2.0
    cy = h / 2.0

    rx = w / 2.0
    ry = h / 2.0

    rMin = min(rx, ry)

    scaleX = rx / rMin
    scaleY = ry / rMin

    coreStop = max(0.0, 1.0 - featherFrac)

    #
    # Build mask manually.
    #
    yy, xx = np.mgrid[0:h, 0:w]

    dx = (xx - cx) / scaleX
    dy = (yy - cy) / scaleY

    r = np.sqrt(dx*dx + dy*dy)

    mask = np.zeros((h, w), np.float32)

    #
    # Fully opaque centre
    #
    inside = r <= coreStop * rMin
    mask[inside] = opacity

    #
    # Feather ring
    #
    feather = (r > coreStop * rMin) & (r <= rMin)

    if feather.any():

        t = (
            r[feather] - coreStop * rMin
        ) / (
            rMin - coreStop * rMin
        )

        mask[feather] = opacity * (1.0 - t)

    #
    # Outside ellipse remains zero.
    #

    mask3 = mask[:, :, None]

    # -------------------------------------------------------
    # Apply destination-in equivalent
    #
    # Canvas:
    #
    # flipped
    # destination-in(mask)
    #
    # becomes
    #
    # flipped * mask
    # -------------------------------------------------------

    masked = np.zeros(
        (h,w,4),
        dtype=np.float32
    )

    masked[:,:,:3] = flipped.astype(np.float32)
    masked[:,:,3] = mask * 255

    # -------------------------------------------------------
    # Composite onto destination
    # -------------------------------------------------------

    region = dst_img[y:y+h, x:x+w]

    alpha = mask[:, :, None]


    # -------------------------------------------------------
    # Handle BGRA patch canvas
    # -------------------------------------------------------

    if region.shape[2] == 4:
        # write only the flipped pixels
        region[:,:,:3] = (
            flipped.astype(np.float32)
        ).astype(np.uint8)
    
        # preserve feather as alpha
        region[:,:,3] = (
            mask * 255
        ).astype(np.uint8)
    else:

        blended = (
            flipped.astype(np.float32) * alpha +
            region.astype(np.float32) * (1.0-alpha)
        )

        region[:] = np.clip(
            blended,
            0,
            255
        ).astype(np.uint8)


    dst_img[y:y+h, x:x+w] = region

def process_image(image, faces):
    H, W = image.shape[:2]
    thatcher = image.copy()

    # ----------------------------------------------------
    # Process every detected face
    # ----------------------------------------------------

    for landmarks in faces:

        # ---------------------------------------------
        # Eye centres
        # ---------------------------------------------

        left_eye = landmark_center(
            landmarks,
            LEFT_EYE,
            W,
            H
        )

        right_eye = landmark_center(
            landmarks,
            RIGHT_EYE,
            W,
            H
        )

        # ---------------------------------------------
        # Face roll angle
        #
        # JS:
        # atan2(re.y-le.y, re.x-le.x)
        # ---------------------------------------------

        roll = math.atan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        )

        # ---------------------------------------------
        # Compute expanded canvas
        #
        # identical maths
        # ---------------------------------------------

        cosA = abs(math.cos(roll))
        sinA = abs(math.sin(roll))

        dW = math.ceil(W * cosA + H * sinA)
        dH = math.ceil(W * sinA + H * cosA)

        # ---------------------------------------------
        # De-rotate image
        #
        # equivalent to:
        #
        # ctx.translate(...)
        # ctx.rotate(-roll)
        #
        # ---------------------------------------------

        derotated = rotate_image(
            image,
            -math.degrees(roll)
        )

        # ---------------------------------------------
        # Rotate landmarks using SAME matrix
        # ---------------------------------------------

        derot_landmarks = transform_landmarks(
            landmarks,
            -roll,
            W,
            H,
            dW,
            dH
        )

        # ---------------------------------------------
        # Build BigJobby boxes
        # ---------------------------------------------

        left_box = get_region_bbox(
            derot_landmarks,
            LEFT_EYE,
            dW,
            dH,
            EYE_PAD
        )

        right_box = get_region_bbox(
            derot_landmarks,
            RIGHT_EYE,
            dW,
            dH,
            EYE_PAD
        )

        mouth_box = get_region_bbox(
            derot_landmarks,
            MOUTH,
            dW,
            dH,
            MOUTH_PAD
        )

        # ---------------------------------------------
        # Build transparent patch
        #
        # JS:
        #
        # const patch = document.createElement(...)
        #
        # ---------------------------------------------

        patch = np.zeros(
            (
                derotated.shape[0],
                derotated.shape[1],
                4
            ),
            dtype=np.uint8
        )

        blit_flipped_region(
            derotated,
            patch,
            left_box,
            FEATHER_FRAC,
            OPACITY
        )

        blit_flipped_region(
            derotated,
            patch,
            right_box,
            FEATHER_FRAC,
            OPACITY
        )

        blit_flipped_region(
            derotated,
            patch,
            mouth_box,
            FEATHER_FRAC,
            OPACITY
        )

        # ---------------------------------------------
        # Rotate patch back
        # ---------------------------------------------

        composite_rotated(
            thatcher,
            patch,
            roll
        )

    # ----------------------------------------------------
    # Final outputs
    # ----------------------------------------------------

    inverted = cv2.rotate(
        thatcher,
        cv2.ROTATE_180
    )

    return thatcher

def thatcherize(input):

    # -------------------------------------------------------
    # Load image
    # -------------------------------------------------------

    img = cv2.imread(input)

    if img is None:
        raise RuntimeError(
            f"Could not open {input}"
        )

    # -------------------------------------------------------
    # Match BigJobby's working canvas size
    # -------------------------------------------------------

    MAX_W = 600
    
    h, w = img.shape[:2]

    if w > MAX_W:

        scale = MAX_W / w

        img = cv2.resize(
            img,
            (
                round(w * scale),
                round(h * scale)
            ),
            interpolation=cv2.INTER_LINEAR
        )

    # -------------------------------------------------------
    # MediaPipe expects RGB
    # -------------------------------------------------------

    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # -------------------------------------------------------
    # Detect faces
    # -------------------------------------------------------

    result = landmarker.detect(
        mp_image
    )

    if (
        result.face_landmarks is None
        or
        len(result.face_landmarks) == 0
    ):
        return None        

    print(
        f"Detected {len(result.face_landmarks)} face(s)"
    )

    # -------------------------------------------------------
    # Run Thatcher transform
    # -------------------------------------------------------

    result_image = process_image(
        img,
        result.face_landmarks
    )
    
    # OpenCV BGR -> PIL RGB
    result_image = cv2.cvtColor(
        result_image,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(result_image)
    

#"""
if __name__ == "__main__":
    thatcherize("data/faces/128_identities/valid/AdamRippon/30.jpg")
#"""