"""
thatcherize.py

Stable Thatcherization using MediaPipe FaceLandmarker.

Key idea:
- DO NOT warp the whole image
- ONLY flip local facial regions in-place
- Avoid sequential geometric transforms that compound distortions
"""

import numpy as np
import cv2
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================================================
# MEDIAPIPE SETUP
# ==================================================

base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)


# ==================================================
# LANDMARK INDICES (MediaPipe FaceMesh topology)
# ==================================================

LEFT_EYE = (
    246, 161, 160, 159, 158, 157, 173, 33,
    7, 163, 144, 145, 153, 154, 155, 133
)

RIGHT_EYE = (
    466, 388, 387, 386, 385, 384, 398, 263,
    249, 390, 373, 374, 380, 381, 382, 362
)

MOUTH = (
    61, 185, 40, 39, 37, 0, 267, 269,
    270, 409, 291, 146, 91, 181, 84,
    17, 314, 405, 321, 375
)


# ==================================================
# LANDMARK UTIL
# ==================================================

def landmarks_to_pixels(landmarks, indices, w, h):
    pts = []
    for i in indices:
        lm = landmarks[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)


# ==================================================
# CORE LOCAL TRANSFORM
# ==================================================

def flip_region(img, points):
    """
    Classic Thatcher-style operation:
    flip ONLY the pixels inside the bounding box of landmarks.
    """

    x, y, w, h = cv2.boundingRect(points)

    # clamp bounds (safety)
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)

    patch = img[y0:y1, x0:x1].copy()

    # horizontal flip (true Thatcher-style eye inversion effect)
    patch = cv2.flip(patch, 0)

    img[y0:y1, x0:x1] = patch

    return img


# ==================================================
# THATCHERIZE
# ==================================================

def thatcherize(img):
    """
    Apply Thatcher effect:
    - flip left eye
    - flip right eye
    - flip mouth
    """

    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        print("landmark extraction failed!")
        return img.copy()

    lm = result.face_landmarks[0]

    left_eye = landmarks_to_pixels(lm, LEFT_EYE, w, h)
    right_eye = landmarks_to_pixels(lm, RIGHT_EYE, w, h)
    mouth = landmarks_to_pixels(lm, MOUTH, w, h)

    # IMPORTANT: NO chained warps — operate in original coordinate space
    rgb = flip_region(rgb, left_eye)
    rgb = flip_region(rgb, right_eye)
    rgb = flip_region(rgb, mouth)

    return Image.fromarray(rgb)