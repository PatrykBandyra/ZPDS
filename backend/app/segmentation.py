from typing import BinaryIO

import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageColor

MASK_COLOR = (255, 255, 255)  # white

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def perform_selfie_segmentation_with_bg_swap(image: BinaryIO, bg_image: BinaryIO) -> np.ndarray:
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
        bg_img = cv2.imdecode(np.frombuffer(bg_image.read(), np.uint8), 1)

        image_height, image_width, _ = img.shape
        bg_img = cv2.resize(bg_img, (image_width, image_height))

        return perform_selfie_segmentation(img, bg_img, selfie_segmentation)


def perform_selfie_segmentation_white_bg(image: BinaryIO, bg_color: str) -> np.ndarray:
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
        bg_img = np.zeros(img.shape, dtype=np.uint8)
        try:
            bg_img[:] = ImageColor.getcolor(bg_color, 'RGB') if bg_color is not None else MASK_COLOR
        except ValueError:
            bg_img[:] = MASK_COLOR

        return perform_selfie_segmentation(img, bg_img, selfie_segmentation, color=True)


def perform_selfie_segmentation(img: np.ndarray,
                                bg_img: np.ndarray,
                                selfie_segmentation: mp_selfie_segmentation.SelfieSegmentation,
                                color: bool = False) -> np.ndarray:
    results = selfie_segmentation.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    condition = np.stack((cv2.blur(results.segmentation_mask, (5, 5)),) * 3, axis=-1) > 0.6
    if color:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR)
    out_img = np.where(condition, img, bg_img)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    return out_img
