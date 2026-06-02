import cv2

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def auto_roi_mask(img: np.ndarray):
    """
    기존 코드의 masking() 기반 ROI 자동 탐색.
    """

    # 컬러 → grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    src = cv2.add(gray, 220)
    src = cv2.GaussianBlur(src, (9, 9), 2)

    kernel = np.ones((3, 3), np.uint8)
    src = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations=2)

    hist = cv2.calcHist([src], [0], None, [256], [0, 256]).ravel()
    sorted_hist = np.argsort(hist)[::-1]

    plate_circle = None
    min_score = float("inf")

    for k in range(8, -1, -1):

        second_dark_value = sorted_hist[k]

        for j in range(8, 0, -1):

            ten_light_value = sorted_hist[j]

            _, binary_image = cv2.threshold(
                src,
                second_dark_value,
                255,
                cv2.THRESH_BINARY
            )

            _, binary_image2 = cv2.threshold(
                src,
                ten_light_value,
                255,
                cv2.THRESH_BINARY
            )

            result = np.zeros_like(src)

            cv2.bitwise_xor(
                binary_image,
                binary_image2,
                result
            )

            circles = cv2.HoughCircles(
                result,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=2000,
                param1=100,
                param2=10,
                minRadius=225,
                maxRadius=249,
            )

            if circles is not None:

                circles = np.round(circles[0, :]).astype("int")

                img_center = (
                    gray.shape[1] // 2,
                    gray.shape[0] // 2
                )

                for x, y, r in circles:

                    if (
                        x - r >= 0
                        and y - r >= 0
                        and x + r < gray.shape[1]
                        and y + r < gray.shape[0]
                    ):

                        distance_to_center = np.sqrt(
                            (x - img_center[0]) ** 2
                            + (y - img_center[1]) ** 2
                        )

                        circle_mask = np.zeros_like(gray, dtype=np.uint8)

                        cv2.circle(
                            circle_mask,
                            (x, y),
                            r,
                            255,
                            -1
                        )

                        mean_val = cv2.mean(
                            gray,
                            mask=circle_mask
                        )[0]

                        score = (
                            distance_to_center * 0.5
                            + abs(mean_val - 128) * 0.5
                        )

                        if score < min_score:

                            min_score = score
                            plate_circle = (x, y, r)

    if plate_circle is None:
        return gray, np.ones_like(gray, dtype=np.uint8) * 255, None

    x, y, r = plate_circle

    mask = np.zeros_like(gray, dtype=np.uint8)

    cv2.circle(
        mask,
        (x, y),
        r,
        255,
        -1
    )

    masked_img = cv2.bitwise_and(gray, mask)

    return masked_img, mask, plate_circle

def apply_circle_roi_to_image(img: np.ndarray, plate_circle):
    """
    원본 이미지에서 찾은 plate_circle을 GT 이미지에도 동일하게 적용.
    """
    if len(img.shape) == 3:
        gray_or_color = img.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        gray_or_color = img.copy()
        mask = np.zeros_like(img, dtype=np.uint8)

    if plate_circle is None:
        return gray_or_color, np.ones_like(mask, dtype=np.uint8) * 255

    x, y, r = plate_circle

    cv2.circle(mask, (x, y), r, 255, -1)

    if len(img.shape) == 3:
        masked_img = cv2.bitwise_and(gray_or_color, gray_or_color, mask=mask)
    else:
        masked_img = cv2.bitwise_and(gray_or_color, mask)

    return masked_img, mask