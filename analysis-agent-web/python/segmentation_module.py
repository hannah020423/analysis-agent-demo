from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import math

def calculate_correction(gray_img: np.ndarray, otsu_val: int) -> int:
    bright_ratio = np.sum(gray_img > otsu_val) / gray_img.size

    edges = cv2.Canny(gray_img, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    _, binary = cv2.threshold(gray_img, otsu_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_count = len(contours)
    mean_area = np.mean([cv2.contourArea(c) for c in contours]) if contours else 0

    bright_score = np.clip((bright_ratio - 0.2) * 5, 0, 1)
    edge_score = 1 - np.clip(edge_density * 10, 0, 1)
    area_score = np.clip((mean_area - 200) / 800, 0, 1)
    contour_score = 1 - np.clip(contour_count / 50, 0, 1)

    score = (bright_score + edge_score + area_score + contour_score) / 4
    return int(score * 60)

def apply_otsu_with_feedback(img: np.ndarray):
    """
    OTSU 수행 후 contour 기반 feedback.
    너무 큰 object가 잡히면 adjusted OTSU로 재-threshold.
    """
    otsu_val, otsu_mask = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    need_adjustment = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0:
            radius = math.sqrt(area / math.pi)
            if radius > 225:
                need_adjustment = True
                break

    adjusted_val = None

    if need_adjustment:
        correction = calculate_correction(img, otsu_val)
        adjusted_val = min(int(otsu_val + correction), 255)

        _, adjusted_mask = cv2.threshold(img, adjusted_val, 255, cv2.THRESH_BINARY)
        return adjusted_mask, int(otsu_val), adjusted_val, True

    return otsu_mask, int(otsu_val), adjusted_val, False


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    기존 코드 방식:
    GaussianBlur + Sharpening
    """

    blur = cv2.GaussianBlur(
        img,
        (3, 3),
        1
    )

    sharpened = cv2.addWeighted(
        img,
        1.5,
        blur,
        -0.5,
        0
    )

    return sharpened


def segment_otsu(img: np.ndarray) -> Tuple[np.ndarray, int]:
    threshold_value, mask = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask, int(threshold_value)


def segment_adaptive(img: np.ndarray) -> np.ndarray:

    mask = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        1501,
        -30
    )

    return mask


def segment_manual(img: np.ndarray, threshold: int) -> np.ndarray:
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return mask


def maybe_invert_mask(mask: np.ndarray) -> np.ndarray:
    """
    세포 영역이 흰색이 되도록 보정.
    흰색 비율이 너무 크면 배경이 흰색으로 잡혔다고 보고 반전.
    """
    white_ratio = np.mean(mask > 0)
    if white_ratio > 0.70:
        return cv2.bitwise_not(mask)
    return mask
