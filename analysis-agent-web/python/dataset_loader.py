from pathlib import Path
from typing import List, Optional
import re

import cv2
import numpy as np
import pandas as pd

from config import IMAGE_EXTS
from roi_module import apply_circle_roi_to_image

def natural_sort_key(path):
    """
    파일명을 숫자 기준으로 자연 정렬하기 위한 key 함수.
    예:
    1_1.jpg
    1_2.jpg
    1_10.jpg
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', path.stem)
    ]

def list_images(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = sorted(
        [p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS],
        key=natural_sort_key
    )

    if not images:
        raise FileNotFoundError(f"No image files found in: {input_dir}")

    return images

def get_patient_id(image_path: Path, input_dir: Path) -> str:
    rel = image_path.relative_to(input_dir)

    parts = rel.parts

    for part in parts[:-1]:
        if part not in ["images", "gt_masks"]:
            return part

    return "single_dataset"

def read_grayscale(path: Path) -> np.ndarray:
    """
    한글/공백이 포함된 Windows 경로에서도 이미지를 읽기 위한 함수.
    cv2.imread 대신 np.fromfile + cv2.imdecode 사용.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    return img

def read_color(path: Path) -> np.ndarray:
    """
    원본 컬러 이미지 읽기.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    return img

def resize_to_width(img: np.ndarray, max_width: int = 800) -> np.ndarray:
    """
    기존 코드처럼 폭 800 기준 resize.
    """
    if img.shape[1] > max_width:
        ratio = max_width / img.shape[1]

        img = cv2.resize(
            img,
            (
                int(img.shape[1] * ratio),
                int(img.shape[0] * ratio)
            )
        )

    return img

def find_gt_mask_with_roi(
    gt_dir: Optional[Path],
    input_dir: Path,
    image_path: Path,
    plate_circle,
    target_shape,
) -> Optional[np.ndarray]:

    if gt_dir is None or not gt_dir.exists():
        return None

    rel_path = image_path.relative_to(input_dir)

    candidates = [
        gt_dir / rel_path,
        gt_dir / rel_path.with_name(f"{image_path.stem}_mask{image_path.suffix}"),
        gt_dir / rel_path.with_name(f"{image_path.stem}_gt{image_path.suffix}"),
    ]

    for c in candidates:
        if c.exists():
            gt_img = read_color(c)

            # 원본 이미지와 동일하게 800 기준 resize
            gt_img = resize_to_width(gt_img)

            # 원본 이미지에서 찾은 ROI를 GT에 동일 적용
            gt_roi_img, _ = apply_circle_roi_to_image(gt_img, plate_circle)

            # ROI 안에서 흰색 GT marker 추출
            gt_mask = extract_gt_white_mask(gt_roi_img)

            if gt_mask.shape != target_shape:
                gt_mask = cv2.resize(
                    gt_mask,
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            return gt_mask

    return None

def extract_gt_white_mask(gt_roi_img: np.ndarray) -> np.ndarray:
    """
    기존 코드 방식:
    ROI 적용된 GT 이미지에서 흰색 marker 영역만 추출.
    """
    if len(gt_roi_img.shape) == 2:
        gt_bgr = cv2.cvtColor(gt_roi_img, cv2.COLOR_GRAY2BGR)
    else:
        gt_bgr = gt_roi_img

    lower_white = np.array([100, 100, 100])
    upper_white = np.array([255, 255, 255])

    gt_mask = cv2.inRange(gt_bgr, lower_white, upper_white)

    return gt_mask