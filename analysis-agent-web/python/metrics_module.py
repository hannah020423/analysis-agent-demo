from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import cv2


def calculate_area(mask: np.ndarray) -> int:
    return int(np.sum(mask > 0))


def calculate_area_ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask > 0))


def calculate_r2_from_masks(pred_mask: np.ndarray, gt_mask: Optional[np.ndarray]) -> Optional[float]:
    """
    픽셀 단위 mask 비교를 통한 R².
    GT가 없으면 None 반환.
    """
    if gt_mask is None:
        return None

    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    y_true = (gt_mask > 0).astype(np.float32).ravel()
    y_pred = (pred_mask > 0).astype(np.float32).ravel()

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0:
        return None
    return 1.0 - (ss_res / ss_tot)

def calculate_dice_iou(mask, gt_mask):

    mask_bin = (mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(mask_bin, gt_bin).sum()
    union = np.logical_or(mask_bin, gt_bin).sum()

    dice = (2.0 * intersection) / (
        mask_bin.sum() + gt_bin.sum() + 1e-8
    )

    iou = intersection / (union + 1e-8)

    return dice, iou




def calculate_regression_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    method별 GT area와 predicted area 간 regression summary 계산.
    x = gt_area_px
    y = area_px
    """

    rows = []

    for method, group in df.groupby("method"):
        g = group.dropna(subset=["gt_area_px", "area_px"])

        if len(g) < 2:
            rows.append({
                "method": method,
                "n": len(g),
                "slope": None,
                "intercept": None,
                "r2_regression": None,
                "mean_area_ratio": g["area_ratio"].mean() if len(g) > 0 else None,
                "std_area_ratio": g["area_ratio"].std() if len(g) > 1 else None,
                "mean_dice": g["dice"].mean() if "dice" in g.columns and len(g) > 0 else None,
                "std_dice": g["dice"].std() if "dice" in g.columns and len(g) > 1 else None,
                "mean_iou": g["iou"].mean() if "iou" in g.columns and len(g) > 0 else None,
                "std_iou": g["iou"].std() if "iou" in g.columns and len(g) > 1 else None,
            })
            continue

        x = g["gt_area_px"].astype(float).values
        y = g["area_px"].astype(float).values

        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = None if ss_tot == 0 else 1 - (ss_res / ss_tot)

        rows.append({
            "method": method,
            "n": len(g),
            "slope": slope,
            "intercept": intercept,
            "r2_regression": r2,
            "mean_area_ratio": g["area_ratio"].mean(),
            "std_area_ratio": g["area_ratio"].std(),
            "mean_dice": g["dice"].mean() if "dice" in g.columns else None,
            "std_dice": g["dice"].std() if "dice" in g.columns else None,
            "mean_iou": g["iou"].mean() if "iou" in g.columns else None,
            "std_iou": g["iou"].std() if "iou" in g.columns else None,
        })

    return pd.DataFrame(rows)


