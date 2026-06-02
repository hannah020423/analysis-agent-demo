from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

def save_mask(mask: np.ndarray, path: Path) -> None:
    """
    한글/공백 경로에서도 저장 가능하도록 cv2.imencode + tofile 사용.
    """
    ext = path.suffix
    success, encoded = cv2.imencode(ext, mask)

    if not success:
        raise ValueError(f"Failed to encode image: {path}")

    encoded.tofile(str(path))


def create_comparison_figure(
    original: np.ndarray,
    preprocessed: np.ndarray,
    masks: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """원본/전처리/대표 segmentation 결과 비교 figure 저장."""
    selected_keys = []

    if "otsu_feedback" in masks:
        selected_keys.append("otsu_feedback")
    elif "otsu" in masks:
        selected_keys.append("otsu")

    if "adaptive" in masks:
        selected_keys.append("adaptive")
    manual_keys = [k for k in masks if k.startswith("manual_")]
    if "manual_50" in manual_keys:
        selected_keys.append("manual_50")
    elif manual_keys:
        selected_keys.append(manual_keys[len(manual_keys) // 2])

    panels = [("Original", original), ("Preprocessed", preprocessed)]
    for key in selected_keys:
        panels.append((key, masks[key]))

    plt.figure(figsize=(12, 6))
    for i, (title, img) in enumerate(panels, start=1):
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def create_area_graph(df: pd.DataFrame, save_path: Path) -> None:
    """Method별 area ratio bar graph 저장."""
    summary = df.groupby("method", as_index=False)["area_ratio"].mean()

    plt.figure(figsize=(8, 4))
    plt.bar(summary["method"], summary["area_ratio"])
    plt.ylabel("Mean area ratio")
    plt.xlabel("Method")
    plt.title("Mean segmented area ratio by method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def create_manual_threshold_graph(df: pd.DataFrame, save_path: Path) -> None:
    """Manual threshold 변화에 따른 area ratio 그래프 저장."""
    manual = df[df["method"].str.startswith("manual_")].copy()
    if manual.empty:
        return
    manual["threshold"] = manual["method"].str.replace("manual_", "", regex=False).astype(int)
    summary = manual.groupby("threshold", as_index=False)["area_ratio"].mean()

    plt.figure(figsize=(8, 4))
    plt.plot(summary["threshold"], summary["area_ratio"], marker="o")
    plt.ylabel("Mean area ratio")
    plt.xlabel("Manual threshold")
    plt.title("Manual threshold sensitivity")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

