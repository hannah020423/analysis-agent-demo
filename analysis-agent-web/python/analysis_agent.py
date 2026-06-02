"""
Analysis Agent Demo
- Input: microscopy cell images
- Output: segmentation masks, metric CSV, area graph, reasoning/critic report

Folder structure:
analysis_agent_demo/
  input/images/      # original images
  input/gt_masks/    # optional ground-truth masks with the same filename
  output/masks/
  output/csv/
  output/figures/
  output/reports/

Run:
python analysis_agent_demo.py --input_dir input/images --gt_dir input/gt_masks --output_dir output

If there is no ground-truth mask:
python analysis_agent_demo.py --input_dir input/images --output_dir output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

import tkinter as tk
from tkinter import filedialog

from config import IMAGE_EXTS, MANUAL_THRESHOLDS
from dataset_loader import (
    list_images,
    get_patient_id,
    read_color,
    resize_to_width,
    find_gt_mask_with_roi,
    natural_sort_key,
)
from roi_module import auto_roi_mask
from segmentation_module import (
    preprocess_image,
    apply_otsu_with_feedback,
    segment_adaptive,
    segment_manual,
    maybe_invert_mask,
)
from metrics_module import (
    calculate_area,
    calculate_area_ratio,
    calculate_r2_from_masks,
    calculate_dice_iou,
    calculate_regression_summary,
)

from visualization_module import (
    save_mask,
    create_comparison_figure,
    create_area_graph,
    create_manual_threshold_graph,
)

from reasoning_module import (
    reasoning_report,
    critic_report,
    create_llm_prompt,
)

from planner_agent import create_plan

from utils import ensure_dirs

def process_image(
    image_path: Path,
    input_dir: Path,
    gt_dir: Optional[Path],
    dirs: Dict[str, Path],
) -> List[Dict]:
    patient_id = get_patient_id(image_path, input_dir)


    # 원본 컬러 이미지 읽기
    original_color = read_color(image_path)

    # 기존 코드처럼 resize
    original_color = resize_to_width(original_color)

    # ROI 자동 탐색
    roi_img, roi_mask, plate_circle = auto_roi_mask(original_color)

    # 기존 코드 방식 preprocessing
    preprocessed = preprocess_image(roi_img)

    gt_mask = find_gt_mask_with_roi(
        gt_dir=gt_dir,
        input_dir=input_dir,
        image_path=image_path,
        plate_circle=plate_circle,
        target_shape=preprocessed.shape
    )
    if gt_mask is not None and gt_mask.shape != preprocessed.shape:
        gt_mask = cv2.resize(gt_mask, (preprocessed.shape[1], preprocessed.shape[0]), interpolation=cv2.INTER_NEAREST)

    gt_area_px = calculate_area(gt_mask) if gt_mask is not None else None

    masks: Dict[str, np.ndarray] = {}
    results: List[Dict] = []

    # Otsu
    otsu_mask, otsu_thr, adjusted_thr, rethresholded = apply_otsu_with_feedback(preprocessed)
    otsu_mask = maybe_invert_mask(otsu_mask)

    masks["otsu_feedback"] = otsu_mask

    # Adaptive
    adaptive_mask = segment_adaptive(preprocessed)
    adaptive_mask = maybe_invert_mask(adaptive_mask)
    masks["adaptive"] = adaptive_mask

    # Manual thresholds
    for thr in MANUAL_THRESHOLDS:
        m = segment_manual(preprocessed, thr)
        m = maybe_invert_mask(m)
        masks[f"manual_{thr}"] = m

    # Save masks and collect metrics
    image_mask_dir = dirs["masks"] / patient_id / image_path.stem
    image_mask_dir.mkdir(parents=True, exist_ok=True)



    for method, mask in masks.items():
        save_mask(mask, image_mask_dir / f"{image_path.stem}_{method}.png")

        if gt_mask is not None:
            dice, iou = calculate_dice_iou(mask, gt_mask)
        else:
            dice, iou = None, None
            
        results.append(
            {
                "patient_id": patient_id,
                "image": image_path.name,
                "method": method,
                "otsu_threshold": otsu_thr if method == "otsu_feedback" else None,
                "area_px": calculate_area(mask),
                "area_ratio": calculate_area_ratio(mask),
                "gt_area_px": gt_area_px,
                "r2": calculate_r2_from_masks(mask, gt_mask),
                "adjusted_threshold": adjusted_thr if method == "otsu_feedback" else None,
                "rethresholded": rethresholded if method == "otsu_feedback" else False,
                "dice": dice,
                "iou": iou,
            }
        )

    patient_fig_dir = dirs["figures"] / patient_id
    patient_fig_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison figure
    create_comparison_figure(
        original=original_color,
        preprocessed=preprocessed,
        masks=masks,
        save_path=patient_fig_dir / f"{image_path.stem}_comparison.png"
    )

    return results


def run_demo(
    input_dir: Path,
    output_dir: Path,
    gt_dir: Optional[Path] = None,
    user_order: str = ""
) -> None:
    dirs = ensure_dirs(output_dir)
    images = list_images(input_dir)

    plan = create_plan()
    with open(dirs["reports"] / "planner_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    all_results: List[Dict] = []
    for img_path in images:
        print(f"[Execution] Processing: {img_path.name}")
        all_results.extend(process_image(img_path, input_dir, gt_dir, dirs))

    df = pd.DataFrame(all_results)

    print("[DEBUG] patient ids:")
    print(df["patient_id"].unique())

    print("[DEBUG] total rows:", len(df))

    csv_path = dirs["csv"] / "analysis_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # -----------------------------
    # 환자별 CSV 저장 추가
    # -----------------------------
    print(df["patient_id"].value_counts())
    for patient_id, patient_df in df.groupby("patient_id"):

        patient_csv = (
            dirs["csv"]
            / f"{patient_id}_analysis_results.csv"
        )

        patient_df.to_csv(
            patient_csv,
            index=False,
            encoding="utf-8-sig"
        )

    regression_summary = calculate_regression_summary(df)
    regression_summary_path = dirs["csv"] / "regression_summary.csv"
    regression_summary.to_csv(regression_summary_path, index=False, encoding="utf-8-sig")

    create_area_graph(df, dirs["figures"] / "method_area_ratio_summary.png")
    create_manual_threshold_graph(df, dirs["figures"] / "manual_threshold_sensitivity.png")

    reasoning = reasoning_report(df)
    critic = critic_report(df)
    llm_prompt = create_llm_prompt(
        df,
        user_order
    )

    report_text = "\n\n".join([reasoning, critic])
    report_path = dirs["reports"] / "analysis_agent_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\n================ Analysis Agent Demo Complete ================")
    print(f"CSV saved: {csv_path}")
    print(f"Figures saved: {dirs['figures']}")
    print(f"Masks saved: {dirs['masks']}")
    print(f"Report saved: {report_path}")
    print("\n" + reasoning)
    print("\n" + critic)

    print("[DEBUG] total images:", len(images))

    for p in images[:20]:
        print("[DEBUG IMAGE]", p.relative_to(input_dir))

def select_folder(title="Select Folder"):
    root = tk.Tk()
    root.withdraw()  # tkinter 기본 창 숨김
    folder = filedialog.askdirectory(title=title)
    return Path(folder) if folder else None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis Agent Demo for cell segmentation")
    parser.add_argument("--no_gui", action="store_true", help="Run without GUI folder selection")
    parser.add_argument("--input_dir", type=str, default="input/images", help="Input image folder")
    parser.add_argument("--gt_dir", type=str, default=None, help="Optional ground truth mask folder")
    parser.add_argument("--output_dir", type=str, default="output", help="Output folder")
    parser.add_argument("--user_order", type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if getattr(args, "no_gui", False):
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        gt_dir = Path(args.gt_dir) if args.gt_dir else None

        run_demo(
            input_dir=input_dir,
            output_dir=output_dir,
            gt_dir=gt_dir,
            user_order=args.user_order
        )

    else:
        print("=== Analysis Agent Demo ===")

        input_dir = select_folder("Select Input Image Folder")

        if input_dir is None:
            print("No input folder selected.")
            exit()

        use_gt = input("Use Ground Truth masks? (y/n): ").lower()

        gt_dir = None
        if use_gt == "y":
            gt_dir = select_folder("Select GT Mask Folder")

        output_dir = select_folder("Select Output Folder")

        if output_dir is None:
            print("No output folder selected.")
            exit()

        run_demo(
            input_dir=input_dir,
            output_dir=output_dir,
            gt_dir=gt_dir
        )