from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def reasoning_report(df: pd.DataFrame) -> str:
    """LLM Reasoning 역할을 흉내 내는 결과 해석 문장 생성."""
    method_summary = df.groupby("method").agg(
        mean_area_ratio=("area_ratio", "mean"),
        std_area_ratio=("area_ratio", "std"),
        mean_r2=("r2", "mean"),
        mean_dice=("dice", "mean"),
        mean_iou=("iou", "mean"),
    ).reset_index()

    # R²가 있으면 R² 우선, 없으면 area ratio 변동성 기준
    has_r2 = df["r2"].notna().any()

    lines = ["[Reasoning Result]", ""]

    if has_r2:
        best = method_summary.sort_values("mean_r2", ascending=False).iloc[0]
        lines.append(
            f"- 평균 R² 기준으로 '{best['method']}' 방법이 가장 높은 성능을 보였습니다."
        )
    else:
        stable = method_summary.sort_values("std_area_ratio", ascending=True).iloc[0]
        lines.append(
            f"- Ground Truth가 없어 R² 평가는 생략했으며, area ratio 변동성 기준으로 '{stable['method']}' 방법이 가장 안정적으로 나타났습니다."
        )

    lines.append("- Otsu는 threshold를 자동으로 결정하므로 반복 실험 자동화에 유리합니다.")
    lines.append("- Adaptive threshold는 조명 불균형이 있는 이미지에서 비교 대상으로 유용합니다.")
    lines.append("- Manual threshold는 threshold 값에 따라 결과가 크게 달라질 수 있어 민감도 확인이 필요합니다.")
    return "\n".join(lines)


def critic_report(df: pd.DataFrame) -> str:
    """Rule-based Critic: 이상 결과 및 재실행 필요 여부 탐지."""
    lines = ["[Critic Result]", ""]
    warnings = []

    # 1) 너무 작은/큰 segmentation 탐지
    abnormal = df[(df["area_ratio"] < 0.01) | (df["area_ratio"] > 0.60)]
    if not abnormal.empty:
        warnings.append(
            f"- {len(abnormal)}개 결과에서 area ratio가 비정상 범위(<1% 또는 >60%)로 탐지되었습니다. ROI 또는 threshold 재검토가 필요합니다."
        )

    # 2) Manual threshold 민감도 탐지
    manual = df[df["method"].str.startswith("manual_")].copy()
    if not manual.empty:
        manual["threshold"] = manual["method"].str.replace("manual_", "", regex=False).astype(int)
        threshold_summary = manual.groupby("threshold")["area_ratio"].mean().sort_index()
        max_jump = threshold_summary.diff().abs().max()
        if pd.notna(max_jump) and max_jump > 0.20:
            warnings.append(
                f"- Manual threshold 변화에 따라 area ratio가 크게 변했습니다(max jump={max_jump:.3f}). threshold 민감도가 높아 재현성 검토가 필요합니다."
            )

    # 3) R²가 낮은 결과 탐지
    if df["r2"].notna().any():
        low_r2 = df[df["r2"].notna() & (df["r2"] < 0.5)]
        if not low_r2.empty:
            warnings.append(
                f"- R² < 0.5인 결과가 {len(low_r2)}개 있습니다. segmentation 실패 가능성을 확인해야 합니다."
            )

    if warnings:
        lines.extend(warnings)
    else:
        lines.append("- 뚜렷한 이상 결과는 탐지되지 않았습니다. 현재 결과는 데모 기준에서 사용 가능합니다.")

    return "\n".join(lines)


def create_llm_prompt(
    df: pd.DataFrame,
    user_order: str = ""
) -> str:

    summary = df.groupby("method").agg(
        mean_area_ratio=("area_ratio", "mean"),
        std_area_ratio=("area_ratio", "std"),
        mean_r2=("r2", "mean"),
        mean_dice=("dice", "mean"),
        std_dice=("dice", "std"),
        mean_iou=("iou", "mean"),
        std_iou=("iou", "std"),
    ).reset_index().to_string(index=False)

    return f"""
당신은 세포 이미지 segmentation 연구를 보조하는 Reasoning Agent입니다.

아래 method별 정량 분석 요약을 기반으로
사용자 오더에 맞게 결과를 해석하세요.

[분석 요약]
{summary}

[사용자 오더]
{user_order if user_order else "Dice, IoU, R²를 기반으로 가장 안정적인 segmentation 방법을 분석하세요."}

분석 결과를 한국어로 자세히 작성하세요.
""".strip()

