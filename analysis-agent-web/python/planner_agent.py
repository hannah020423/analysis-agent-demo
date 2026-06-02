from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from config import MANUAL_THRESHOLDS

def create_plan() -> Dict:
    """LLM Planner가 만든 실행 계획을 흉내 내는 데모용 plan."""
    return {
        "methods": ["otsu", "adaptive", "manual"],
        "manual_thresholds": MANUAL_THRESHOLDS,
        "preprocessing": ["median_blur", "clahe"],
        "metrics": ["area_px", "area_ratio", "r2_if_gt_exists"],
        "outputs": ["mask_images", "csv", "area_graph", "report"],
    }
