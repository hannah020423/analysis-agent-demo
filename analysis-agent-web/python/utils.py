from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "masks": output_dir / "masks",
        "csv": output_dir / "csv",
        "figures": output_dir / "figures",
        "reports": output_dir / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs