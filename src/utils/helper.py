from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.config import config


def save_figure(figure_name: str, subfolder: str) -> Path:
    """Save the current matplotlib figure under reports/figures."""
    figure_dir = Path(config["paths"]["figures"]) / subfolder
    figure_dir.mkdir(parents=True, exist_ok=True)

    path = figure_dir / f"{figure_name}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    return path
