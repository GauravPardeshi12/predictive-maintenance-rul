from pathlib import Path
import matplotlib.pyplot as plt

from src.utils.logger import logger
from src.utils.config import config

def save_figure(figure_name: str, subfolder: str) -> Path:
    """
    Save the current matplotlib figure.
    """

    figure_dir = Path(config["paths"]["figures"]) / subfolder

    figure_dir.mkdir(parents= True, exist_ok = True)

    figure_path = figure_dir / f"{figure_name}.png"

    plt.savefig(figure_path, dpi= 300, bbox_inches= "tight")

    logger.info(f"Figure saved to {figure_path}")

    return figure_path


