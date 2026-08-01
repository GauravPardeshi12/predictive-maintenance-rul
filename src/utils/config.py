from pathlib import Path
import yaml
from src.utils.logger import logger


def load_config() -> dict:
    """
    Load project configuration from config.yaml.

    Returns
    --------
    dict
        Project configuration.
    """
    config_path = Path("configs/config.yaml")

    logger.info("Loading project configuration.")

    if not config_path.exists():
        logger.error("Configuration file not found.")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r" , encoding= "utf-8") as file:
        config = yaml.safe_load(file)

    return config

config = load_config()

    