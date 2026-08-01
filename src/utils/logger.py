"""
Central logging configuration for the project.
"""

import logging
from pathlib import Path

def setup_logger(log_file: str = "project.log") -> logging.Logger:
    """"
    Configure and return a project logger

    Parameters
    -----------------
    log_file: str
    Name of the log file

    Return
    -----------------
    logging.Logger
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("PredictiveMaintenance")

    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_dir / log_file)

    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
