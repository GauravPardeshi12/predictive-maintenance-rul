from src.utils.config import config


NUM_SENSORS = config["dataset"]["num_sensors"]


def get_sensor_columns() -> list[str]:
    """Return the configured sensor column names."""
    return [f"sensor_{i}" for i in range(1, NUM_SENSORS + 1)]
