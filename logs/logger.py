"""Logger configuration for the project."""

import logging.config
from pathlib import Path
import yaml

# === Absolute path to logs directory ===
LOG_DIR = Path(__file__).parent  # Absolute path to logs/
LOG_FILE = LOG_DIR / "app.log"   # Path to the log file
LOGGER_SETTINGS_PATH = LOG_DIR / "logger.yaml"  # Path to logger.yaml

# === Check if logger.yaml exists ===
if not LOGGER_SETTINGS_PATH.exists():
    raise FileNotFoundError(f"Logger configuration file not found: {LOGGER_SETTINGS_PATH}")

# === Load logger configuration from logger.yaml ===
def setup_logging(default_level=logging.INFO):
    """
    Setup logging configuration from logger.yaml file.
    If logger.yaml does not exist, it defaults to basic logging.
    """
    with open(LOGGER_CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)
        print(f"Loaded logging configuration from: {LOGGER_CONFIG_PATH}")

# === Setup the logger ===
setup_logging()

# === Get logger instance ===
logger = logging.getLogger("customLogger")

# === Helper function for logging messages ===
def log_message(level, message, **kwargs):
    """
    Logs a message with additional context.

    Args:
        level (int): Logging level (e.g., logging.INFO, logging.ERROR).
        message (str): Log message.
        **kwargs: Additional context key-value pairs.
    """
    context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.log(level, f"{message} | {context}" if context else message)
