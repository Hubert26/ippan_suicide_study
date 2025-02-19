# config.py
"""Project configuration settings."""

import yaml
from pathlib import Path

# Load settings from `settings.yaml`
with open("settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

# Assign paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / settings["paths"]["data"]
RESULTS_DIR = PROJECT_ROOT / settings["paths"]["results"]
PLOTS_DIR = PROJECT_ROOT / settings["paths"]["plots"]
MOMENT_OF_SUICIDE_FEATURES = settings["moment_of_suicide_features"]
SOCIO_DEMOGRAPHIC_FEATURES = settings["socio_demographic_features"]

# Ensure directories exist before proceeding
for directory in [DATA_DIR, RESULTS_DIR, PLOTS_DIR]:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(PROJECT_ROOT)
    print(DATA_DIR)
    print(RESULTS_DIR)
    print(PLOTS_DIR)
    print(MOMENT_OF_SUICIDE_FEATURES)
    print(SOCIO_DEMOGRAPHIC_FEATURES)