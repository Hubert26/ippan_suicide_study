# config.py
"""Project configuration settings."""

import yaml
import sys
from dotenv import dotenv_values
from pathlib import Path


# Load environment variables from the .env file
env_vars = dotenv_values() 

# Get the PROJECT_ROOT path
PROJECT_ROOT = Path(env_vars.get("PROJECT_ROOT", "/"))  # Default to "/" for Docker

if not PROJECT_ROOT.exists():
    raise ValueError(f"PROJECT_ROOT is not a valid path: {PROJECT_ROOT}")

# Add PROJECT_ROOT to sys.path if not already present
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load settings from `settings.yaml`
with open("settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

# Assign paths
DATA_DIR = PROJECT_ROOT / settings["paths"]["data"]
RESULTS_DIR = PROJECT_ROOT / settings["paths"]["results"]
PLOTS_DIR = PROJECT_ROOT / settings["paths"]["plots"]
MOMENT_OF_SUICIDE_FEATURES = settings["moment_of_suicide_features"]
SOCIO_DEMOGRAPHIC_FEATURES = settings["socio_demographic_features"]

# Ensure directories exist before proceeding
for directory in [DATA_DIR, RESULTS_DIR, PLOTS_DIR]:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

# Print configuration (useful for debugging)
def print_config(): 
    print("PROJECT_ROOT:", PROJECT_ROOT, "\n")
    print("DATA_DIR:", DATA_DIR, "\n")
    print("RESULTS_DIR:", RESULTS_DIR, "\n")
    print("PLOTS_DIR:", PLOTS_DIR, "\n")
    print("MOMENT_OF_SUICIDE_FEATURES:", MOMENT_OF_SUICIDE_FEATURES, "\n")
    print("SOCIO_DEMOGRAPHIC_FEATURES:", SOCIO_DEMOGRAPHIC_FEATURES, "\n")


if __name__ == "__main__":
    print_config()