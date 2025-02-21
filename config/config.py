# config.py
"""Project configuration settings."""

import sys
from dotenv import dotenv_values
from pathlib import Path
import yaml

# Load environment variables from the .env file
env_vars = dotenv_values() 

# Get the PROJECT_ROOT path
PROJECT_ROOT = Path(env_vars.get("PROJECT_ROOT", "/"))  # Default to "/" for Docker

if not PROJECT_ROOT.exists():
    raise ValueError(f"PROJECT_ROOT is not a valid path: {PROJECT_ROOT}")

# Add PROJECT_ROOT to sys.path if not already present
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

CONFIG_DIR = Path(__file__).parent  # Absolute path to config/
CONFIG_SETTINGS_PATH = CONFIG_DIR / "config.yaml"  # Path to config.yaml

def load_yaml(file_path: Path):
    """Helper function to load a YAML configuration file."""
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
    
# Loaging paths
CONFIGS = load_yaml(CONFIG_SETTINGS_PATH)

DATA_DIR = PROJECT_ROOT / CONFIGS["paths"]["data"]
RESULTS_DIR = PROJECT_ROOT / CONFIGS["paths"]["results"]
PLOTS_DIR = PROJECT_ROOT / CONFIGS["paths"]["plots"]

# Ensure directories exist before proceeding
for directory in [DATA_DIR, RESULTS_DIR, PLOTS_DIR]:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    
# Print configuration (useful for debugging)
def print_paths(): 
    print("PROJECT_ROOT:", PROJECT_ROOT, "\n")
    print("DATA_DIR:", DATA_DIR, "\n")
    print("RESULTS_DIR:", RESULTS_DIR, "\n")
    print("PLOTS_DIR:", PLOTS_DIR, "\n")
    
if __name__ == "__main__":
    print_paths()