from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

WORKSPACE_PATH = Path(os.getenv("WORKSPACE_PATH"))
DATA_DIR = Path(os.getenv("DATA_DIR"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR"))
PLOTS_DIR = Path(os.getenv("PLOTS_DIR"))
TABLES_DIR = Path(os.getenv("TABLES_DIR"))


def create_directories():
    """
    Create necessary directories if they do not exist.
    """
    # List of directories to create
    directories = [WORKSPACE_PATH, DATA_DIR, RESULTS_DIR, PLOTS_DIR, TABLES_DIR]

    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")


# Call the function to create directories
create_directories()
