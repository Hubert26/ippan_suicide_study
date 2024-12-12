# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:19:39 2024

@author: huber
"""

# %%
from pathlib import Path


# Define directories
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
# %%
