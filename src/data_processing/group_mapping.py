"""
Group Mapping Module

This module assigns groups to rows in a dataset based on specific criteria such as age, gender, and fatality. It is
primarily used for feature engineering in datasets related to suicide studies.

Functionalities:
- Map age ranges to broader categories.
- Create grouped categories based on combinations of age, gender, and fatality status.
- Ensure outputs are formatted and ready for analysis.

Main Steps:
1. Map age ranges to broader groups (e.g., '19_24' -> '19_34').
2. Generate new group columns for:
   - Age and gender (`Group_AG`).
   - Age and fatality (`Group_AF`).
   - Age, gender, and fatality (`Group_AGF`).
"""

import sys
from pathlib import Path
from dotenv import dotenv_values
import pandas as pd
from typing import List, Tuple

# Load environment variables from the .env file
env_vars = dotenv_values()  # Load variables from the .env file

# Get the workspace path from the environment variables
WORKSPACE_PATH = Path(env_vars.get("WORKSPACE_PATH"))  # Fetch WORKSPACE_PATH from .env

if not WORKSPACE_PATH:
    raise ValueError("WORKSPACE_PATH is not defined in the .env file or is empty.")

# Add the WORKSPACE_PATH folder to the Python path
sys.path.append(str(WORKSPACE_PATH))

# Import custom utility functions
from src.config.utils import read_csv, write_csv

DATA_DIR = Path(env_vars["DATA_DIR"])

# ==============================================================================
# Constants and Configuration
# ==============================================================================
REQUIRED_COLUMNS = ["ID", "AgeGroup", "Gender", "Fatal"]

AGE_MAPPING = {
    "07_12": "00_18",
    "13_18": "00_18",
    "19_24": "19_34",
    "25_29": "19_34",
    "30_34": "19_34",
    "35_39": "35_64",
    "40_44": "35_64",
    "45_49": "35_64",
    "50_54": "35_64",
    "55_59": "35_64",
    "60_64": "35_64",
    "65_69": "65",
    "70_74": "65",
    "75_79": "65",
    "80_84": "65",
    "85": "65",
}

AGE_GENDER_MAPPING = [
    ("00_18", "F"),
    ("00_18", "M"),
    ("19_34", "F"),
    ("19_34", "M"),
    ("35_64", "F"),
    ("35_64", "M"),
    ("65", "F"),
    ("65", "M"),
]


AGE_FATALITY_MAPPING = [
    ("00_18", 0),
    ("00_18", 1),
    ("19_34", 0),
    ("19_34", 1),
    ("35_64", 0),
    ("35_64", 1),
    ("65", 0),
    ("65", 1),
]

AGE_GENDER_FATALITY_MAPPING = [
    ("00_18", "F", 0),
    ("00_18", "F", 1),
    ("00_18", "M", 0),
    ("00_18", "M", 1),
    ("19_34", "F", 0),
    ("19_34", "F", 1),
    ("19_34", "M", 0),
    ("19_34", "M", 1),
    ("35_64", "F", 0),
    ("35_64", "F", 1),
    ("35_64", "M", 0),
    ("35_64", "M", 1),
    ("65", "F", 0),
    ("65", "F", 1),
    ("65", "M", 0),
    ("65", "M", 1),
]


# %%
# ==============================================================================
# Helper Functions
# ==============================================================================
def map_to_groups(
    df: pd.DataFrame,
    group_column: str,
    mapping_columns: List[str],
    group_mapping: List[Tuple],
) -> pd.DataFrame:
    """
    Assign groups to rows in a DataFrame based on specified criteria.

    This function iterates through the rows of a DataFrame and assigns a group value
    based on the values of certain columns and predefined group mappings.

    Args:
        df (pd.DataFrame): Input DataFrame containing data to be grouped.
        group_column (str): Name of the column to store the group assignments.
        mapping_columns (List[str]): List of column names to use as criteria for group assignment.
        group_mapping (List[Tuple]): Predefined group mappings (as tuples).

    Returns:
        pd.DataFrame: DataFrame with a new column containing group assignments.

    Example:
        mapping_columns = ["AgeGroup", "Gender"]
        group_mapping = [("00_18", "F"), ("19_34", "M")]
        -> Adds a new column `group_column` with values "00_18_F", "19_34_M", etc.
    """
    df[group_column] = pd.Series(dtype="object")  # Ensure the column exists
    for index, row in df.iterrows():
        for values in group_mapping:
            if len(values) == len(mapping_columns) and all(
                row[mapping_columns[i]] == values[i]
                for i in range(len(mapping_columns))
            ):
                df.at[index, group_column] = f"{'_'.join(map(str, values))}"
                break
    return df


# %%
# ==============================================================================
# Main Processing Logic
# ==============================================================================
def run_group_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform group mapping on a dataset to create new categorical group columns.

    Groups are created based on:
    - Age group (`Group_A`).
    - Age and gender (`Group_AG`).
    - Age and fatality (`Group_AF`).
    - Age, gender, and fatality (`Group_AGF`).

    Args:
        df (pd.DataFrame): Input DataFrame containing the necessary columns:
            - `AgeGroup`: Age categories.
            - `Gender`: Gender values.
            - `Fatal`: Binary fatality status (0 or 1).

    Returns:
        pd.DataFrame: Transformed DataFrame with additional group columns.
    """
    df["Group_A"] = df["AgeGroup"].map(AGE_MAPPING)
    df = map_to_groups(df, "Group_AG", ["Group_A", "Gender"], AGE_GENDER_MAPPING)
    df = map_to_groups(df, "Group_AF", ["Group_A", "Fatal"], AGE_FATALITY_MAPPING)
    df = map_to_groups(
        df, "Group_AGF", ["Group_A", "Gender", "Fatal"], AGE_GENDER_FATALITY_MAPPING
    )
    return df


if __name__ == "__main__":
    # Load raw data
    csv_file_path = DATA_DIR / "processed" / "imputed_data.csv"
    df_raw = read_csv(csv_file_path, delimiter=",", low_memory=False)

    # Filter to required columns
    df = df_raw[REQUIRED_COLUMNS].copy()

    # Run group mapping
    df = run_group_mapping(df)

    # Save the group data
    csv_file_path = DATA_DIR / "processed" / "group_set.csv"
    write_csv(data=df, file_path=csv_file_path, index=False)