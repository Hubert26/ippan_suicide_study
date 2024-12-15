"""
Data engineering module for processing and feature extraction from the imputed suicide study dataset.

This module includes assigning groups based on age, gender and fatal, and performing one-hot encoding on categorical variables.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

DATA_DIR = os.getenv('DATA_DIR')

# %%
# Read CSV File
csv_file_path = Path(DATA_DIR) / "imputed" / "imputed_data.csv"
try:
    df_imputed = pd.read_csv(csv_file_path, delimiter=",", low_memory=False)
except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found.")
    sys.exit(1)

# %%
# ================================================================================
# Data Preparation
# ================================================================================

# Assign AgeGroup2 from AgeGroup
age_group_mapping = {
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

df_imputed["AgeGroup2"] = df_imputed["AgeGroup"].map(age_group_mapping)

# Create CountContext column
df_imputed["CountContext"] = df_imputed.filter(like="Context").sum(axis=1)

# Assign New Groups

# Assign groups based on age and gender
# Group_AG
age_gender_mapping = [
    ("00_18", 0),
    ("00_18", 1),
    ("19_34", 0),
    ("19_34", 1),
    ("35_64", 0),
    ("35_64", 1),
    ("65", 0),
    ("65", 1),
]

# Assign group based on AgeGroup and Gender


def assign_group(df, age_column, gender_column, group_column, group_mapping):
    conditions = [
        (df[age_column] == age) & (df[gender_column] == gender)
        for age, gender in group_mapping
    ]
    choices = [f"{age}_{gender}" for age, gender in group_mapping]
    df[group_column] = np.select(conditions, choices, default=np.nan)
    return df


df_imputed = assign_group(
    df_imputed, "AgeGroup2", "Gender", "Group_AG", age_gender_mapping
)


# Group_AF
# Define mappings for age and fatality groups
age_fatal_mapping = [
    ("00_18", 0),
    ("00_18", 1),
    ("19_34", 0),
    ("19_34", 1),
    ("35_64", 0),
    ("35_64", 1),
    ("65", 0),
    ("65", 1),
]

# Assign group based on AgeGroup and Fatality


def assign_group_af(df, age_column, fatal_column, group_column, group_mapping):
    conditions = [
        (df[age_column] == age) & (df[fatal_column] == fatal)
        for age, fatal in group_mapping
    ]
    choices = [f"{age}_{fatal}" for age, fatal in group_mapping]
    df[group_column] = np.select(conditions, choices, default=np.nan)
    return df


df_imputed = assign_group_af(
    df_imputed, "AgeGroup2", "Fatal", "Group_AF", age_fatal_mapping
)


# Group_AGF
# Define mappings for age, gender, and fatality groups
age_gender_fatal_mapping = [
    ("00_18", 0, 0),
    ("00_18", 0, 1),
    ("00_18", 1, 0),
    ("00_18", 1, 1),
    ("19_34", 0, 0),
    ("19_34", 0, 1),
    ("19_34", 1, 0),
    ("19_34", 1, 1),
    ("35_64", 0, 0),
    ("35_64", 0, 1),
    ("35_64", 1, 0),
    ("35_64", 1, 1),
    ("65", 0, 0),
    ("65", 0, 1),
    ("65", 1, 0),
    ("65", 1, 1),
]

# Assign group based on AgeGroup, Gender, and Fatality


def assign_group_agf(
    df, age_column, gender_column, fatal_column, group_column, group_mapping
):
    conditions = [
        (df[age_column] == age)
        & (df[gender_column] == gender)
        & (df[fatal_column] == fatal)
        for age, gender, fatal in group_mapping
    ]
    choices = [f"{age}{gender}{fatal}" for age, gender, fatal in group_mapping]
    df[group_column] = np.select(conditions, choices, default=np.nan)
    return df


df_imputed = assign_group_agf(
    df_imputed, "AgeGroup2", "Gender", "Fatal", "Group_AGF", age_gender_fatal_mapping
)


# Saving Processed Data
# Save the final feature set to CSV
file_name = "final_feature_set.csv"
output_file_path = DATA_DIR / "prepped"
try:
    df_imputed.to_csv(output_file_path / file_name, index=False)
except Exception as e:
    print(f"Error: Failed to write to file {output_file_path / file_name}. {str(e)}")
    sys.exit(1)

# %%
# ==============================================================================
# One-Hot Encoding
# ==============================================================================

# One-Hot Encoding
# Define columns for one-hot encoding
columns_to_encode = [
    "AbuseInfo",
    "Income",
    "Method",
    "Education",
    "WorkInfo",
    "Substance",
    "Place",
    "Marital",
]

# Include boolean columns in the encoding process
bool_columns = [
    "Fatal",
    "Gender",
    "Context_Other",
    "Context_FamilyConflict",
    "Context_HeartBreak",
    "Context_Finances",
    "Context_SchoolWork",
    "Context_CloseDeath",
    "Context_Crime",
    "Context_Disability",
    "Context_MentalHealth",
    "Context_HealthLoss",
]

# Convert Gender column to categorical type
df_imputed["Gender"] = df_imputed["Gender"].astype("category")

# Map Gender values to integers using rename_categories
df_imputed["Gender"] = df_imputed["Gender"].cat.rename_categories({"M": 1, "F": 0})

# Convert boolean columns to booleans
df_imputed[bool_columns] = df_imputed[bool_columns].astype(bool)

# Combine the columns to encode
columns_to_encode.extend(bool_columns)

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df_imputed[columns_to_encode], drop_first=False)

# Merge additional columns
# Define columns to merge after encoding
columns_to_merge = df_imputed.columns.difference(
    columns_to_encode + bool_columns
).tolist()
df_encoded[columns_to_merge] = df_imputed[columns_to_merge].copy()

# Save the encoded data
file_name = "encoded_final_set.csv"
output_file_path = DATA_DIR / "encoded"
try:
    df_encoded.to_csv(output_file_path / file_name, index=False)
except Exception as e:
    print(f"Error: Failed to write to file {output_file_path / file_name}. {str(e)}")
    sys.exit(1)
