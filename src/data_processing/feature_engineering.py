"""
Data engineering module for processing and feature extraction from the imputed suicide study dataset.

This module includes assigning groups based on age, gender and fatal, and performing one-hot encoding on categorical variables.
"""

from pathlib import Path
import sys
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Convert DATA_DIR to a Path object
DATA_DIR = Path(os.getenv("DATA_DIR"))

# %%
# Read CSV File
csv_file_path = DATA_DIR / "imputed" / "imputed_data.csv"
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
group_mapping = {
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

df_imputed["AgeGroup2"] = df_imputed["AgeGroup"].map(group_mapping)

# Create CountContext column
df_imputed["CountContext"] = df_imputed.filter(like="Context").sum(axis=1)


# Assign New Groups
# Function to assign groups
def assign_group(df, group_column, mapping_columns, group_mapping):
    """
    Assign groups based on specified columns and mapping.

    Parameters:
    - df: DataFrame containing the data
    - group_column: Name of the column to store the group assignment
    - mapping_columns: List of column names to use for mapping
    - group_mapping: List of tuples with mapping criteria

    Returns:
    - DataFrame with the new group assignments
    """
    # Initialize the group column with NaN and set the correct dtype
    df[group_column] = pd.Series(dtype="object")  # Ensure the column is of object type

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        for values in group_mapping:
            # Check if the current row matches the group mapping
            if len(values) == len(mapping_columns) and all(
                row[mapping_columns[i]] == values[i]
                for i in range(len(mapping_columns))
            ):
                # Assign the group if there's a match
                df.at[index, group_column] = f"{'_'.join(map(str, values))}"
                break  # Exit the loop once a match is found

    return df


# Assign group based on AgeGroup and Gender
# Group_AG
mapping_columns = ["AgeGroup2", "Gender"]
group_mapping = [
    ("00_18", "F"),
    ("00_18", "M"),
    ("19_34", "F"),
    ("19_34", "M"),
    ("35_64", "F"),
    ("35_64", "M"),
    ("65", "F"),
    ("65", "M"),
]

df_imputed = assign_group(df_imputed, "Group_AG", mapping_columns, group_mapping)

# Assign group based on AgeGroup and Fatality
# Group_AF
mapping_columns = ["AgeGroup2", "Fatal"]
group_mapping = [
    ("00_18", 0),
    ("00_18", 1),
    ("19_34", 0),
    ("19_34", 1),
    ("35_64", 0),
    ("35_64", 1),
    ("65", 0),
    ("65", 1),
]

df_imputed = assign_group(df_imputed, "Group_AF", mapping_columns, group_mapping)


# Group_AGF
# Define mappings for age, gender, and fatality groups
mapping_columns = ["AgeGroup2", "Gender", "Fatal"]
group_mapping = [
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

df_imputed = assign_group(df_imputed, "Group_AGF", mapping_columns, group_mapping)


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
