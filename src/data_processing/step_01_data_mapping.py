"""Data mapping module for standardizing and transforming suicide study dataset.
Handles data from both 2023 and 2013-2022 periods."""

from typing import Dict, List, Optional

import pandas as pd

from config.config import DATA_DIR
from settings.settings import (
    COLUMN_MAPPINGS_2013_2022,
    COLUMN_MAPPINGS_2023,
    MOMENT_OF_SUICIDE_FEATURES,
    SOCIO_DEMOGRAPHIC_FEATURES,
    VALUE_MAPPINGS_2013_2022,
    VALUE_MAPPINGS_2023,
)
from src.utils.utils import read_csv, read_excel, write_csv

PREXES_TO_RETAIN = (
    MOMENT_OF_SUICIDE_FEATURES + SOCIO_DEMOGRAPHIC_FEATURES + ["ID", "Date", "AgeGroup"]
)


def map_columns(df: pd.DataFrame, column_mappings: Dict[str, str]) -> pd.DataFrame:
    """Rename columns in the DataFrame based on a mapping dictionary."""
    df.rename(columns=column_mappings, inplace=True)
    return df


def map_features(
    df: pd.DataFrame, column_mappings: Dict[str, str], value_mappings: Dict[str, Dict]
) -> pd.DataFrame:
    """Apply value mappings to DataFrame columns."""
    for old_col, new_col in column_mappings.items():
        if new_col in value_mappings and new_col in df.columns:
            # Convert column to object to handle mixed types
            df[new_col] = df[new_col].astype(object)

            # Apply mapping
            mapped = df[new_col].map(value_mappings[new_col])

            # Replace invalid values with pd.NA to ensure compatibility
            mapped = mapped.where(mapped.notna(), pd.NA)

            # Convert to object type
            df.loc[:, new_col] = mapped.astype(object)
    return df


def merge_columns(
    df: pd.DataFrame, columns: List[str], output_column: str
) -> pd.DataFrame:
    """
    Merge multiple columns into a single column, prioritizing non-null values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of column names to merge.
        output_column (str): Name of the resulting merged column.

    Returns:
        pd.DataFrame: Updated DataFrame with the merged column.
    """
    if not columns:
        return df

    df = df.copy()

    # Merge columns and drop originals
    merged_column = df[columns].bfill(axis=1).iloc[:, 0]
    df.loc[:, output_column] = merged_column
    df.drop(columns=columns, inplace=True, errors="ignore")
    return df


def encode_columns_by_prefix(
    df: pd.DataFrame,
    column_prefix: str,
    output_prefix: str,
    value_mapping: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Encode unique values from columns with a specific prefix into binary columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_prefix (str): Prefix of the columns to encode.
        output_prefix (str): Prefix for the resulting binary columns.
        value_mapping (Optional[Dict]): Optional mapping for values before encoding.

    Returns:
        pd.DataFrame: Updated DataFrame with binary-encoded columns.
    """
    target_columns = [col for col in df.columns if col.startswith(column_prefix)]
    if not target_columns:
        return df  # No columns with the specified prefix

    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Apply value mapping if provided
    if value_mapping:
        for col in target_columns:
            if col in df.columns:
                mapped_column = df[col].map(value_mapping)
                # Replace NaN with pd.NA for compatibility with object type
                mapped_column = mapped_column.astype(object).where(
                    mapped_column.notna(), pd.NA
                )
                df.loc[:, col] = mapped_column.astype(object)

    # Get unique non-null values across all target columns
    unique_values = pd.concat(
        [df[col].dropna() for col in target_columns], axis=0
    ).unique()

    for value in unique_values:
        binary_column = f"{output_prefix}_{value}"
        # Use .loc to modify the DataFrame, handling NA values explicitly
        df.loc[:, binary_column] = df[target_columns].apply(
            lambda row: int(value in row.dropna().values), axis=1
        )

    # Drop the original columns
    df.drop(columns=target_columns, inplace=True, errors="ignore")
    return df


def clean_data(
    df: pd.DataFrame,
    value_mappings: Dict[str, Dict],
    prefixes_to_retain: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Clean and preprocess data, including filtering columns based on mappings.

    Args:
        df (pd.DataFrame): Input DataFrame.
        value_mappings (Dict[str, Dict]): Mapping dictionary for values.
        column_mappings (Optional[Dict[str, str]]): Mapping dictionary for column names.

    Returns:
        pd.DataFrame: Cleaned and filtered DataFrame.
    """
    df = df.copy()

    # Drop rows with empty or NaN IDs
    if "ID" in df.columns:
        df = df.loc[~df["ID"].isna() & (df["ID"].str.strip() != "")]

    # Convert Date to datetime and extract Year/Month
    if "Date" in df.columns:
        # First, try to convert 'YYYYMM' format to datetime
        date_column_YYYYMM = pd.to_datetime(df["Date"], format="%Y%m", errors="coerce")
        # Then, convert any remaining valid date strings to datetime
        date_column_rest = pd.to_datetime(df["Date"], errors="coerce")
        df.loc[:, "Date"] = date_column_YYYYMM.fillna(date_column_rest)
        df["Date"] = df["Date"].astype("datetime64[ns]")

        if df["Date"].isna().all():
            raise ValueError(
                "All values in 'Date' column are invalid and could not be parsed."
            )

        # Create year and month columns if Date is valid
        # Ensure the Date column is in datetime format
        if pd.api.types.is_datetime64_any_dtype(df["Date"]):
            # Create year and month columns
            df["DateY"] = df["Date"].dt.year.astype("Int64").astype(str).str.zfill(4)
            df["DateM"] = df["Date"].dt.month.astype("Int64").astype(str).str.zfill(2)
        else:
            raise ValueError("Column 'Date' could not be converted to datetime format.")

    # Encode Context columns into binary features
    df = encode_columns_by_prefix(
        df,
        column_prefix="Context",
        output_prefix="Context",
        value_mapping=value_mappings.get("Context"),
    )

    # Merge AbuseInfo columns into a single column
    df = merge_columns(
        df,
        columns=[col for col in df.columns if col.startswith("AbuseInfo")],
        output_column="AbuseInfo",
    )

    # Filter columns to retain only those with prefixes in prefixes_to_retain
    if prefixes_to_retain:
        columns_to_retain = {
            col: col
            for col in df.columns
            if any(col.startswith(prefix) for prefix in prefixes_to_retain)
        }
        mapped_columns = set(columns_to_retain.values())
        current_columns = set(df.columns)
        columns_to_retain = current_columns & mapped_columns  # Keep only mapped columns
        unmapped_columns = current_columns - columns_to_retain  # Find unmapped columns

        # Drop unmapped columns
        df.drop(columns=list(unmapped_columns), inplace=True, errors="ignore")

    return df


def run_data_mapping(
    df_raw_2023: pd.DataFrame,
    df_raw_2013_2022: pd.DataFrame,
    column_mappings_2023: Dict[str, str],
    value_mappings_2023: Dict[str, Dict],
    column_mappings_2013_2022: Dict[str, str],
    value_mappings_2013_2022: Dict[str, Dict],
) -> pd.DataFrame:
    """Run data mapping for 2023 and 2013-2022 datasets and combine them."""
    df_2023 = map_columns(df_raw_2023, column_mappings_2023)
    df_2023 = map_features(df_2023, column_mappings_2023, value_mappings_2023)
    df_2023 = clean_data(df_2023, value_mappings_2023, PREXES_TO_RETAIN)

    df_2013_2022 = map_columns(df_raw_2013_2022, column_mappings_2013_2022)
    df_2013_2022 = map_features(
        df_2013_2022, column_mappings_2013_2022, value_mappings_2013_2022
    )
    df_2013_2022 = clean_data(df_2013_2022, value_mappings_2013_2022, PREXES_TO_RETAIN)

    df_combined = pd.concat([df_2023, df_2013_2022], ignore_index=True)

    df_combined.drop(columns=["ID"], inplace=True)  # Drop the original "ID" column
    df_combined.reset_index(drop=False, inplace=True)  # Reset the index
    df_combined.rename(
        columns={"index": "ID"}, inplace=True
    )  # Rename the new index column to "ID"
    return df_combined


# ================================================================================
# PROCESS DATASETS
# ================================================================================
if __name__ == "__main__":
    excel_file_path = DATA_DIR / "raw" / "Samobojstwa_2023.xlsx"
    df_raw_2023 = read_excel(excel_file_path)

    csv_file_path = DATA_DIR / "raw" / "final_samobojstwa_2013_2022.csv"
    df_raw_2013_2022 = read_csv(csv_file_path, delimiter=",", low_memory=False)

    df_mapped = run_data_mapping(
        df_raw_2023,
        df_raw_2013_2022,
        COLUMN_MAPPINGS_2023,
        VALUE_MAPPINGS_2023,
        COLUMN_MAPPINGS_2013_2022,
        VALUE_MAPPINGS_2013_2022,
    )

    # Save combined dataset
    csv_file_path = DATA_DIR / "processed"
    write_csv(
        data=df_mapped,
        file_path=csv_file_path / "mapped_data.csv",
        index=False,
    )
