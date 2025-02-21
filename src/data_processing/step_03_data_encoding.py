"""
Data encoding module for encoding features from the imputed suicide study dataset.

This module includes:
- Performing one-hot encoding on categorical variables.
"""

from typing import List, Optional

import pandas as pd

from config.config import DATA_DIR
from settings.settings import MOMENT_OF_SUICIDE_FEATURES, SOCIO_DEMOGRAPHIC_FEATURES
from src.utils.utils import read_csv, write_csv


def perform_one_hot_encoding(
    df: pd.DataFrame, columns_to_encode: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Perform one-hot encoding on specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_encode (Optional[List[str]]): List of columns to encode. If None, all columns are processed.

    Returns:
        pd.DataFrame: DataFrame with binary mapping and one-hot encoded columns, combined with untouched columns.
    """
    df = df.copy()

    if columns_to_encode is None:
        columns_to_encode = df.columns.tolist()

    encoded_parts = []
    for column in columns_to_encode:
        if df[column].isna().all():
            # Warn about columns with only NaNs
            print(f"Warning: Column '{column}' contains only NaNs and will be skipped.")
            continue

        unique_values = df[column].dropna().unique()
        if len(unique_values) <= 2:
            # Alphabetically map binary columns to 0 and 1
            mapping = {key: idx for idx, key in enumerate(sorted(unique_values))}
            encoded_column = (
                df[column]
                .map(mapping)
                .astype("Int64")  # Nullable integer type for NaNs
            )
            encoded_parts.append(encoded_column.rename(column))
        else:
            # Perform one-hot encoding for columns with more than two unique values
            one_hot = pd.get_dummies(df[column], prefix=column, drop_first=False)
            one_hot = one_hot.astype(int)  # Convert to integer type
            encoded_parts.append(one_hot)

    # Combine encoded columns
    encoded_df = pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame()

    # Merge with columns not encoded
    untouched_columns = df.drop(columns=columns_to_encode)
    final_df = pd.concat([untouched_columns, encoded_df], axis=1)

    return final_df


def run_data_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data encoding on the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame with engineered features.
    """
    # Determine columns to encode based on prefixes
    feature_prefixes = set(SOCIO_DEMOGRAPHIC_FEATURES + MOMENT_OF_SUICIDE_FEATURES)
    columns_to_encode = list(
        filter(
            lambda col: any(col.startswith(prefix) for prefix in feature_prefixes),
            df.columns,
        )
    )

    # Perform one-hot encoding on specified features
    encoded_df = perform_one_hot_encoding(df, columns_to_encode=columns_to_encode)
    return encoded_df


if __name__ == "__main__":
    # Load raw data
    csv_file_path = DATA_DIR / "processed" / "imputed_data.csv"
    df_raw = read_csv(csv_file_path, delimiter=",", low_memory=False)

    df_encoded = run_data_encoding(df_raw)

    # Save the encoded data
    file_name = "encoded_data.csv"
    output_file_path = DATA_DIR / "processed"
    write_csv(data=df_encoded, file_path=output_file_path / file_name, index=False)
