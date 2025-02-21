"""
Data imputation module for handling missing values in suicide study dataset.
"""

import numpy as np
import pandas as pd

from src.helpers.config import DATA_DIR
from src.helpers.utils import read_csv, write_csv


def get_neighboring_age_groups(age_groups, current_age_group):
    """
    Gets adjacent age groups for a given age group.

    Args:
        age_groups (list): Sorted list of age groups.
        current_age_group (str/int): Target age group.

    Returns:
        tuple: Previous and next age groups (None if not applicable).
    """
    earlier_age_group = (
        age_groups[age_groups.index(current_age_group) - 1]
        if current_age_group != age_groups[0]
        else None
    )
    later_age_group = (
        age_groups[age_groups.index(current_age_group) + 1]
        if current_age_group != age_groups[-1]
        else None
    )
    return earlier_age_group, later_age_group


def filter_columns_by_missing_data(
    dataframe, accept_probability=(0, 100), columns=None
):
    """
    Filters columns based on missing data percentage thresholds.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        accept_probability (tuple): Range of acceptable missing data percentages.
        columns (list, optional): Specific columns to check.

    Returns:
        list: Columns within acceptable missing data range.
    """
    if columns is None:
        columns = dataframe.columns

    missing_percent = dataframe[columns].isnull().mean() * 100
    return missing_percent[
        (missing_percent < accept_probability[1])
        & (missing_percent > accept_probability[0])
    ].index.tolist()


def fill_missing_values_by_probability(dataframe, column_name):
    """
    Fills missing values using probability distribution of existing values.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        column_name (str): Target column for imputation.

    Returns:
        pd.DataFrame: DataFrame with imputed values.
    """
    dataframe = dataframe.copy()
    null_index = dataframe[dataframe[column_name].isnull()].index
    value_counts_result = dataframe[column_name].value_counts()

    if not value_counts_result.empty:
        value_list = value_counts_result.index.tolist()
        probabilities = [
            count / value_counts_result.sum() for count in value_counts_result.values
        ]
        dataframe.loc[null_index, column_name] = np.random.choice(
            value_list, size=len(null_index), p=probabilities
        )

    return dataframe


def run_data_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data imputation on the given DataFrame.

    This includes:
    - Handling missing critical data by dropping rows with missing critical fields.
    - Filling missing values in categorical columns like 'AbuseInfo'.
    - Imputing missing values for numerical and categorical columns using group-specific probabilities.
    - Handling contextual information imputation.

    Args:
        df (pd.DataFrame): Input DataFrame with missing values.

    Returns:
        pd.DataFrame: Imputed DataFrame with missing values handled.
    """
    # Split data and context
    context_columns = [col for col in df.columns if col.startswith("Context_")]

    # Check if required columns are present
    required_columns = {"AgeGroup", "Gender", "ID"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns in the DataFrame: {missing_columns}")

    df_context = df[context_columns + ["AgeGroup", "Gender", "ID"]]
    df_data = df.drop(columns=context_columns, inplace=False)

    # Drop rows with missing critical data
    critical_columns = ["Gender", "AgeGroup", "Date"]
    selected_index = df_data[df_data[critical_columns].isna().any(axis=1)].index
    df_data = df_data.drop(index=selected_index)
    df_context = df_context.drop(index=selected_index)

    # Fill missing values in 'AbuseInfo'
    df_data["AbuseInfo"] = df_data["AbuseInfo"].fillna("Not")

    # Fill missing values in age and gender groups
    age_groups = sorted(df_data["AgeGroup"].dropna().unique())
    gender_groups = sorted(df_data["Gender"].dropna().unique())

    accept_probability = (0, 75)
    imputed_df = df_data.copy()

    for age_group in age_groups:
        for gender_group in gender_groups:
            filtered_data = df_data[
                (df_data["AgeGroup"] == age_group) & (df_data["Gender"] == gender_group)
            ]
            selected_columns = filter_columns_by_missing_data(
                filtered_data, accept_probability=accept_probability, columns=None
            )

            for column_name in filtered_data.columns:
                if column_name in selected_columns:
                    imputed_values = fill_missing_values_by_probability(
                        filtered_data, column_name
                    )
                    imputed_df.loc[imputed_values.index, column_name] = imputed_values[
                        column_name
                    ]
                else:
                    earlier_age_group, later_age_group = get_neighboring_age_groups(
                        age_groups, age_group
                    )

                    neighboring_age_groups = [age_group]
                    if earlier_age_group is not None:
                        neighboring_age_groups.append(earlier_age_group)
                    if later_age_group is not None:
                        neighboring_age_groups.append(later_age_group)

                    neighboring_data = df_data[
                        (df_data["AgeGroup"].isin(neighboring_age_groups))
                        & (df_data["Gender"] == gender_group)
                    ]
                    imputed_neighboring_values = fill_missing_values_by_probability(
                        neighboring_data, column_name
                    )
                    filtered_imputed_values = imputed_neighboring_values[
                        (imputed_neighboring_values["AgeGroup"] == age_group)
                        & (imputed_neighboring_values["Gender"] == gender_group)
                    ]
                    imputed_df.loc[filtered_imputed_values.index, column_name] = (
                        filtered_imputed_values[column_name]
                    )

    # Initialize context DataFrame
    imputed_df_context = pd.DataFrame(columns=df_context.columns)

    # Impute context values
    for age_group in age_groups:
        for gender_group in gender_groups:
            filtered_data = df_context[
                (df_context["AgeGroup"] == age_group)
                & (df_context["Gender"] == gender_group)
            ]
            rows_without_context = filtered_data[
                filtered_data[context_columns].sum(axis=1) == 0
            ]
            rows_with_context = filtered_data[
                filtered_data[context_columns].sum(axis=1) > 0
            ]

            column_sums = rows_with_context[context_columns].sum()
            total_context_rows = len(rows_with_context)
            column_probabilities = column_sums / total_context_rows

            column_probabilities = column_probabilities[column_probabilities > 0]
            if column_probabilities.sum() > 0:
                column_probabilities /= column_probabilities.sum()

            total_rows = len(filtered_data)
            proportion_without_context = len(rows_without_context) / total_rows

            if (
                accept_probability[0]
                <= proportion_without_context
                <= accept_probability[1]
            ):
                for idx in rows_without_context.index:
                    chosen_column = np.random.choice(
                        column_probabilities.index, p=column_probabilities
                    )
                    filtered_data.loc[idx, chosen_column] = 1

            imputed_df_context = pd.concat(
                [imputed_df_context, filtered_data], ignore_index=False
            )

    # Clean up context DataFrame
    imputed_df_context.drop(columns=["AgeGroup", "Gender"], inplace=True)

    # Merge imputed data
    imputed_data = pd.merge(imputed_df, imputed_df_context, on="ID", how="left")

    return imputed_data


if __name__ == "__main__":
    # Load raw data
    csv_file_path = DATA_DIR / "processed" / "mapped_data.csv"
    df_raw = read_csv(csv_file_path, delimiter=",", low_memory=False)

    # Run data imputation
    df_imputed = run_data_imputation(df_raw)

    # Save results
    file_name = "imputed_data.csv"
    output_file_path = DATA_DIR / "processed"
    write_csv(data=df_imputed, file_path=output_file_path / file_name, index=False)
