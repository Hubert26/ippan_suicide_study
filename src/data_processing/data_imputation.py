"""
Data imputation module for handling missing values in suicide study dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

from python_data_analysis_utils.utils.dataframe_utils import read_csv_file, write_to_csv
from python_project.config.config import DATA_DIR

np.random.seed(42)

output_file_path = DATA_DIR / 'imputed'

def analyze_dataframe_columns(dataframe):
    """
    Analyzes DataFrame columns by calculating missing data statistics and unique value counts.

    Args:
        dataframe (pd.DataFrame): Input DataFrame for analysis.

    Returns:
        pd.DataFrame: Analysis results with columns:
            - column_name: Name of each column
            - missing_values_total: Count of missing values
            - missing_values_percent: Percentage of missing values
            - unique_values_count: Count of unique values
            - unique_value_counts: String of unique values and their counts
    """
    results = []
    
    for col in dataframe.columns:
        missing_total = dataframe[col].isnull().sum()
        missing_percent = 100 * missing_total / len(dataframe)
        
        unique_count = dataframe[col].nunique()
        value_counts = dataframe[col].value_counts().to_dict()
        value_counts_str = ', '.join([f'{k}: {v}' for k, v in value_counts.items()])
        
        results.append([
            col, 
            missing_total, 
            missing_percent, 
            unique_count, 
            value_counts_str
        ])
    
    analysis_df = pd.DataFrame(
        results, 
        columns=['column_name', 'missing_values_total', 'missing_values_percent', 'unique_values_count', 'unique_value_counts']
    )
    
    return analysis_df

def nan_exploration_in_rows(dataframe):
    """
    Analyzes NaN distribution across DataFrame rows.

    Args:
        dataframe (pd.DataFrame): Input DataFrame for analysis.

    Returns:
        pd.DataFrame: Analysis results with columns:
            - NaN_count: Number of NaN values in each row
            - Total: Count of rows with that NaN count
            - Percent: Percentage of rows with that NaN count
    """
    nan_counts = dataframe.isna().sum(axis=1).value_counts()
    full_index = list(range(0, len(dataframe.columns) + 1))
    nan_counts = nan_counts.reindex(full_index, fill_value=0)
    nan_counts = nan_counts.sort_index()
    nan_counts_percent = (nan_counts / len(dataframe)) * 100
    
    missing_data_rows = pd.concat([
        pd.Series(full_index, name='NaN_count'), 
        nan_counts.rename('Total'), 
        nan_counts_percent.rename('Percent')
    ], axis=1)
    
    return missing_data_rows

def filter_dataframe(dataframe, **filters):
    """
    Filters DataFrame based on column criteria while preserving index.

    Args:
        dataframe (pd.DataFrame): Input DataFrame to filter.
        **filters: Column name and value pairs for filtering.
                  Example: filter_dataframe(df, GroupAge1=30, Gender=['F', 'M'])

    Returns:
        pd.DataFrame: Filtered DataFrame with original index preserved.
    """
    filtered_dataframe = dataframe.copy()
    
    for column, value in filters.items():
        if isinstance(value, list):
            filtered_dataframe = filtered_dataframe[filtered_dataframe[column].isin(value)]
        else:
            filtered_dataframe = filtered_dataframe[filtered_dataframe[column] == value]
    
    return filtered_dataframe

def get_neighboring_age_groups(age_groups, current_age_group):
    """
    Gets adjacent age groups for a given age group.

    Args:
        age_groups (list): Sorted list of age groups.
        current_age_group (str/int): Target age group.

    Returns:
        tuple: Previous and next age groups (None if not applicable).
    """
    earlier_age_group = age_groups[age_groups.index(current_age_group) - 1] if current_age_group != age_groups[0] else None
    later_age_group = age_groups[age_groups.index(current_age_group) + 1] if current_age_group != age_groups[-1] else None
    return earlier_age_group, later_age_group

def filter_columns_by_missing_data(dataframe, accept_probability=(0, 100), columns=None):
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
        (missing_percent < accept_probability[1]) & 
        (missing_percent > accept_probability[0])
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
        probabilities = [count / value_counts_result.sum() for count in value_counts_result.values]
        dataframe.loc[null_index, column_name] = np.random.choice(value_list, size=len(null_index), p=probabilities)
    
    return dataframe

# Data import
csv_file_path = DATA_DIR / 'mapped' / 'mapped_data.csv'
df_raw = read_csv_file(csv_file_path, low_memory=False)

# Split data and context
context_columns = [col for col in df_raw.columns if col.startswith('Context_')]
df_context = df_raw[context_columns + ["GroupAge1", "Gender", "ID"]]
df_data = df_raw.drop(columns=context_columns, inplace=False)

# Data exploration
analysis_df = analyze_dataframe_columns(df_data)
missing_data_rows = nan_exploration_in_rows(df_data)

# Drop rows with missing critical data
selected_index = df_data[df_data['Gender'].isna() | df_data['GroupAge1'].isna() | df_data['Date'].isna()].index
df_data = df_data.drop(index=selected_index)
df_context = df_context.drop(index=selected_index)

# Fill missing values in AbuseInfo
df_data["AbuseInfo"] = df_data["AbuseInfo"].fillna("Not")

# Fill missing values in age groups
age_groups = sorted(list(set(df_data['GroupAge1'])))
gender_groups = sorted(list(set(df_data['Gender'])))
columns = df_data.columns.to_list()

accept_probability = (0,75)
imputed_df = df_data.copy()

# Impute missing values by age and gender groups
for age_group in age_groups:
    for gender_group in gender_groups:
        filtered_data = filter_dataframe(df_data, GroupAge1=age_group, Gender=gender_group)
        selected_columns = filter_columns_by_missing_data(filtered_data, accept_probability=accept_probability, columns=None)
        
        for column_name in filtered_data.columns.tolist():
            if column_name in selected_columns:
                imputed_values = fill_missing_values_by_probability(filtered_data, column_name)
                imputed_df.loc[imputed_values.index, column_name] = imputed_values[column_name]
            else:
                earlier_age_group, later_age_group = get_neighboring_age_groups(age_groups, age_group)
                
                neighboring_age_groups = [age_group]
                if earlier_age_group is not None:
                    neighboring_age_groups.append(earlier_age_group)
                if later_age_group is not None:
                    neighboring_age_groups.append(later_age_group)
    
                neighboring_data = filter_dataframe(df_data, GroupAge1=neighboring_age_groups, Gender=gender_group)
                imputed_neighboring_values = fill_missing_values_by_probability(neighboring_data, column_name)
                filtered_imputed_values = filter_dataframe(imputed_neighboring_values, GroupAge1=age_group, Gender=gender_group)
                imputed_df.loc[filtered_imputed_values.index, column_name] = filtered_imputed_values[column_name]

# Initialize context DataFrame
imputed_df_context = pd.DataFrame(columns=df_context.columns)

# Impute context values
for age_group in age_groups:
    for gender_group in gender_groups:
        filtered_data = filter_dataframe(df_context, GroupAge1=age_group, Gender=gender_group)
        rows_without_context = filtered_data[filtered_data[context_columns].sum(axis=1) == 0]
        rows_with_context = filtered_data[filtered_data[context_columns].sum(axis=1) > 0]
        
        column_sums = rows_with_context[context_columns].sum()
        total_context_rows = len(rows_with_context)
        column_probabilities = column_sums / total_context_rows
        
        column_probabilities = column_probabilities[column_probabilities > 0]
        if column_probabilities.sum() > 0:
            column_probabilities /= column_probabilities.sum()
        
        total_rows = len(filtered_data)
        proportion_without_context = len(rows_without_context) / total_rows
        
        if accept_probability[0] <= proportion_without_context <= accept_probability[1]:
            for idx in rows_without_context.index:
                chosen_column = np.random.choice(
                    column_probabilities.index, 
                    p=column_probabilities
                )
                filtered_data.loc[idx, chosen_column] = 1
        
        imputed_df_context = pd.concat([imputed_df_context, filtered_data], ignore_index=False)

# Clean up context DataFrame
imputed_df_context.drop(columns=['GroupAge1', 'Gender'], inplace=True)

# Data validation
missing_data_rows = nan_exploration_in_rows(imputed_df)
analysis_df = analyze_dataframe_columns(imputed_df)
rows_without_context = imputed_df_context[imputed_df_context[context_columns].sum(axis=1) == 0]

# Merge imputed data
imputed_data = pd.merge(imputed_df, imputed_df_context, on='ID', how='left')

# Save results
file_name = 'imputed_data.csv'
write_to_csv(imputed_data, output_file_path / file_name, index=False)
