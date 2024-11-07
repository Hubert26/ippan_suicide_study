# -*- coding: utf-8 -*-
"""data_exploration.ipynb"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


from config import *
from utils.dataframe_utils import read_csv_file, write_to_csv

np.random.seed(42)

#%%
output_file_path = DATA_DIR / 'imputed'

#%%



#%%
def analyze_dataframe_columns(dataframe):
    """
    Analyzes the columns of a DataFrame by calculating both missing data statistics
    and the number of unique values with their counts for each column.

    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
                      - 'column_name': The name of each column in the DataFrame.
                      - 'missing_values_total': The total count of missing values in each column.
                      - 'missing_values_percent': The percentage of missing values in each column.
                      - 'unique_values_count': The count of unique values in each column.
                      - 'unique_value_counts': A string representation of unique values and their counts.
    """
    # List to store the results
    results = []

    # Iterate over each column in the DataFrame
    for col in dataframe.columns:
        # Missing data statistics
        missing_total = dataframe[col].isnull().sum()
        missing_percent = 100 * missing_total / len(dataframe)
        
        # Unique values and their counts
        unique_count = dataframe[col].nunique()
        value_counts = dataframe[col].value_counts().to_dict()
        value_counts_str = ', '.join([f'{k}: {v}' for k, v in value_counts.items()])
        
        # Append the results for this column
        results.append([
            col, 
            missing_total, 
            missing_percent, 
            unique_count, 
            value_counts_str
        ])
    
    # Convert the list to a DataFrame
    analysis_df = pd.DataFrame(
        results, 
        columns=['column_name', 'missing_values_total', 'missing_values_percent', 'unique_values_count', 'unique_value_counts']
    )
    
    return analysis_df

#%%
def nan_exploration_in_rows(dataframe):
    """
    Explores the distribution of NaN values across rows in a DataFrame.

    This function calculates:
    1. The total number of rows with a specific count of NaN values.
    2. The percentage of rows for each count of NaN values.
    3. The number of NaNs in each row.

    Missing counts are filled with zeros to ensure a complete overview.

    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: A DataFrame with three columns:
                      - 'NaN_count': The count of NaN values in each row.
                      - 'Total': The total number of rows with that NaN count.
                      - 'Percent': The percentage of rows with that NaN count.
    """
    # Step 1: Count NaNs in each row
    nan_counts = dataframe.isna().sum(axis=1).value_counts()

    # Step 2: Create full index range (from 0 to total number of columns)
    full_index = list(range(0, len(dataframe.columns) + 1))

    # Step 3: Reindex to fill missing values with 0
    nan_counts = nan_counts.reindex(full_index, fill_value=0)

    # Step 4: Sort by NaN count (ascending)
    nan_counts = nan_counts.sort_index()

    # Step 5: Calculate percentage of rows for each NaN count
    nan_counts_percent = (nan_counts / len(dataframe)) * 100

    # Step 6: Combine total and percentage into a single DataFrame
    missing_data_rows = pd.concat([pd.Series(full_index, name='NaN_count'), 
                                   nan_counts.rename('Total'), 
                                   nan_counts_percent.rename('Percent')], axis=1)

    return missing_data_rows

#%%
def filter_dataframe(dataframe, **filters):
    """
    Filters the DataFrame based on the specified column criteria without resetting or changing the index.

    Args:
        dataframe (pd.DataFrame): The DataFrame to filter.
        **filters: Key-value pairs where the key is the column name and the value is the filter criterion.
                   The value can be a single value or a list of values for filtering.
                   Example: filter_dataframe(df, GroupAge1=30, Gender=['F', 'M'])

    Returns:
        pd.DataFrame: The filtered DataFrame that matches all the filter conditions, 
                      with the original index preserved.
    """
    filtered_dataframe = dataframe.copy()

    # Apply each filter condition to the DataFrame
    for column, value in filters.items():
        # If the filter value is a list, use the isin() method for multiple values
        if isinstance(value, list):
            filtered_dataframe = filtered_dataframe[filtered_dataframe[column].isin(value)]
        else:
            # Otherwise, filter for exact matches
            filtered_dataframe = filtered_dataframe[filtered_dataframe[column] == value]
    
    return filtered_dataframe

#%%
def get_neighboring_age_groups(age_groups, current_age_group):
    """
    Get the neighboring age groups (previous and next) for a given age group.

    Args:
        age_groups (list): List of sorted age groups.
        current_age_group (str/int): The current age group.

    Returns:
        tuple: A tuple containing the previous and next age group (None if not applicable).
    """
    earlier_age_group = age_groups[age_groups.index(current_age_group) - 1] if current_age_group != age_groups[0] else None
    later_age_group = age_groups[age_groups.index(current_age_group) + 1] if current_age_group != age_groups[-1] else None
    return earlier_age_group, later_age_group

#%%
def filter_columns_by_missing_data(dataframe, accept_probability=(0, 100), columns=None):
    """
    Calculate the missing data percentages and filter columns based on a threshold.

    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze for missing data.
        accept_probability (tuple): The range of acceptable missing data percentages (default: (0, 100)).
        columns (list, optional): The list of columns to check. If None, all columns are considered (default: None).

    Returns:
        list: A list of columns that have missing data percentages within the specified range.
    """
    if columns is None:
        columns = dataframe.columns

    missing_percent = dataframe[columns].isnull().mean() * 100
    return missing_percent[(missing_percent < accept_probability[1]) & (missing_percent > accept_probability[0])].index.tolist()

#%%
def fill_missing_values_by_probability(dataframe, column_name):
    """
    Fill missing values in a specified column of a DataFrame based on the probability distribution
    of the existing values in that column.

    Args:
        dataframe (pd.DataFrame): The DataFrame to modify.
        column_name (str): The column name to fill missing values for.

    Returns:
        pd.DataFrame: A new DataFrame with missing values filled for the specified column.
    """
    # Create a copy of the original dataframe to avoid modifying it
    dataframe = dataframe.copy()

    # Get the rows with missing values for the specified column
    null_index = dataframe[dataframe[column_name].isnull()].index

    # Calculate value counts of existing (non-null) values in the column
    value_counts_result = dataframe[column_name].value_counts()

    if not value_counts_result.empty:
        # Extract the unique values and their probabilities
        value_list = value_counts_result.index.tolist()
        probabilities = [count / value_counts_result.sum() for count in value_counts_result.values]

        # Use np.random.choice to fill the missing values based on the probability distribution
        dataframe.loc[null_index, column_name] = np.random.choice(value_list, size=len(null_index), p=probabilities)
    
    return dataframe


#%%
"""#Data import"""
csv_file_path = DATA_DIR / 'mapped' / 'mapped_data.csv'
df_raw = read_csv_file(csv_file_path, low_memory=False)
#df_raw = read_csv_file(csv_file_path, delimiter=',', low_memory=False, index_col=None, dtype={'DateY': str, 'DateM': str})


#%%
#df_data & df_context
context_columns = [col for col in df_raw.columns if col.startswith('Context_')]
df_context = df_raw[context_columns + ["GroupAge1", "Gender", "ID"]]
df_data = df_raw.drop(columns=context_columns, inplace=False)

#%%
"""Exploration"""
analysis_df = analyze_dataframe_columns(df_data)
missing_data_rows = nan_exploration_in_rows(df_data)


#%%
# Drop rows with NaN in Gender, GroupAge1, Date
selected_index = df_data[df_data['Gender'].isna() | df_data['GroupAge1'].isna() | df_data['Date'].isna()].index
df_data = df_data.drop(index=selected_index)
df_context = df_context.drop(index=selected_index)

#%%
# Fill NaN in AbuseInfo
df_data["AbuseInfo"] = df_data["AbuseInfo"].fillna("Not")

#%%
"""#Fill NaN in age groups"""
age_groups = sorted(list(set(df_data['GroupAge1'])))
gender_groups = sorted(list(set(df_data['Gender'])))
columns = df_data.columns.to_list()

accept_probability = (0,75)
imputed_df = df_data.copy()

#%%
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

#%%
# Initialize an empty DataFrame with the same columns as df_context
imputed_df_context = pd.DataFrame(columns=df_context.columns)

# Iterate over each combination of age group and gender group
for age_group in age_groups:
    for gender_group in gender_groups:
        # Filter the data based on age and gender groups
        filtered_data = filter_dataframe(df_context, GroupAge1=age_group, Gender=gender_group)
        
        # Find rows where all context columns have 0
        rows_without_context = filtered_data[filtered_data[context_columns].sum(axis=1) == 0]
        
        # Find rows where at least one context column has 1
        rows_with_context = filtered_data[filtered_data[context_columns].sum(axis=1) > 0]
        
        # Calculate column sums for rows with context
        column_sums = rows_with_context[context_columns].sum()
        
        # Calculate probabilities for each context column
        total_context_rows = len(rows_with_context)
        column_probabilities = column_sums / total_context_rows
        
        # Remove columns with zero probability and normalize probabilities
        column_probabilities = column_probabilities[column_probabilities > 0]
        if column_probabilities.sum() > 0:
            column_probabilities /= column_probabilities.sum()
        
        # Calculate the proportion of rows with context
        total_rows = len(filtered_data)
        proportion_without_context = len(rows_without_context) / total_rows
        
        # Check if the proportion is within the acceptable range
        if accept_probability[0] <= proportion_without_context <= accept_probability[1]:
            # Handle rows without context
            for idx in rows_without_context.index:
                # Randomly choose a context column based on the calculated probabilities
                chosen_column = np.random.choice(
                    column_probabilities.index, 
                    p=column_probabilities
                )
                
                # Set the value to 1 in the chosen column
                filtered_data.loc[idx, chosen_column] = 1
        
        # Append the modified filtered_data to imputed_df_context
        imputed_df_context = pd.concat([imputed_df_context, filtered_data], ignore_index=False)
                
#%%
imputed_df_context.drop(columns=['GroupAge1', 'Gender'], inplace=True)


#%%
"""#Checking"""
missing_data_rows = nan_exploration_in_rows(imputed_df)
analysis_df = analyze_dataframe_columns(imputed_df)

rows_without_context = imputed_df_context[imputed_df_context[context_columns].sum(axis=1) == 0]

#%%
imputed_data = pd.merge(imputed_df, imputed_df_context, on='ID', how='left')

#%%
"""#Zapis"""  
file_name = 'imputed_data.csv'
write_to_csv(imputed_data, output_file_path / file_name, index=False)
