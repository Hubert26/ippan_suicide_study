"""
Data imputation module for handling missing values in suicide study dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

DATA_DIR = os.getenv('DATA_DIR')

np.random.seed(42)

output_file_path = Path(DATA_DIR) / 'imputed'

#================================================================================
# IMPUTATION FUNCTIONS
#================================================================================

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

#================================================================================
# MAIN
#================================================================================

# Data import
csv_file_path = Path(DATA_DIR) / 'mapped' / 'mapped_data.csv'
df_raw = pd.read_csv(csv_file_path)

# Split data and context
context_columns = [col for col in df_raw.columns if col.startswith('Context_')]
df_context = df_raw[context_columns + ["AgeGroup", "Gender", "ID"]]
df_data = df_raw.drop(columns=context_columns, inplace=False)

# Drop rows with missing critical data
selected_index = df_data[df_data['Gender'].isna() | df_data['AgeGroup'].isna() | df_data['Date'].isna()].index
df_data = df_data.drop(index=selected_index)
df_context = df_context.drop(index=selected_index)

# Fill missing values in AbuseInfo
df_data["AbuseInfo"] = df_data["AbuseInfo"].fillna("Not")

# Fill missing values in age groups
age_groups = sorted(list(set(df_data['AgeGroup'])))
gender_groups = sorted(list(set(df_data['Gender'])))
columns = df_data.columns.to_list()

accept_probability = (0,75)
imputed_df = df_data.copy()

# Impute missing values by age and gender groups
for age_group in age_groups:
    for gender_group in gender_groups:
        filtered_data = df_data[(df_data['AgeGroup'] == age_group) & (df_data['Gender'] == gender_group)]
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
    
                neighboring_data = df_data[(df_data['AgeGroup'].isin(neighboring_age_groups)) & (df_data['Gender'] == gender_group)]
                imputed_neighboring_values = fill_missing_values_by_probability(neighboring_data, column_name)
                filtered_imputed_values = imputed_neighboring_values[(imputed_neighboring_values['AgeGroup'] == age_group) & (imputed_neighboring_values['Gender'] == gender_group)]
                imputed_df.loc[filtered_imputed_values.index, column_name] = filtered_imputed_values[column_name]

# Initialize context DataFrame
imputed_df_context = pd.DataFrame(columns=df_context.columns)

# Impute context values
for age_group in age_groups:
    for gender_group in gender_groups:
        filtered_data = df_context[(df_context['AgeGroup'] == age_group) & (df_context['Gender'] == gender_group)]
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
imputed_df_context.drop(columns=['AgeGroup', 'Gender'], inplace=True)

# Merge imputed data
imputed_data = pd.merge(imputed_df, imputed_df_context, on='ID', how='left')

# Save results
file_name = 'imputed_data.csv'
imputed_data.to_csv(output_file_path / file_name, index=False)
