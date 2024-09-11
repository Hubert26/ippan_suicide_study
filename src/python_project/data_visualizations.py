# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 12:09:58 2024

@author: huber
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from utils.dataframe_utils import read_csv_file, filter_dataframe
from utils.visualizations_utils import create_multiple_bar_plot, create_subplots, save_plot


#%%
current_working_directory = Path.cwd()
grandparent_directory = current_working_directory.parent.parent
output_file_path = grandparent_directory / 'data' / 'plots'

#%%



#%%
"""#Data import"""
csv_file_path = grandparent_directory / 'data' / 'imputed' / 'imputed_data.csv'
df_imputed = read_csv_file(csv_file_path, delimiter=',', low_memory=False, index_col=None, dtype={'DateY': str, 'DateM': str})

csv_file_path = grandparent_directory / 'data' / 'mapped' / 'mapped_data.csv'
df_raw = read_csv_file(csv_file_path, delimiter=',', low_memory=False, index_col=None, dtype={'DateY': str, 'DateM': str})

#%%
age_groups = sorted(list(set(df_imputed['GroupAge1'])))
gender_groups = sorted(list(set(df_imputed['Gender'])))
context_columns = [col for col in df_imputed.columns if col.startswith('Context_')]
columns_to_plot = ['Education', 'WorkInfo', 'AbuseInfo', 'Income', 'Substance', 'Marital', 'Method', 'Place', 'Context']
#%%
# Iterate over all combinations of age groups and gender groups
for age_group in age_groups:
    for gender_group in gender_groups:
        
        # Set the number of plots based on the number of columns to plot
        n_plots = len(columns_to_plot)
                
        # Create subplots with 2 columns and enough rows to accommodate all plots
        fig, axes = create_subplots(n_plots, n_cols=2, figsize=(30, n_plots * 5))

        # Loop through each column to plot
        for i, column in enumerate(columns_to_plot):
            ax = axes[i]  # Get the current subplot axis
            
            # Filter the data for the current age group and gender
            filtered_data_imputed = filter_dataframe(df_imputed, GroupAge1=age_group, Gender=gender_group)
            filtered_data_raw = filter_dataframe(df_raw, GroupAge1=age_group, Gender=gender_group)
            
            # Convert the value counts to dictionaries for plotting
            if column == 'Context':
                distribution_imputed = {col.replace('Context_', ''): filtered_data_imputed[col].sum() for col in context_columns}
                distribution_raw = distribution_raw = {col.replace('Context_', ''): filtered_data_raw[col].sum() for col in context_columns}
            else:
                distribution_imputed = filtered_data_imputed[column].value_counts().to_dict()
                distribution_raw = filtered_data_raw[column].value_counts().to_dict()
            
            # Plot the distributions on the current axis using the custom plotting function
            create_multiple_bar_plot([distribution_raw, distribution_imputed], 
                                     ax=ax,
                                     plot_title=column,
                                     legend_labels=["Raw", "Imputed"],
                                     y_label='Count',
                                     bar_width=0.20,
                                     show_values=True,
                                     x_label_rotation=45,
                                     value_rotation = 90)
        
        # Adjust the layout of the plots to avoid overlap
        plt.tight_layout()
            
        # Save plots to file
        file_name = age_group + '_' + gender_group + '.png'
        save_plot(fig, output_file_path / 'raw_vs_imputed' / file_name)

#%%
columns_to_plot = ['Gender', 'GroupAge2', 'Education', 'WorkInfo', 'AbuseInfo', 'Income', 'Substance', 'Marital', 'Method', 'Place', 'Context']

fatal_data = filter_dataframe(df_imputed, Fatal = 1)
not_fatal_data = filter_dataframe(df_imputed, Fatal = 0)

# Loop through each column to plot
for column in columns_to_plot:
    if column == 'Context':
        fatal_distribution = {col.replace('Context_', ''): fatal_data[col].sum() for col in context_columns}
        not_fatal_distribution = {col.replace('Context_', ''): not_fatal_data[col].sum() for col in context_columns}
    else:
        fatal_distribution = fatal_data[column].value_counts().to_dict()
        not_fatal_distribution = not_fatal_data[column].value_counts().to_dict()

    fig, ax = plt.subplots(figsize=(15,10))
    
    ax = create_multiple_bar_plot(
        [fatal_distribution, not_fatal_distribution],  # dane dystrybucji
        ax=ax,
        plot_title=column,
        legend_labels=["Fatal", "Not Fatal"],  # Opis serii danych
        y_label='Count',
        bar_width=0.20,
        show_values=True,
        x_label_rotation=45,
        value_rotation=90
    )
    
    # Save plots to file
    file_name = column + '.png'
    save_plot(fig, output_file_path / 'fatal_vs_notfatal' / file_name)


