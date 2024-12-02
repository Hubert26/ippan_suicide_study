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
import glob

from config import *
from utils.dataframe_utils import read_csv_file, filter_dataframe
from utils.matplotlib_utils import create_multi_series_bar_chart_matplotlib, create_subplots_matplotlib, save_fig_matplotlib, create_multi_series_scatter_plot_matplotlib
from utils.file_utils import create_directory

#%%
#================================================================================
#Data reading, merging, settings
#================================================================================
#read data
csv_file_path = DATA_DIR / 'prepped' / 'final_feature_set.csv'
df_final = read_csv_file(csv_file_path, delimiter=',', low_memory=False, index_col=None)

csv_file_path = DATA_DIR / 'mapped' / 'mapped_data.csv'
df_raw = read_csv_file(csv_file_path, delimiter=',', low_memory=False, index_col=None)

csv_folder_path = RESULTS_DIR / 'poLCA'
csv_files = glob.glob(str(csv_folder_path / "*.csv"))
dataframes = [pd.read_csv(file, delimiter=',', low_memory=False, index_col=None) for file in csv_files]
from functools import reduce
lca_classes = reduce(lambda left, right: pd.merge(left, right, on='ID', how='outer'), dataframes)

csv_file_path = DATA_DIR / 'encoded' / 'encoded_data.csv'
df_encoded = read_csv_file(csv_file_path, delimiter=',', low_memory=False, index_col=None)

#%%
#merging
df_final = df_final.merge(lca_classes, on='ID', how='left')

df_raw = df_raw[df_raw['ID'].isin(df_final['ID'])]
selected_columns = ['ID'] + [col for col in df_final.columns if col.startswith('Group_')]
df_raw = df_raw.merge(df_final[selected_columns], on='ID', how='left')
df_raw = df_raw.merge(lca_classes, on='ID', how='left')

df_encoded = df_encoded.merge(lca_classes, on='ID', how='left')

#%%
#settings
group_types = [col for col in df_final.columns if col.startswith('Group_')]
years = sorted(list(set(df_final['DateY'].astype(int))))
context_columns = [col for col in df_final.columns if col.startswith('Context_')]
columns_to_plot = ['Education', 'WorkInfo', 'AbuseInfo', 'Income', 'Substance', 'Marital', 'Method', 'Place', 'Context', 'Gender', 'Fatal']

#%%







#%%
#================================================================================
#raw_vs_imputed
#================================================================================
create_directory(PLOTS_DIR / 'raw_vs_imputed')
plt.close('all')


# Iterate over all combinations of age groups and gender groups
for group in groups:
   
    # Set the number of plots based on the number of columns to plot
    n_plots = len(columns_to_plot)
            
    # Create subplots with 2 columns and enough rows to accommodate all plots
    fig, axes = create_subplots_matplotlib(n_plots, n_cols=2, figsize=(30, n_plots * 5))

    # Loop through each column to plot
    for i, column in enumerate(columns_to_plot):
        ax = axes[i]  # Get the current subplot axis
        
        # Filter the data for the current age group and gender
        filtered_data_final = filter_dataframe(df_final, Group=group)
        filtered_data_raw = filter_dataframe(df_raw, Group=group)
        
        # Convert the value counts to dictionaries for plotting
        if column == 'Context':
            distribution_final = {col.replace('Context_', ''): filtered_data_final[col].sum() for col in context_columns}
            distribution_raw = distribution_raw = {col.replace('Context_', ''): filtered_data_raw[col].sum() for col in context_columns}
        else:
            distribution_final = filtered_data_final[column].value_counts().to_dict()
            distribution_raw = filtered_data_raw[column].value_counts().to_dict()
        
        # Plot the distributions on the current axis using the custom plotting function
        create_multi_series_bar_chart_matplotlib([distribution_raw, distribution_final], 
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
    file_name = 'Group_' + group +'.png'
    save_fig_matplotlib(fig, PLOTS_DIR / 'raw_vs_imputed' / file_name)
    plt.close(fig)


#%%




#%%
#================================================================================
#fatal_vs_notfatal
#================================================================================
create_directory(PLOTS_DIR / 'fatal_vs_notfatal')
plt.close('all')


fatal_data = filter_dataframe(df_final, Fatal = 1)
not_fatal_data = filter_dataframe(df_final, Fatal = 0)

# Loop through each column to plot
for column in columns_to_plot:
    if column == 'Context':
        fatal_distribution = {col.replace('Context_', ''): fatal_data[col].sum() for col in context_columns}
        not_fatal_distribution = {col.replace('Context_', ''): not_fatal_data[col].sum() for col in context_columns}
    else:
        fatal_distribution = fatal_data[column].value_counts().to_dict()
        not_fatal_distribution = not_fatal_data[column].value_counts().to_dict()

    fig, ax = plt.subplots(figsize=(15,10))
    
    ax = create_multi_series_bar_chart_matplotlib(
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
    save_fig_matplotlib(fig, PLOTS_DIR / 'fatal_vs_notfatal' / file_name)
    plt.close(fig)
    
#%%






#%%
#================================================================================
#poLCA classes in time
#================================================================================
create_directory(PLOTS_DIR / 'poLCA_classes_time')
plt.close('all')

for group_type in group_types:
    create_directory(PLOTS_DIR / 'poLCA_classes_time' / group_type)
    
    pc_column = 'Predicted_Class_' + group_type
    groups = sorted(list(set(df_final[group_type])))
    poLCA_classes = sorted(list(set(df_final[pc_column])))
    
    for group in groups:
        group_data = filter_dataframe(df_final, **{group_type: group})
        
        data_series = []
        
        for poLCA_class in poLCA_classes:  
            data_x = []
            data_y = []
            
            for year in years:
                year_data = filter_dataframe(group_data, DateY = year)
                year_count = year_data['ID'].nunique()
                class_data = filter_dataframe(year_data, **{pc_column: poLCA_class})
                class_count = class_data['ID'].nunique()
                data_y.append(class_count/year_count*100)
                data_x.append(year) 
            
            data_series.append({'x': data_x, 'y': data_y, 'label': "Class " + str(poLCA_class)})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        create_multi_series_scatter_plot_matplotlib(data_series, ax, plot_title=group,
                                                    x_label='Year',
                                                    y_label='Suicide profiles among total suicide decedents, %',
                                                    linestyle='--',
                                                    curve_type='spline',
                                                    show_y_grid = True,
                                                    x_label_rotation = 45,
                                                    show_all_x_values = True)
        # Save plots to file
        file_name = group +'.png'
        save_fig_matplotlib(fig, PLOTS_DIR / 'poLCA_classes_time' / group_type / file_name)
        plt.close(fig)
    
#%%    
    
    
#%%
#================================================================================
# distribution of variables in poLCA classes
#================================================================================
create_directory(PLOTS_DIR / 'variables_poLCA_classes')
plt.close('all')


selected_columns = [col for col in df_encoded.columns if any(col.startswith(prefix) for prefix in columns_to_plot)]
selected_columns = sorted(selected_columns)

class_colors = ['blue', 'orange', 'green', 'red', 'purple']

for group_type in group_types:
    create_directory(PLOTS_DIR / 'variables_poLCA_classes' / group_type)
    
    if group_type == "Group_AF":
        columns = [col for col in selected_columns if col != "Fatal"]
    elif group_type == "Group_AG":
        columns = [col for col in selected_columns if col != "Gender"]
    elif group_type == "Group_AFG":
        columns = [col for col in selected_columns if col not in ["Fatal", "Gender"]]

    
    pc_column = 'Predicted_Class_' + group_type
    groups = sorted(list(set(df_final[group_type])))
    poLCA_classes = sorted(list(set(df_final[pc_column])))
    
    for group in groups:
        # Set the number of plots based on the number of classes
        n_plots = len(poLCA_classes)
        
        # Create subplots with 1 column and enough rows to accommodate all plots
        fig, axes = create_subplots_matplotlib(n_plots, n_cols=n_plots, figsize=(n_plots * 10, 50))
        fig.suptitle(group, fontsize=36)
        
        group_data = filter_dataframe(df_encoded, **{group_type: group})
        n_group = group_data['ID'].nunique()
        
        for i, poLCA_class in enumerate(poLCA_classes):
            ax = axes[i]  # Get the current subplot axis
            
            class_data = filter_dataframe(group_data, **{pc_column: poLCA_class})
            n_class = class_data['ID'].nunique()
            percentage = round(n_class / n_group * 100, 1)
    
            
            true_counts = class_data[columns].sum()
            data_x = true_counts.index.tolist()
            data_y = true_counts.values.tolist()
            data_y = [x / n_class * 100 for x in data_y]
            
                
            if i == 0:
                show_yticks = True
            else:
                show_yticks = False
                
            # Plot the distributions on the current axis using the custom plotting function
            create_multi_series_bar_chart_matplotlib(
                [dict(zip(data_x, data_y))],
                ax=ax,
                invert_axes=True,
                title_props={
                    'text': f"Class {poLCA_class}\n(n={n_class} [{percentage}%])",
                    'fontsize': 36,
                    'position': 'left'
                },
                axis_label_props={
                    'x_label': 'Individuals, %',
                    'y_label': '',
                    'x_fontsize': 26,
                    'x_label_show': True,
                    'y_label_show': False
                },
                legend_props={
                    'show_legend': False
                },
                ticks_props={
                    'show_yticks': False,#show_yticks,
                    'y_fontsize': 16
                },
                bar_props={
                    'width': 0.20,
                    'alpha': 0.5,
                    'color': [class_colors[i]]
                },
                grid_props={
                    'show_grid': True,
                    'axis': 'x'
                },
                additional_line={
                    'show': True,
                    'axis': 'y',
                    'function': lambda x: 0*x+50,
                    'linewidth': 2,
                    'color': 'red',
                    'linestyle': ':',
                    'alpha': 0.5
                },
                margins_props={
                    'y_margin': 0.01,
                    'x_margin': 0.01
                }
            )
        
        # Adjust the layout of the plots to avoid overlap
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
            
        # Save plots to file
        file_name = group +'.png'
        save_fig_matplotlib(fig, PLOTS_DIR / 'variables_poLCA_classes' / group_type / file_name)
        plt.close(fig)