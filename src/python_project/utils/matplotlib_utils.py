# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:28:28 2024

@author: huber
"""


import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from utils.file_utils import create_directory

#%%
def create_scatter_plot(x_data, y_data, ax, title='', xlabel='', ylabel='', color='blue', alpha=0.5):
    """
    Creates a scatter plot for the specified x and y data.

    Parameters:
    - x_data (list or np.array): The data for the x-axis.
    - y_data (list or np.array): The data for the y-axis.
    - ax (matplotlib.axes.Axes): The matplotlib Axes object where the plot will be drawn.
    - title (str, optional): The title for the plot. Default is an empty string.
    - xlabel (str, optional): The label for the x-axis. Default is an empty string.
    - ylabel (str, optional): The label for the y-axis. Default is an empty string.
    - color (str, optional): The color of the points in the plot. Default is 'blue'.
    - alpha (float, optional): The transparency of the points. Default is 0.5.

    Raises:
    - ValueError: If the lengths of x_data and y_data do not match.
    - TypeError: If the provided axis (`ax`) is not a valid matplotlib Axes object.
    """
    # Validate inputs
    if len(x_data) != len(y_data):
        raise ValueError("The lengths of x_data and y_data must match.")
    
    if not isinstance(ax, plt.Axes):
        raise TypeError("The provided ax is not a valid matplotlib Axes object.")
    
    # Create the scatter plot
    ax.scatter(x_data, y_data, color=color, alpha=alpha)

    # Set plot titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel or 'X-axis')
    ax.set_ylabel(ylabel or 'Y-axis')
    
#%%
def create_multi_series_scatter_plot_matplotlib(data_series, ax=None, **kwargs):
    """
    Creates a scatter plot with multiple data series and optional lines connecting points,
    using an existing Matplotlib axis (for use in subplots).

    Parameters:
    - data_series (list of dicts): A list where each dictionary represents a dataset to plot.
                                   Each dictionary should have keys 'x' and 'y' for data points and optionally 'label' for legend.
    - ax (matplotlib.axes.Axes, optional): Matplotlib axis object to plot on. If None, creates a new axis.
    - kwargs: Additional keyword arguments for customization.

    Additional Keyword Arguments:
    - scatter_colors: List of colors for the scatter points and lines.
    - legend_labels: List of labels for the legend (overrides 'label' in data_series if provided).
    - plot_title: Title of the plot.
    - x_label: Label for the x-axis.
    - x_label_rotation: Angle (in degrees) for rotating x-axis category labels (default is 0).
    - y_label: Label for the y-axis.
    - alpha: Transparency of the points (default is 0.5).
    - line_alpha: Transparency of the lines (default is 0.5).
    - linestyle: Style of the connecting line (default is None, meaning no line).
    - curve_type: Type of line connection ('line' for straight line, 'spline' for smooth curve, default is 'line').
    - legend_position: Position of the legend (default is 'best').
    - show_grid: Boolean to show gridlines (default is False).

    Returns:
    - ax: Matplotlib axis with the created scatter plot.
    """
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Set default values for optional keyword arguments
    scatter_colors = kwargs.get('scatter_colors', ['blue', 'orange', 'green', 'red', 'purple'])
    legend_labels = kwargs.get('legend_labels', [d.get('label', f'Series {i+1}') for i, d in enumerate(data_series)])
    plot_title = kwargs.get('plot_title', '')
    x_label = kwargs.get('x_label', '')
    x_label_rotation = kwargs.get('x_label_rotation', 0)
    y_label = kwargs.get('y_label', '')
    alpha = kwargs.get('alpha', 0.5)
    line_alpha = kwargs.get('line_alpha', 0.5)
    linestyle = kwargs.get('linestyle', None)  # Default to no line
    curve_type = kwargs.get('curve_type', 'line')  # Default to straight line
    legend_position = kwargs.get('legend_position', 'best')
    show_x_grid = kwargs.get('show_x_grid', False)
    show_y_grid = kwargs.get('show_y_grid', False)
    show_all_x_values = kwargs.get('show_all_x_values', False)

    # Plot each data series
    for i, series in enumerate(data_series):
        x_data = series.get('x')
        y_data = series.get('y')
        label = legend_labels[i]
        color = scatter_colors[i % len(scatter_colors)]

        # Plot scatter points
        ax.scatter(x_data, y_data, color=color, alpha=alpha, label=label)
        
        # Plot line connecting the points if linestyle is not None
        if linestyle is not None:
            # Plot line or spline curve connecting the points based on curve_type
            if curve_type == 'spline'and len(x_data) >= 4:
                from scipy.interpolate import make_interp_spline
                import numpy as np
                x_data = np.array(x_data)
                y_data = np.array(y_data)
                # Create smooth curve
                x_smooth = np.linspace(x_data.min(), x_data.max(), 300)  # More points for smoothness
                spline = make_interp_spline(x_data, y_data, k=3)  # k=3 is for cubic spline
                y_smooth = spline(x_smooth)
                ax.plot(x_smooth, y_smooth, linestyle=linestyle, color=color, alpha=line_alpha)
            elif curve_type == 'line':
                # Plot straight line connecting points
                ax.plot(x_data, y_data, linestyle=linestyle, color=color, alpha=line_alpha)

    # Set plot title and labels
    ax.set_title(plot_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Show grid if specified
    ax.grid(axis='x', visible=show_x_grid)
    ax.grid(axis='y', visible=show_y_grid)

    # Set x-axis tick labels with rotation
    if show_all_x_values:
        ax.set_xticks(data_series[0]['x'])  # Assuming x values are the same for each series
        ax.set_xticklabels(data_series[0]['x'], rotation=x_label_rotation)  # Set the tick labels with rotation
    else:
        ax.set_xticklabels([], rotation=x_label_rotation)  # Clear labels or set to a default value if not showing all
    
    # Show legend
    ax.legend(loc=legend_position)

    return ax

#%%
def create_bar_plot(data, column, ax, title='', xlabel='', ylabel='Count', color='blue'):
    """
    Creates a bar plot for the specified column.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data to plot.
    - column (str): The column name for which the distribution will be plotted.
    - ax (matplotlib.axes.Axes): The matplotlib Axes object where the plot will be drawn.
    - title (str, optional): The title for the plot. Default is an empty string.
    - xlabel (str, optional): The label for the x-axis. Default is an empty string.
    - ylabel (str, optional): The label for the y-axis. Default is 'Count'.
    - color (str, optional): The color of the bars in the plot. Default is 'blue'.
    
    Raises:
    - ValueError: If the specified column is not present in the DataFrame.
    - TypeError: If the provided axis (`ax`) is not a valid matplotlib Axes object.
    """
    # Validate inputs
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the provided DataFrame.")
    
    if not isinstance(ax, plt.Axes):
        raise TypeError("The provided ax is not a valid matplotlib Axes object.")
    
    # Calculate the value counts for the column
    distribution = data[column].value_counts()

    # Create the bar plot
    distribution.plot(kind='bar', ax=ax, color=color)

    # Set plot titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel(ylabel)

#%%
def create_multi_series_bar_chart_matplotlib(data, ax=None, **kwargs):
    """
    Creates a bar plot with multiple data series using an existing Matplotlib axis (for use in subplots).
    
    Parameters:
    - data: List of dictionaries, where each dictionary represents a dataset to plot.
            The keys of the dictionaries are used as the x-axis labels.
    - ax: Matplotlib axis object to plot on. If None, creates a new axis.
    - fill_missing: Boolean, whether to fill missing keys with 0 (default is True).
    - invert_axes: Boolean, whether to invert the axes (default is False).
    - kwargs: Additional keyword arguments for customization.
    
    Returns:
    - ax: Matplotlib axis with the created bar plot.
    """
    
    # Extract all unique keys from the data to use as labels for the x-axis
    labels = sorted(set().union(*(d.keys() for d in data)))
    
    # Retrieve keyword arguments or use defaults
    fill_missing = kwargs.get('fill_missing', True),
    invert_axes = kwargs.get('invert_axes', False),
    legend_labels = kwargs.get('legend_labels', [f'Series {i+1}' for i in range(len(data))])
    bar_colors = kwargs.get('bar_colors', plt.cm.tab10.colors)
    plot_title = kwargs.get('plot_title', 'Multiple Bar Plot')
    x_label = kwargs.get('x_label', 'Categories')
    y_label = kwargs.get('y_label', 'Values')
    bar_width = kwargs.get('bar_width', 0.35)
    legend_position = kwargs.get('legend_position', 'best')
    show_grid = kwargs.get('show_grid', False)
    show_x_grid = kwargs.get('show_x_grid', False)
    show_y_grid = kwargs.get('show_y_grid', False)
    show_values = kwargs.get('show_values', False)
    value_format = kwargs.get('value_format', "{:.1f}")
    x_label_rotation = kwargs.get('x_label_rotation', 0)
    y_label_rotation = kwargs.get('y_label_rotation', 0)
    value_rotation = kwargs.get('value_rotation', 0)
    title_position = kwargs.get('title_position', 'center')
    show_legend = kwargs.get('show_legend', True)
    show_x_labels = kwargs.get('show_x_labels', True)
    show_y_labels = kwargs.get('show_y_labels', True)
    show_x_values = kwargs.get('show_x_values', True)
    show_y_values = kwargs.get('show_y_values', True)
    alpha = kwargs.get('alpha', 1.0)  # Default is fully opaque
    y_axis_margin = kwargs.get('y_axis_margin', 0.05)
    y_grid_midpoint = kwargs.get('y_grid_midpoint', None)
    
    x_grid_midpoint = kwargs.get('x_grid_midpoint', None)
    # Only create a default dictionary if x_grid_midpoint is not provided
    if x_grid_midpoint is not None and isinstance(x_grid_midpoint, dict):
        # Ensure defaults are used if the dictionary is missing some keys
        x_grid_midpoint = {**{
            'value': None,
            'linewidth': 1.5,
            'dashes': (10, 5),
            'color': 'gray',
            'linestyle': '--'
        }, **x_grid_midpoint}
    else:
        # If it's None or not provided, no need to create anything
        x_grid_midpoint = None



    # Prepare the data for plotting, filling in missing keys if necessary
    plot_data = []
    for d in data:
        if fill_missing:
            plot_data.append([d.get(label, 0) for label in labels])
        else:
            plot_data.append([d[label] if label in d else None for label in labels])

    # If no axis is provided, create one
    if ax is None:
        fig, ax = plt.subplots()

    # Calculate positions for the bars
    num_categories = len(labels)
    total_width = bar_width * len(data)
    spacing = (1 - total_width) / (num_categories + 1)
    x = np.arange(num_categories) * (total_width + spacing)

    for i, series in enumerate(plot_data):
        if invert_axes:  # If axes are inverted, create horizontal bars
            bars = ax.barh(x + i * bar_width, series, bar_width, label=legend_labels[i], 
                           color=bar_colors[i], alpha=alpha)
        else:
            bars = ax.bar(x + i * bar_width, series, bar_width, label=legend_labels[i], 
                          color=bar_colors[i], alpha=alpha)

        # Add values on top of the bars
        if show_values:
            for bar in bars:
                value = bar.get_width() if invert_axes else bar.get_height()
                if value != 0:
                    if invert_axes:
                        ax.text(value, bar.get_y() + bar.get_height() / 2,
                                value_format.format(value), va='center', ha='left', rotation=value_rotation)
                    else:
                        ax.text(bar.get_x() + bar.get_width() / 2, value,
                                value_format.format(value), ha='center', va='bottom', rotation=value_rotation)

    # Customize the plot
    ax.set_title(plot_title, loc=title_position)

    if invert_axes:
        if show_x_labels:
            ax.set_xlabel(y_label)  # X-axis should show 'Values' (now horizontal)
        else:
            ax.set_xlabel("")

        if show_y_labels:
            ax.set_yticks(x + bar_width * (len(data) - 1) / 2)
            ax.set_yticklabels(labels, rotation=x_label_rotation)
        else:
            ax.set_ylabel("")

        if show_x_values:
            ax.xaxis.set_visible(True)
        else:
            ax.xaxis.set_visible(False)

        if show_y_values:
            ax.set_yticks(x + bar_width * (len(data) - 1) / 2)  # Y-axis now shows categories (previous X values)
            ax.set_yticklabels(labels,  rotation=y_label_rotation)  # Label categories on Y-axis
        else:
            ax.set_yticks([])

    else:
        if show_x_labels:
            ax.set_xlabel(x_label)
        else:
            ax.set_xlabel("")

        if show_y_labels:
            ax.set_ylabel(y_label)
        else:
            ax.set_ylabel("")

        if show_x_values:
            ax.set_xticks(x + bar_width * (len(data) - 1) / 2)
            ax.set_xticklabels(labels, rotation=x_label_rotation)
        else:
            ax.set_xticks([])

        if show_y_values:
            ax.yaxis.set_visible(True)
        else:
            ax.yaxis.set_visible(False)

    # Show grid based on user settings
    if show_grid:
        ax.grid(True)
    if show_x_grid:
        ax.xaxis.grid(True)
    if show_y_grid:
        ax.yaxis.grid(True)
    
    if x_grid_midpoint and x_grid_midpoint['value'] is not None:  # Proceed if x_grid_midpoint has a value
        # Get the current x-axis limits
        x_min, x_max = ax.get_xlim()

        # Calculate midpoint if 'auto' or use the specified value
        x_grid_midpoint['value'] = (x_min + x_max) / 2 if x_grid_midpoint['value'] == 'auto' else x_grid_midpoint['value']

        # Set the x-axis limits
        ax.set_xlim(x_grid_midpoint['value'] - (x_max - x_min) / 2, x_grid_midpoint['value'] + (x_max - x_min) / 2)

        # Add vertical line at the midpoint
        ax.axvline(x=x_grid_midpoint['value'], color=x_grid_midpoint['color'], linestyle=x_grid_midpoint['linestyle'], 
                   linewidth=x_grid_midpoint['linewidth'], dashes=x_grid_midpoint['dashes'])
        
    # Set a custom Y-axis range if `y_grid_midpoint` is specified
    if y_grid_midpoint is not None:
        # Get the current y-axis limits
        y_min, y_max = ax.get_ylim()
        
        # Calculate midpoint as the middle between y_min and y_max if y_grid_midpoint is True
        midpoint = (y_min + y_max) / 2 if y_grid_midpoint == 'auto' else y_grid_midpoint

        # Set the y-axis limit with midpoint in the center
        ax.set_ylim(midpoint - (y_max - y_min) / 2, midpoint + (y_max - y_min) / 2)

        # Ensure a grid line appears at the midpoint
        ax.axhline(y=midpoint, color='gray', linestyle='--', linewidth=0.5)

    # Set the Y-axis margins
    ax.margins(y=y_axis_margin)
        
    # Show legend if enabled
    if show_legend:
        ax.legend(loc=legend_position)

    return ax

#%%
def create_subplots_matplotlib(n_plots, n_cols=2, figsize=(30, 5)):
    """
    Creates a figure with subplots in a grid layout.

    Parameters:
    - n_plots (int): The number of subplots to create.
    - n_cols (int, optional): The number of columns in the subplot grid. Default is 2.
    - figsize (tuple, optional): The size of the figure in inches (width, height). Default is (15, 5).
    
    Returns:
    - fig (matplotlib.figure.Figure): The created matplotlib figure object.
    - axes (list of matplotlib.axes.Axes): A flattened list of axes objects (subplots).
    
    Raises:
    - ValueError: If the number of plots (`n_plots`) is less than 1 or if `n_cols` is less than 1.
    """
    # Validate inputs
    if n_plots < 1:
        raise ValueError("The number of plots (n_plots) must be at least 1.")
    
    if n_cols < 1:
        raise ValueError("The number of columns (n_cols) must be at least 1.")

    # Determine the number of rows required to fit the plots
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array for easy iteration
    axes = axes.flatten() if n_plots > 1 else [axes]

    return fig, axes

#%%
def save_fig_matplotlib(fig, file_path: str) -> None:
    """
    Save a Matplotlib or Seaborn plot to a file in the specified format and directory.

    This function saves a plot to a file with the format specified in the file_path extension 
    and ensures that the output directory exists. It supports both Matplotlib and Seaborn figures.

    Parameters:
    - fig (plt.Figure): The plot object to be saved. Can be a Matplotlib or Seaborn figure.
    - file_path (str): The path where the plot will be saved, including the file name and extension.

    Raises:
    - ValueError: If the file format (extracted from file_path) is not supported.
    - TypeError: If the provided figure is neither a Matplotlib nor a Seaborn figure.

    Returns:
    None
    """

    # List of supported formats
    supported_formats = ["png", "jpg", "svg", "pdf"]

    # Extract the format from the file path
    file_extension = Path(file_path).suffix.lstrip('.')
    
    # Ensure the provided format is valid
    if file_extension not in supported_formats:
        raise ValueError(f"Unsupported format '{file_extension}'. Supported formats are: {', '.join(supported_formats)}.")
    
    # Ensure the directory exists
    dir_path = Path(file_path).parent
    if dir_path.is_dir():
        create_directory(dir_path)
    
    # Check if the figure is a Matplotlib or Seaborn figure
    if isinstance(fig, plt.Figure):
        # Save the Matplotlib or Seaborn figure as an image file (PNG, JPG, SVG, PDF)
        fig.savefig(file_path, format=file_extension)
    else:
        raise TypeError("The 'fig' parameter must be a Matplotlib 'plt.Figure'.")
        
#%%
# =============================================================================
# def corr_heatmap(df, title=None, color='viridis'):
#     # Tworzenie własnej mapy kolorów z 20 odcieniami od -1 do 1
#     colors = sns.color_palette(color, 20)
#     cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=20)
#     
#     with sns.axes_style("white"):
#         f, ax = plt.subplots(figsize=(10, 10))
#         sns.heatmap(df,
# # =============================================================================
# # to annotate on heatmap you need previous version of matplotlib              
# # pip install matplotlib==3.7.3
# # =============================================================================
#                     annot=df.round(2),
#                     vmax=1,
#                     vmin=-1,
#                     center=0,
#                     square=True,
#                     xticklabels=df.columns,
#                     yticklabels=df.index,
#                     cmap=cmap,
#                     linewidths=.5,
#                     cbar_kws={"shrink": 0.7, 'ticks': np.linspace(-1, 1, 21)})
#         # Ustawienie rotacji etykiet
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#         ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#     
#     if not title:
#         title = 'heatmap'
#     
#     plt.title(title)
# 
#     return f, title
# =============================================================================


#%%
if __name__ == "__main__":
    current_working_directory = Path.cwd()
    output_file_path = current_working_directory / 'plots'
#%%
# =============================================================================
#     fig, ax = plt.subplots(figsize=(10, 6))
#     data_series = [
#          {'x': [1, 2, 3, 4], 'y': [4, 5, 6, 7], 'label': 'Series 1'},
#          {'x': [1, 2, 3, 4], 'y': [6, 4, 2, 1], 'label': 'Series 2'},
#          {'x': [1, 2, 3, 4], 'y': [5, 7, 8,10], 'label': 'Series 3'}
#     ]
#     create_multi_series_scatter_plot_matplotlib(data_series, ax, plot_title='Multi-Series Scatter Plot', x_label='X-axis', y_label='Y-axis', linestyle='--', curve_type='spline')
#     plt.show()
# =============================================================================

#%%
    # Sample data: list of dictionaries
    data1 = [{'A': 5, 'B': 3, 'C': 7}, {'A': 2, 'C': 4, 'D': 5}, {'B': 6, 'C': 3, 'D': 4}]
    data2 = [{'A': 4, 'B': 2, 'C': 5}, {'A': 1, 'B': 4, 'C': 6}]
    data3 = [{'X': 3, 'Y': 5, 'Z': 6}, {'X': 4, 'Y': 7}]
    data4 = [{'P': 6, 'Q': 8, 'R': 4}, {'P': 5, 'Q': 7}]

    # Create subplots: 4 plots in 2 columns
    fig, axes = create_subplots_matplotlib(n_plots=4, n_cols=2, figsize=(12, 8))

    # Use create_multiple_bar_plot to create plots in each subplot
    create_multi_series_bar_chart_matplotlib(data1, ax=axes[0],
                             plot_title="Plot 1",
                             legend_labels=["Series 1", "Series 2", "Series 3"],
                             bar_width = 0.20,
                             show_values = True,
                             x_label_rotation = 45,
                             invert_axes = True)
    
    create_multi_series_bar_chart_matplotlib(data2,
                             ax=axes[1],
                             plot_title="Plot 2",
                             legend_labels=["Series 1", "Series 2"],
                             bar_width = 0.20,
                             show_values = True,
                             x_label_rotation = 45,
                             invert_axes = True)
    
    create_multi_series_bar_chart_matplotlib(data3,
                             ax=axes[2],
                             plot_title="Plot 3",
                             legend_labels=["Series 1", "Series 2"],
                             bar_width = 0.20,
                             show_values = True,
                             x_label_rotation = 45,
                             invert_axes = True)
    
    create_multi_series_bar_chart_matplotlib(data4,
                             ax=axes[3],
                             plot_title="Plot 4",
                             legend_labels=["Series 1", "Series 2"],
                             bar_width = 0.20,
                             show_values = True,
                             x_label_rotation = 45,
                             invert_axes = True)

    # Adjust layout
    plt.tight_layout()

    # Save plots to file
    save_fig_matplotlib(fig, file_path=output_file_path / 'subplots_multi_series_bar_charts.png')

#%%