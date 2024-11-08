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
    - kwargs: Additional keyword arguments for customization.
    
    Returns:
    - ax: Matplotlib axis with the created bar plot.
    """
    # Retrieve keyword arguments or use defaults
    fill_missing = kwargs.get('fill_missing', True)
    invert_axes = kwargs.get('invert_axes', False)
    
    # Title
    title_props = kwargs.get('title_props', {})
    title_props = {
        'text': title_props.get('text', 'Multiple Bar Plot'),
        'color': title_props.get('color', 'black'),
        'fontsize': title_props.get('fontsize', 12),
        'position': title_props.get('position', 'center')
    }
    
    # Axis labels
    axis_label_props = kwargs.get('axis_label_props', {})
    axis_label_props = {
        'x_label': axis_label_props.get('x_label', 'Categories'),
        'x_label_show': axis_label_props.get('x_label_show', True),
        'x_color': axis_label_props.get('x_color', 'black'),
        'x_fontsize': axis_label_props.get('x_fontsize', 10),
        'y_label': axis_label_props.get('y_label', 'Values'),
        'y_label_show': axis_label_props.get('y_label_show', True),
        'y_color': axis_label_props.get('y_color', 'black'),
        'y_fontsize': axis_label_props.get('y_fontsize', 10)
    }
    
    # Legend
    legend_props = kwargs.get('legend_props', {})
    legend_props = {
        'show_legend': legend_props.get('show_legend', True),
        'legend_labels': legend_props.get('legend_labels', [f'Series {i+1}' for i in range(len(data))]),
# =============================================================================
#         'location': legend_props.get('location', 'best'),
#         'fontsize': legend_props.get('fontsize', 10),
#         'text_color': legend_props.get('text_color', 'black'),
#         'frame_on': legend_props.get('frame_on', True),
#         'frame_color': legend_props.get('frame_color', 'gray'),
#         'frame_alpha': legend_props.get('frame_alpha', 0.8),
#         'background_alpha': legend_props.get('background_alpha', 0.5),
#         'ncol': legend_props.get('ncol', 1),
#         'borderpad': legend_props.get('borderpad', 1),
#         'labelspacing': legend_props.get('labelspacing', 0.5),
#         'handlelength': legend_props.get('handlelength', 2),
#         'handletextpad': legend_props.get('handletextpad', 0.5)
# =============================================================================
    }
    
    # Ticks
    ticks_props = kwargs.get('ticks_props', {})
    ticks_props = {
        'show_xticks': ticks_props.get('show_xticks', True),
        'x_fontsize': ticks_props.get('x_fontsize', 10),
        'x_color': ticks_props.get('x_color','black'),
        'x_rotation': ticks_props.get('x_rotation',0),
        'x_alignment': ticks_props.get('x_alignment', 'center'),
        'show_yticks': ticks_props.get('show_yticks', True),
        'y_fontsize': ticks_props.get('y_fontsize', 10),
        'y_color': ticks_props.get('y_color','black'),
        'y_rotation': ticks_props.get('y_rotation',0),
        'y_alignment': ticks_props.get('y_alignment', 'center'),
    }
    
    # Bars
    bar_props = kwargs.get('bar_props', {})
    bar_props = {
        'color': bar_props.get('color', plt.cm.tab10.colors),
        'alpha': bar_props.get('alpha', 0.8),
        'width': bar_props.get('width', 0.8),
        'edgecolor': bar_props.get('edgecolor', 'black'),
        'linewidth': bar_props.get('linewidth', 1.0),
        'hatch': bar_props.get('hatch', None),
        'align': bar_props.get('align', 'center'),
        'zorder': bar_props.get('zorder', 2)
    }
    
    # Grid
    grid_props = kwargs.get('grid_props', {})
    grid_props = {
        'show_grid': grid_props.get('show_grid', True),
        'axis': grid_props.get('axis', 'both'),
        'color': grid_props.get('color', 'gray'),
        'linestyle': grid_props.get('linestyle', '--'),
        'linewidth': grid_props.get('linewidth', 0.5),
        'alpha': grid_props.get('alpha', 0.7),
        'which': grid_props.get('which', 'major'),
        'zorder': grid_props.get('zorder', 1)
    }
    
    # Value labels
    value_labels_props = kwargs.get('value_labels_props', {})
    value_labels_props = {
        'show': value_labels_props.get('show', False),
        'format': value_labels_props.get('format', "{:.1f}"),
        'rotation': value_labels_props.get('rotation', 0),
        'color': value_labels_props.get('color', 'black'),
        'fontsize': value_labels_props.get('fontsize', 10),
        'fontweight': value_labels_props.get('fontweight', 'normal'),
        'va': value_labels_props.get('va', 'bottom'),
        'ha': value_labels_props.get('ha', 'center'),
        'offset': value_labels_props.get('offset', 5)
    }
    
    # Additional line
    additional_line = kwargs.get('additional_line', {})
    additional_line = {
        'show': additional_line.get('show', False),
        'show_in_legend': additional_line.get('show_in_legend', True),
        'axis': additional_line.get('axis', 'x'),
        'color': additional_line.get('color', 'gray'),
        'linewidth': additional_line.get('linewidth', 2),
        'linestyle': additional_line.get('linestyle', '--'),
        'alpha': additional_line.get('alpha', 0.7),
        'function': additional_line.get('function', lambda x: np.full_like(x, 0)),
    }
# =============================================================================
#     additional_line = {
#         'axis': additional_line.get('axis', 'y'),
#         'coefficients': additional_line.get('coefficients', [0, 0]),
#         'color': additional_line.get('color', 'red'),
#         'linewidth': additional_line.get('linewidth', 2),
#         'linestyle': additional_line.get('linestyle', '--'),
#         'alpha': additional_line.get('alpha', 0.8)
#     }
# =============================================================================
    
    # Margins
    margins_props = kwargs.get('margins_props', {})
    margins_props = {
        'x_margin': margins_props.get('x_margin', None),
        'y_margin': margins_props.get('y_margin', None)
        #'x_axis_padding': margins_props.get('x_axis_padding', 0.1),
        #'y_axis_padding': margins_props.get('y_axis_padding', 0.1),
    }
    
    # Extract all unique keys from the data to use as ticks
    ticks = sorted(set().union(*(d.keys() for d in data)))
    
    # Prepare the data for plotting, filling in missing keys if necessary
    plot_data = []
    for d in data:
        if fill_missing:
            plot_data.append([d.get(tick, 0) for tick in ticks])
        else:
            plot_data.append([d[tick] if tick in d else None for tick in ticks])

    # If no axis is provided, create one
    if ax is None:
        fig, ax = plt.subplots()

    # Calculate positions for the bars
    num_categories = len(ticks)
    total_width = bar_props['width'] * len(data)
    spacing = (1 - total_width) / (num_categories + 1)
    bar_positions = np.arange(num_categories) * (total_width + spacing)

    for i, series in enumerate(plot_data):
        if invert_axes:  # If axes are inverted, create horizontal bars
            bars = ax.barh(
                bar_positions + i * bar_props['width'],
                series,
                bar_props['width'],
                label=legend_props['legend_labels'][i], 
                color=bar_props['color'][i],
                alpha=bar_props['alpha']
            )
        else:
            bars = ax.bar(
                bar_positions + i * bar_props['width'],
                series,
                bar_props['width'],
                label=legend_props['legend_labels'][i], 
                color=bar_props['color'][i],
                alpha=bar_props['alpha']
            )

        # Add values on top of the bars
        if value_labels_props['show']:
            for bar in bars:
                value = bar.get_width() if invert_axes else bar.get_height()
                if value != 0:
                    if invert_axes:
                        ax.text(
                            value,
                            bar.get_y() + bar.get_height() / 2,
                            value_labels_props['format'].format(value),
                            va=value_labels_props['va'],
                            ha=value_labels_props['ha'],
                            rotation=value_labels_props['rotation']
                        )
                    else:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            value,
                            value_labels_props['format'].format(value),
                            va=value_labels_props['va'],
                            ha=value_labels_props['ha'],
                            rotation=value_labels_props['rotation']
                        )

    # Title
    ax.set_title(
        title_props['text'],
        color=title_props['color'],
        fontsize=title_props['fontsize'],
        loc=title_props['position']
    )

    # Axis labels
    if invert_axes:
        ax.set_ylabel(
            axis_label_props['y_label'],
            color=axis_label_props['y_color'],
            fontsize=axis_label_props['y_fontsize']
        )
        ax.set_yticks(bar_positions + bar_props['width'] * (len(data) - 1) / 2)
        ax.set_yticklabels(ticks, rotation=ticks_props['x_rotation'])
        if not ticks_props['show_xticks']:
            ax.xaxis.set_visible(False)
    else:
        ax.set_xlabel(
            axis_label_props['x_label'],
            color=axis_label_props['x_color'],
            fontsize=axis_label_props['x_fontsize']
        )
        ax.set_xticks(bar_positions + bar_props['width'] * (len(data) - 1) / 2)
        ax.set_xticklabels(ticks, rotation=ticks_props['x_rotation'])
        if not ticks_props['show_yticks']:
            ax.yaxis.set_visible(False)
            
    # Ticks
    if ticks_props['show_xticks']:
       ax.tick_params(axis='x', 
                      labelsize=ticks_props['x_fontsize'], 
                      labelcolor=ticks_props['x_color'], 
                      rotation=ticks_props['x_rotation'], 
                      labelleft=ticks_props['x_alignment'])
       
    if ticks_props['show_yticks']:
        ax.tick_params(axis='y', 
                       labelsize=ticks_props['y_fontsize'], 
                       labelcolor=ticks_props['y_color'], 
                       rotation=ticks_props['y_rotation'], 
                       labelleft=ticks_props['y_alignment'])
    
    # Add additional line if needed
    if additional_line['show']:
        # Generate x or y values based on the axis
        if additional_line['axis'] == 'x':
            x_values = np.linspace(*ax.get_xlim(), 100)
            y_values = additional_line['function'](x_values)
        elif additional_line['axis'] == 'y':
            y_values = np.linspace(*ax.get_ylim(), 100)
            x_values = additional_line['function'](y_values)
        
        # Plot the additional line if both x_values and y_values are available
        if x_values is not None and y_values is not None:
            ax.plot(
                x_values, y_values,
                color=additional_line['color'],
                linewidth=additional_line['linewidth'],
                linestyle=additional_line['linestyle'],
                alpha=additional_line['alpha'],
                label='Additional Line' if additional_line['show_in_legend'] else None
            )
        
        # Show legend only if necessary
        if (additional_line['show_in_legend'] and legend_props['show_legend']):
            ax.legend()

# =============================================================================
#     # Set margins and paddings
#     x_min = min(bar_positions) - margins_props['x_margin'] - margins_props['x_axis_padding']
#     x_max = max(bar_positions) + total_width + margins_props['x_margin'] + margins_props['x_axis_padding']
#     y_min = min(min(plot_data)) - margins_props['y_margin'] - margins_props['y_axis_padding']
#     y_max = max(max(plot_data)) + margins_props['y_margin'] + margins_props['y_axis_padding']
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)
# =============================================================================
    if margins_props['x_margin'] is not None:
        ax.margins(x=margins_props['x_margin'])
    
    if margins_props['y_margin'] is not None:
        ax.margins(y=margins_props['y_margin'])
    
    # Grid
    if grid_props['show_grid']:
        ax.grid(
            axis=grid_props['axis'],
            color=grid_props['color'],
            linestyle=grid_props['linestyle'],
            linewidth=grid_props['linewidth'],
            alpha=grid_props['alpha'],
            which=grid_props['which'],
            zorder=grid_props['zorder']
        )

    # Legend
    if legend_props['show_legend']:
        ax.legend(
            labels=legend_props['legend_labels'],
# =============================================================================
#             loc=legend_props['location'],
#             fontsize=legend_props['fontsize'],
#             frameon=legend_props['frame_on'],
#             framealpha=legend_props['frame_alpha'],
#             facecolor=legend_props['background_alpha'],
#             edgecolor=legend_props['frame_color'],
#             ncol=legend_props['ncol'],
#             borderpad=legend_props['borderpad'],
#             labelspacing=legend_props['labelspacing'],
#             handlelength=legend_props['handlelength'],
#             handletextpad=legend_props['handletextpad'],
#             labelcolor=legend_props['text_color']
# =============================================================================
        )

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
    create_multi_series_bar_chart_matplotlib(data1, ax=axes[0], title_props={'text': 'Wykres 1'}, legend_props={'legend_labels': ['Seria 1', 'Seria 2', 'Seria 3']})
    create_multi_series_bar_chart_matplotlib(data2, ax=axes[1], title_props={'text': 'Wykres 2'}, legend_props={'legend_labels': ['Seria A', 'Seria B']}, bar_props={'color': ['red', 'green']}, additional_line={'show': True, 'show_in_legend': False, 'axis': 'x', 'function': lambda x: x+3})
    create_multi_series_bar_chart_matplotlib(data3, ax=axes[2], title_props={'text': 'Wykres 3'}, invert_axes=True, legend_props={'legend_labels': ['Seria X', 'Seria Y']}, additional_line={'show': True, 'axis': 'y', 'function': lambda x: 0*x+3})
    create_multi_series_bar_chart_matplotlib(data4, ax=axes[3], title_props={'text': 'Wykres 4'}, legend_props={'show_legend': False}, bar_props={'alpha': 0.6}, ticks_props={'x_rotation': 45})
    

    # Adjust layout
    plt.tight_layout()

    # Save plots to file
    save_fig_matplotlib(fig, file_path=output_file_path / 'subplots_multi_series_bar_charts.png')

#%%