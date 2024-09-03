# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:40:41 2024

@author: Hubert26
"""

#%%
import os
from pathlib import Path
import shutil
import openpyxl
import pandas as pd

#%%
def read_excel_file(file_path):
    """
    Opens an Excel file from the provided path.
    
    Args:
        file_path (str): The path to the Excel file.
    
    Returns:
        pd.DataFrame: The contents of the Excel file as a DataFrame.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be opened or is not a valid Excel file.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    # Try to open the Excel file
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        raise ValueError(f"Failed to open the Excel file: {e}")
        
#%%
def read_csv_file(file_path, **kwargs):
    """
    Reads a CSV file into a DataFrame.

    This function uses pandas' read_csv method to load data from a CSV file into a DataFrame.
    It allows for various parameters used in pd.read_csv to be passed as keyword arguments.

    Args:
        file_path (str): The path to the CSV file to be read.
        **kwargs: Additional keyword arguments passed to pd.read_csv, e.g., dtype, sep, header, 
        index_col, na_values, low_memory.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the CSV file.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If there is an error parsing the file.
    """
    try:
        # Load the CSV file into a DataFrame with additional parameters
        dataframe = pd.read_csv(file_path, **kwargs)
        return dataframe
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(f"File not found: {file_path}") from fnf_error
    except pd.errors.EmptyDataError as ede_error:
        raise ValueError(f"The file is empty: {file_path}") from ede_error
    except pd.errors.ParserError as pe_error:
        raise ValueError(f"Error parsing the file: {file_path}") from pe_error
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

#%%
def write_to_excel(dataframe, file_path):
    """
    Writes a DataFrame to an Excel file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to write.
        file_path (str): The path where the Excel file will be saved.

    Raises:
        ValueError: If the DataFrame is not valid or if there is an error during file writing.
        OSError: If the file cannot be created or written to.
    """
    # Validate the input dataframe
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The provided input is not a valid pandas DataFrame.")
    
    # Attempt to write the DataFrame to an Excel file
    try:
        dataframe.to_excel(file_path, index=False, engine='openpyxl')
        print(f"DataFrame successfully written to {file_path}")
    except ValueError as ve:
        # Catch errors related to the DataFrame or file writing issues
        raise ValueError(f"Failed to write DataFrame to Excel: {ve}")
    except OSError as oe:
        # Catch errors related to file system issues
        raise OSError(f"Failed to create or write to file: {oe}")
    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(f"An unexpected error occurred: {e}")

#%%
def write_to_csv(dataframe, file_path, **kwargs):
    """
    Writes a DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be written to the CSV file.
        file_path (str): The path where the CSV file will be saved.
        **kwargs: Additional keyword arguments passed to pd.DataFrame.to_csv, e.g., header, index.

    Raises:
        ValueError: If the DataFrame is invalid or if there is an error during file writing.
        OSError: If the file cannot be created or written to.
    """
    # Validate the input dataframe
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The provided input is not a valid pandas DataFrame.")
    
    # Ensure the directory exists
    dir_path = Path(file_path).parent
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Attempt to write the DataFrame to a CSV file
    try:
        dataframe.to_csv(file_path, **kwargs)
        print(f"DataFrame successfully written to {file_path}")
    except ValueError as ve:
        # Catch errors related to the DataFrame or file writing issues
        raise ValueError(f"Failed to write DataFrame to CSV: {ve}")
    except OSError as oe:
        # Catch errors related to file system issues
        raise OSError(f"Failed to create or write to file: {oe}")
    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(f"An unexpected error occurred: {e}")

#%%
def create_directory(dir_path):
    """
    Creates a directory if it does not exist.

    Args:
        dir_path (str): The path to the directory.

    Raises:
        OSError: If the directory cannot be created.
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        raise OSError(f"Failed to create directory: {e}")
        
#%%
def list_file_paths(directory, extension=None):
    """
    Retrieves a list of file paths from the specified directory.

    This function scans the provided directory and returns a list of file paths. 
    If an extension is specified, only files with that extension are included in the list.
    The function uses the `pathlib` module for directory and file operations.

    Args:
        directory (str): The path to the directory to search. Must be a valid directory path.
        extension (str, optional): The file extension to filter by (e.g., '.xlsx'). 
                                   If None, all files are returned. Should include the dot ('.').

    Returns:
        list of str: A list of file paths (as strings) that match the specified extension. 
                     If no extension is specified, all files in the directory are included.

    Raises:
        ValueError: If the provided `directory` is not a valid directory or does not exist.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"The directory {directory} does not exist or is not a directory.")
    
    if extension:
        return [str(file) for file in dir_path.rglob(f'*{extension}') if file.is_file()]
    else:
        return [str(file) for file in dir_path.rglob('*') if file.is_file()]

#%%
def delete_file(file_path):
    """
    Deletes a file at the given path.

    Args:
        file_path (str): The path to the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be deleted.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    try:
        os.remove(file_path)
    except Exception as e:
        raise OSError(f"Failed to delete file: {e}")
        
#%%
def copy_file(src_path, dest_path):
    """
    Copies a file from the source path to the destination path.

    Args:
        src_path (str): The source file path.
        dest_path (str): The destination file path.

    Raises:
        FileNotFoundError: If the source file does not exist.
        OSError: If the file cannot be copied.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file does not exist: {src_path}")
    
    try:
        shutil.copy(src_path, dest_path)
    except Exception as e:
        raise OSError(f"Failed to copy file: {e}")
        
#%%
