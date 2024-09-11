# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:40:41 2024

@author: Hubert26
"""

#%%
import os
from pathlib import Path
import shutil


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
