from dotenv import dotenv_values
import os
from typing import Optional, List, Dict
from pathlib import Path
import pandas as pd


def load_environment_variables(
    env_file_path: str = ".env",
    required_vars: Optional[List[str]] = None,
) -> Dict[str, Optional[str]]:
    """
    Load all environment variables from a specified .env file and ensure required variables are present.

    Args:
        env_file_path (str): Path to the .env file. Default is '.env'.
        required_vars (Optional[List[str]]): A list of environment variable names that are required.
            If specified, the function will raise a ValueError if any of these variables are missing or empty.

    Returns:
        Dict[str, Optional[str]]: A dictionary containing all environment variables with their corresponding values
        from the .env file.

    Raises:
        FileNotFoundError: If the specified .env file does not exist.
        ValueError: If any of the required variables are missing or empty.
    """
    # Check if the .env file exists
    env_file = Path(env_file_path)
    if not env_file.exists():
        raise FileNotFoundError(f"The .env file at '{env_file_path}' does not exist!")

    # Load variables from the .env file
    env_vars = dotenv_values(env_file_path)

    # Check for required variables if specified
    if required_vars:
        missing_vars = [
            var for var in required_vars if var not in env_vars or not env_vars[var]
        ]
        if missing_vars:
            raise ValueError(
                f"Required environment variables are missing or empty: {', '.join(missing_vars)}"
            )

    return env_vars


def split_string(
    input_string: str, delimiter: str = ",", strip_whitespace: bool = True
) -> List[str]:
    """
    Split a string by a specified delimiter and optionally strip whitespace from each element.

    Args:
        input_string (str): The input string to be split.
        delimiter (str): The character used to split the string (default is ',').
        strip_whitespace (bool): Whether to strip whitespace from each split element (default is True).

    Returns:
        List[str]: A list of strings obtained by splitting the input string.
    """
    elements = input_string.split(delimiter)
    if strip_whitespace:
        return [element.strip() for element in elements]
    return elements


def check_file_exists(file_path, error_message=None):
    """
    Check if a file exists at the specified path.

    Args:
        file_path (str): The path of the file to check.
        error_message (str, optional): A custom error message if the file does not exist.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(error_message or f"File does not exist: {file_path}")


def check_folder_exists(folder_path, error_message=None):
    """
    Check if a folder exists at the specified path.

    Args:
        folder_path (str): The path of the folder to check.
        error_message (str, optional): A custom error message if the folder does not exist.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            error_message or f"Folder does not exist: {folder_path}"
        )


def create_folder(folder_path):
    """
    Create a folder if it does not exist.

    Args:
        folder_path (str): The path of the folder to create.

    Raises:
        OSError: If the folder cannot be created.
    """
    if not os.path.isdir(folder_path):
        print(f"Folder does not exist. Creating folder: {folder_path}")
        os.makedirs(folder_path, exist_ok=True)
        if os.path.isdir(folder_path):
            print(f"Folder {folder_path} was successfully created.")
        else:
            raise OSError(f"Failed to create folder: {folder_path}")
    else:
        print(f"Folder {folder_path} already exists.")


def read_csv(file_path, **kwargs):
    """
    Read a CSV file and return its content as a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.
        **kwargs: Additional arguments to pass to pandas.read_csv.

    Returns:
        pd.DataFrame: The contents of the CSV file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read.
    """
    check_file_exists(file_path, f"CSV file not found: {file_path}")
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {file_path}\nError: {e}")


def write_csv(data, file_path, **kwargs):
    """
    Write a pandas DataFrame to a CSV file with customizable options.

    Args:
        data (pd.DataFrame): The data frame to be written to the CSV file.
        file_path (str): The path to save the CSV file.
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_csv.

    Raises:
        ValueError: If the file cannot be written.
    """
    try:
        data.to_csv(file_path, **kwargs)
        print(f"CSV file successfully written to: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to write CSV file: {file_path}\nError: {e}")


def read_excel(file_path, **kwargs):
    """
    Read an Excel file and return its content as a pandas DataFrame.

    Args:
        file_path (str): The path to the Excel file.
        **kwargs: Additional arguments to pass to pandas.read_excel.

    Returns:
        pd.DataFrame: The contents of the Excel file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read.
    """
    # Check if the file exists
    check_file_exists(file_path, f"Excel file not found: {file_path}")

    # Try reading the Excel file
    try:
        return pd.read_excel(file_path, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {file_path}\nError: {e}")


def write_excel(file_path: str, data: pd.DataFrame, **kwargs):
    """
    Write a pandas DataFrame to an Excel file. Allows overwriting the file or appending to it.

    Args:
        file_path (str): The path to the Excel file.
        data (pd.DataFrame): The DataFrame to write to the file.
        **kwargs: Additional arguments for pandas Excel writer and `DataFrame.to_excel`.
            - mode (str): Writing mode ('w' to overwrite the file or 'a' to append). Defaults to 'w'.
            - engine (str): Engine to use for writing. Defaults to 'openpyxl'.
            - if_sheet_exists (str): Action if the sheet exists ('error', 'replace', or 'new'). Defaults to 'replace'.
            - sheet_name (str): The name of the sheet to write to. Defaults to 'Sheet1'.
            - Additional arguments supported by pandas `DataFrame.to_excel`.

    Raises:
        ValueError: If the writing mode or other parameters are invalid.
    """
    try:
        # Extract optional arguments with defaults
        mode = kwargs.pop("mode", "w")
        engine = kwargs.pop("engine", "openpyxl")

        # Handle if_sheet_exists only for append mode
        if mode == "a":
            if_sheet_exists = kwargs.pop("if_sheet_exists", "replace")
            with pd.ExcelWriter(
                file_path, mode=mode, engine=engine, if_sheet_exists=if_sheet_exists
            ) as writer:
                data.to_excel(writer, **kwargs)
        elif mode == "w":
            # Write mode (overwrite file entirely)
            with pd.ExcelWriter(file_path, mode=mode, engine=engine) as writer:
                data.to_excel(writer, **kwargs)
        else:
            raise ValueError("Invalid mode. Use 'w' for overwrite or 'a' for append.")

    except Exception as e:
        raise ValueError(f"Failed to write to Excel file: {file_path}. Error: {e}")


def left_join_excel_sheets(file_path, base_df=None, on=None, **kwargs):
    """
    Iteratively left joins all sheets of an Excel file to a base DataFrame.

    Args:
        file_path (str): Path to the Excel file.
        base_df (pd.DataFrame or None): The base DataFrame to join with. If None, starts with the first sheet.
        on (list or str): Column(s) to join on. Passed to pd.merge.
        **kwargs: Additional arguments for read_excel.

    Returns:
        pd.DataFrame: A single DataFrame after left joining all sheets.
    """
    # Load the Excel file and get all sheet names
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {file_path}. Error: {e}")

    # Initialize base_df if not provided
    if base_df is None:
        try:
            base_df = read_excel(file_path, sheet_name=sheet_names[0], **kwargs)
            sheet_names = sheet_names[1:]  # Remove the first sheet from processing
        except Exception as e:
            raise ValueError(f"Error processing sheet {sheet_names[0]}: {e}")

    # Iteratively join each sheet
    for sheet in sheet_names:
        try:
            # Read the current sheet
            sheet_df = read_excel(file_path, sheet_name=sheet, **kwargs)

            # Perform a left join with the base DataFrame
            base_df = base_df.merge(sheet_df, how="left", on=on)

        except Exception as e:
            print(f"Error processing sheet {sheet}: {e}")

    return base_df


def aggregate_vertical(
    df, index_columns=None, target_columns=None, header="Aggregations", agg_func="count"
):
    """
    Aggregates a DataFrame vertically across specified index_columns.
    Creates a hierarchical index: first level as column names (index_columns),
    second level as unique values from those columns, with aggregations for target_columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to aggregate.
    - index_columns (list): List of column names to aggregate.
    - target_columns (list or None): List of numerical column names to include in the aggregation.
    - header (str): Header for the resulting aggregation columns.
    - agg_func (str): Aggregation function to apply ('count', 'sum', 'mean').

    Returns:
    - pd.DataFrame: A DataFrame with a hierarchical index and aggregated columns.
    """
    if index_columns is None or not index_columns:
        raise ValueError("index_columns cannot be None or empty.")
    if agg_func not in ["count", "sum", "mean"]:
        raise ValueError("agg_func must be one of ['count', 'sum', 'mean'].")

    # Initialize an empty list to store results
    result = []

    for col in index_columns:
        if target_columns is None or not target_columns:
            # Aggregate the unique values in index_columns
            if agg_func == "count":
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, "Aggregation"]
            elif agg_func == "sum" or agg_func == "mean":
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"Column '{col}' must be numeric for {agg_func}.")
                counts = df[col].agg([agg_func]).reset_index()
                counts.columns = [col, "Aggregation"]

            counts["Column"] = col
            counts.set_index(["Column", col], inplace=True)
            counts.rename(columns={"Aggregation": header}, inplace=True)
        else:
            # Perform aggregation for each target column grouped by the index column
            agg_df = df.groupby(col)[target_columns].agg(agg_func).reset_index()
            agg_df["Column"] = col
            agg_df.set_index(["Column", col], inplace=True)
            counts = agg_df

        # Append to results
        result.append(counts)

    # Concatenate results for all index_columns
    summary_df = pd.concat(result)

    return summary_df