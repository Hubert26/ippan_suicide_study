library("dotenv")
library(poLCA)
library(dplyr)
library(writexl)
library(readxl)
library(purrr)


load_environment_variables <- function(required_vars = NULL, env_file_path = ".env") {
  # Load all environment variables from a .env file and ensure required variables are present.
  #
  # Args:
  #   required_vars: A character vector of required environment variable names. Defaults to NULL.
  #   env_file_path: Path to the .env file. Defaults to ".env".
  #
  # Returns:
  #   A named character vector of all environment variables.
  #
  # Raises:
  #   Error if any of the required variables are missing or empty.

  # Load the .env file if it exists
  if (file.exists(env_file_path)) {
    dotenv::load_dot_env(env_file_path)
  } else {
    stop(paste(".env file not found at the specified path:", env_file_path))
  }

  # Fetch all environment variables
  env_vars <- Sys.getenv()

  # Check for required variables
  if (!is.null(required_vars)) {
    missing_vars <- required_vars[!required_vars %in% names(env_vars) | env_vars[required_vars] == ""]
    if (length(missing_vars) > 0) {
      stop(paste("Required environment variables are missing or empty:", paste(missing_vars, collapse = ", ")))
    }
  }

  # Return all environment variables as a named character vector
  return(env_vars)
}

split_string <- function(input_string, delimiter = ",", strip_whitespace = TRUE) {
  # Split a string by a specified delimiter and optionally strip whitespace from each element.
  # Args:
  #   input_string: A string to be split.
  #   delimiter: A character used as the delimiter for splitting. Defaults to ','.
  #   strip_whitespace: A logical value indicating whether to strip whitespace. Defaults to TRUE.
  # Returns:
  #   A character vector of split strings.

  elements <- unlist(strsplit(input_string, delimiter, fixed = TRUE))

  if (strip_whitespace) {
    elements <- trimws(elements)
  }

  # Remove names if they exist
  names(elements) <- NULL

  return(elements)
}

check_file_exists <- function(file_path, error_message = NULL) {
  # Check if a file exists at the specified path.
  #
  # Args:
  #   file_path: A character string representing the path of the file to check.
  #   error_message: Optional. A custom error message to display if the file does not exist.
  #
  # Returns:
  #   None. Stops execution with an error if the file does not exist.
  #
  # Raises:
  #   Error if the file does not exist.

  if (!file.exists(file_path)) {
    stop(error_message %||% paste("File does not exist:", file_path))
  }
}

check_folder_exists <- function(folder_path, error_message = NULL) {
  # Check if a folder exists at the specified path.
  #
  # Args:
  #   folder_path: A character string representing the path of the folder to check.
  #   error_message: Optional. A custom error message to display if the folder does not exist.
  #
  # Returns:
  #   None. Stops execution with an error if the folder does not exist.
  #
  # Raises:
  #   Error if the folder does not exist.

  if (!dir.exists(folder_path)) {
    stop(error_message %||% paste("Folder does not exist:", folder_path))
  }
}

create_folder <- function(folder_path) {
  # Create a folder if it does not exist.
  #
  # Args:
  #   folder_path: A character string representing the path of the folder to create.
  #
  # Returns:
  #   None. Prints messages about the folder's existence or creation status.
  #
  # Raises:
  #   Error if the folder cannot be created.

  # Check if the folder exists
  tryCatch(
    {
      check_folder_exists(folder_path)
      print(paste("Folder", folder_path, "already exists."))
    },
    error = function(e) {
      # If the folder does not exist, create it
      print(paste("Folder does not exist. Creating folder:", folder_path))
      dir.create(folder_path, recursive = TRUE)

      if (dir.exists(folder_path)) {
        print(paste("Folder", folder_path, "was successfully created."))
      } else {
        stop(paste("Failed to create folder:", folder_path))
      }
    }
  )
}

read_csv <- function(file_path, ...) {
  # Read a CSV file and return its content as a data frame.
  #
  # Args:
  #   file_path: A character string representing the path to the CSV file.
  #   ...: Additional arguments passed to the read.csv function, such as sep, header, etc.
  #
  # Returns:
  #   A data frame containing the contents of the CSV file.
  #
  # Raises:
  #   Error if the file does not exist or cannot be read.

  # Check if the file exists
  check_file_exists(file_path, paste("CSV file not found:", file_path))

  # Try reading the CSV file with additional arguments
  tryCatch(
    {
      data <- read.csv(file_path, stringsAsFactors = FALSE, ...)
      return(data)
    },
    error = function(e) {
      stop(paste("Failed to read CSV file:", file_path, "\nError:", e$message))
    }
  )
}

write_csv <- function(data, file_path, ...) {
  # Write a data frame to a CSV file with customizable options.
  #
  # Args:
  #   data: A data frame to be written to the CSV file.
  #   file_path: A character string representing the path to save the CSV file.
  #   ...: Additional arguments passed to the write.csv function, such as row.names, sep, etc.
  #
  # Returns:
  #   None. Writes the data to the specified file.
  #
  # Raises:
  #   Error if the file cannot be written.

  tryCatch(
    {
      write.csv(data, file = file_path, ...)
      print(paste("CSV file successfully written to:", file_path))
    },
    error = function(e) {
      stop(paste("Failed to write CSV file:", file_path, "\nError:", e$message))
    }
  )
}

write_excel <- function(file_path, data, ...) {
  # Validate inputs
  if (!inherits(data, "data.frame")) {
    stop("The 'data' argument must be a data.frame or tibble.")
  }

  # Normalize and validate file path
  file_path <- normalizePath(file_path, winslash = "/", mustWork = FALSE)
  print(paste("Normalized file path:", file_path))

  if (!grepl("\\.xlsx$", file_path, ignore.case = TRUE)) {
    stop("The file path must end with '.xlsx'.")
  }

  # Check if the directory exists
  dir_path <- dirname(file_path)
  if (!dir.exists(dir_path)) {
    stop(paste("The directory does not exist:", dir_path))
  }

  # Extract additional arguments
  args <- list(...)
  sheet_name <- if (!is.null(args$sheet_name)) args$sheet_name else "Sheet1"
  if_sheet_exists <- if (!is.null(args$if_sheet_exists)) args$if_sheet_exists else "replace"
  print(paste("Sheet name:", sheet_name))
  print(paste("If sheet exists:", if_sheet_exists))

  # Check if file exists
  file_exists <- file.exists(file_path)
  print(paste("File exists:", file_exists))

  wb_data <- list()

  if (file_exists) {
    # Load existing workbook
    tryCatch(
      {
        sheet_names <- readxl::excel_sheets(file_path)
        wb_data <- map(setNames(sheet_names, sheet_names), ~ readxl::read_excel(file_path, sheet = .x))
      },
      error = function(e) {
        stop(paste("Failed to read existing Excel file. Error:", e$message))
      }
    )
  }

  # Handle sheet existence
  if (sheet_name %in% names(wb_data)) {
    if (if_sheet_exists == "replace") {
      wb_data[[sheet_name]] <- data
    } else if (if_sheet_exists == "overlay") {
      wb_data[[sheet_name]] <- rbind(wb_data[[sheet_name]], data)
    } else {
      stop("Invalid 'if_sheet_exists' value. Use 'replace' or 'overlay'.")
    }
  } else {
    wb_data[[sheet_name]] <- data
  }

  # Write the data to the Excel file
  tryCatch(
    {
      writexl::write_xlsx(wb_data, path = file_path)
      message(paste("Data successfully written to:", file_path, "in sheet:", sheet_name))
    },
    error = function(e) {
      stop(paste("Failed to save workbook at:", file_path, "\nError:", e$message))
    }
  )
}

read_excel <- function(file_path, ...) {
  # Validate the file path
  file_path <- normalizePath(file_path, winslash = "/", mustWork = FALSE)
  if (!grepl("\\.xlsx$", file_path, ignore.case = TRUE)) {
    stop("The file path must end with '.xlsx'.")
  }

  # Check if the file exists
  if (!file.exists(file_path)) {
    stop(paste("The file does not exist:", file_path))
  }

  # Load the workbook
  wb <- tryCatch(
    wb_load(file_path),
    error = function(e) {
      stop(paste("Failed to load the Excel file:", file_path, "\nError:", e$message))
    }
  )

  # Retrieve sheet names
  sheet_names <- wb$sheet_names
  print(paste("Sheet names in the workbook:", toString(sheet_names)))

  # Extract additional arguments
  args <- list(...)
  sheet_name <- if (!is.null(args$sheet_name)) args$sheet_name else sheet_names[1] # Default to the first sheet
  if (!sheet_name %in% sheet_names) {
    stop(paste("The specified sheet does not exist:", sheet_name))
  }

  # Read the data from the specified sheet
  data <- tryCatch(
    wb_to_df(wb, sheet = sheet_name),
    error = function(e) {
      stop(paste("Failed to read the sheet:", sheet_name, "\nError:", e$message))
    }
  )

  # Return the data
  return(data)
}

print( "test" )
library(styler)
styler::style_file("utils.R")
