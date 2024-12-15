# latent_class_analysis.R
# This script performs latent class analysis (LCA) using the poLCA package.
# It reads encoded data, selects relevant columns, and executes LCA for specified groups.
# Results are saved to designated output files.

# Load necessary libraries
library(dplyr)
library(here)
library(poLCA)  # Ensure poLCA is installed
library(dotenv)  # Load environment variables from .env file

# Load environment variables
load_dot_env()

# Get the workspace path from the .env file
workspace_path <- Sys.getenv("WORKSPACE_PATH")
absolute_workspace_path <- normalizePath(workspace_path)

# Get the current workspace path
current_workspace_path <- normalizePath(here())
print(paste("Current workspace path:", current_workspace_path))

# Compare the paths
if (absolute_workspace_path != current_workspace_path) {
  warning(paste("Workspace paths do not match! .env path:", absolute_workspace_path, "Current path:", current_workspace_path))
} else {
  print("Workspace paths match.")
}

#================================================================================
# FOLDER SETTINGS
#================================================================================

# Define the path to the input data CSV file
data_path <- normalizePath(file.path(absolute_workspace_path, "data", "encoded", "encoded_final_set.csv"))

# Navigate to 'results/poLCA' folder in 'ippan_suicide_study'
results_folder <- normalizePath(file.path(absolute_workspace_path, "results", "poLCA"))

# Function to create folder if it does not exist
create_folder_if_not_exists <- function(folder_path) {
  if (!dir.exists(folder_path)) {
    print(paste("Folder does not exist. Creating folder:", folder_path))
    dir.create(folder_path, recursive = TRUE)
    if (dir.exists(folder_path)) {
      print(paste("Folder", folder_path, "was successfully created."))
    } else {
      print(paste("Failed to create folder", folder_path))
    }
  } else {
    print(paste("Folder", folder_path, "already exists."))
  }
}

create_folder_if_not_exists(results_folder)

#================================================================================
# READ DATA
#================================================================================

# Function to read data
read_data <- function(path) {
  data <- read.csv(path, header = TRUE, sep = ",", stringsAsFactors = FALSE)
  data[] <- lapply(data, function(x) {
    if (all(x %in% c("True", "False"))) {
      return(as.integer(x == "True"))  # Convert "True" to 1, "False" to 0
    } else {
      return(x)  # Leave other columns unchanged
    }
  })
  return(data)
}

# Check if the file exists
if (!file.exists(data_path)) {
    stop(paste("File does not exist:", data_path))
}

# Read the data
encoded_data <- read_data(data_path)

# List of prefixes for selected columns
selected_prefixes <- c('Income', 'Method', 'Education',
                       'WorkInfo', 'Substance', 'Place', 'Marital', 'Context', 'AbuseInfo', 'Gender', 'Fatal')

# Selecting columns that start with any of the prefixes
selected_columns <- grep(paste0("^(", paste(selected_prefixes, collapse = "|"), ")"), 
                         names(encoded_data), value = TRUE)

# Creating a new data set containing only selected columns
selected_data <- encoded_data[, selected_columns]

# Recode the data to start from 1
selected_data <- selected_data + 1

# Add ID column
selected_data$ID <- encoded_data$ID

#================================================================================
# poLCA for Group
#================================================================================

data <- selected_data

# Choose Group column
group_column <- "Group_AGF"  # Replace with the desired group column name, e.g., "Group_AF" 

if (group_column %in% names(encoded_data)) {
  data[[group_column]] <- encoded_data[[group_column]]
} else {
  warning(paste("Column", group_column, "doesn't exist in encoded_data."))
}

if (group_column == "Group_AF") {
  data <- data[, !names(data) %in% "Fatal"]
} else if (group_column == "Group_AG") {
  data <- data[, !names(data) %in% "Gender"]
} else if (group_column == "Group_AGF") {
  data <- data[, !names(data) %in% c("Fatal", "Gender")]
}

# Function to perform LCA
perform_lca <- function(data, group_column, n_classes = 5) {
  group_values <- unique(data[[group_column]])
  lca_classes <- data.frame(ID = data$ID)
  lca_classes[[paste("Predicted_Class", group_column, sep = "_")]] <- NA
  
  for (group_value in group_values) {
    filtered_rows <- data %>% filter(.data[[group_column]] == group_value)
    id_column <- filtered_rows$ID
    filtered_rows <- filtered_rows[, !names(filtered_rows) %in% c(group_column, "ID"), drop = FALSE]
    lca_formula <- as.formula(paste("cbind(", paste(colnames(filtered_rows), collapse = ", "), ") ~ 1"))
    
    # Execute latent class analysis
    lca_result <- poLCA(lca_formula, data = filtered_rows, nclass = n_classes, na.rm = TRUE)
    lca_classes[[paste("Predicted_Class", group_column, sep = "_")]][lca_classes$ID %in% id_column] <- lca_result$predclass
    
    # Save results to file
    file_name <- paste0(group_value, ".txt")
    folder_name <- file.path(results_folder, group_column)
    create_folder_if_not_exists(folder_name)
    file_path <- file.path(folder_name, file_name)
    sink(file_path)
    cat("#================================================================================\n")
    cat(paste("# poLCA for", group_value, "\n"))
    cat("#================================================================================\n")
    print(lca_result)
    cat("\n# Class counts\n")
    print(table(lca_result$predclass))
    sink()
    print(paste("Saved in:", file_name))
  }
  
  return(lca_classes)
}

# Perform LCA
lca_classes <- perform_lca(data, group_column)

# Save final results to CSV
file_name <- paste0(group_column, ".csv")
output_data_path <- file.path(results_folder, file_name)
write.csv(lca_classes, output_data_path, row.names = FALSE)
print(paste("LCA classes saved to:", output_data_path))