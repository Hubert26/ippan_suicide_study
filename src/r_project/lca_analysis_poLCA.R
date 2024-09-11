library(dplyr)
library(here)
library(poLCA)

# Define the path to the CSV file
data_path <- normalizePath(file.path(here(), "..", "..", "data", "imputed", "imputed_data.csv"))

# Read the CSV file into a data frame
data_raw <- read.csv(data_path, header = TRUE, sep = ",", stringsAsFactors = FALSE)

# Define the columns to include in the analysis
factor_columns <- c("Education",
                    "Age2",
                    "WorkInfo",
                    "AbuseInfo",
                    "Gender",                                  
                    "Income",                
                    "Substance",
                    "Marital",           
                    "Method",                
                    "Place",
                    "Fatal")

context_columns <- c("Context_Finances",      
                     "Context_HeartBreak",
                     "Context_Disability",
                     "Context_SchoolWork",    
                     "Context_CloseDeath",
                     "Context_HealthLoss",
                     "Context_MentalHealth",  
                     "Context_Crime",
                     "Context_Other",
                     "Context_FamilyConflict")

# Combine factor and context columns
selected_columns <- c(factor_columns, context_columns)

# Select relevant columns from the data frame
data_selected <- dplyr::select(data_raw, dplyr::all_of(selected_columns))

# Function to add an underscore to column names
add_underscore <- function(df, columns) {
  # Check if all columns are present in the data frame
  missing_cols <- setdiff(columns, colnames(df))
  if (length(missing_cols) > 0) {
    stop("Missing columns: ", paste(missing_cols, collapse = ", "))
  }
  
  # Add an underscore to column names
  new_colnames <- paste0(colnames(df)[colnames(df) %in% columns], "_")
  colnames(df)[colnames(df) %in% columns] <- new_colnames
  
  return(df)
}

# Define columns to which underscore will be added, excluding "Fatal"
underscore_columns <- setdiff(factor_columns, "Fatal")

# Apply underscore to selected columns
data_selected <- add_underscore(data_selected, underscore_columns)

# Perform one-hot encoding
encoded_data <- model.matrix(~ . - 1, data = data_selected)

# Convert the matrix to a data frame if necessary
if (is.matrix(encoded_data) || is.array(encoded_data)) {
  encoded_data <- as.data.frame(encoded_data)
}

# Replace dashes with underscores in column names
colnames(encoded_data) <- gsub("-", "_", colnames(encoded_data))
# Rename the column
colnames(encoded_data) <- gsub("Age2_65+", "Age2_65", colnames(encoded_data))



# Perform Latent Class Analysis
lca_formula <- as.formula(paste("~", paste(colnames(encoded_data), collapse = " + ")))
n_classes <- 3
lca_result <- poLCA(lca_formula, data = encoded_data, nclass = n_classes, na.rm = TRUE)

# Print the result of the Latent Class Analysis
print(lca_result)
