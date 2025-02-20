library(dplyr)
library(here)
library(poLCA)
library(readr)
source(file.path(Sys.getenv("PROJECT_ROOT"), "src", "helpers", "config.R"))
source(file.path(Sys.getenv("PROJECT_ROOT"), "src", "helpers", "utils.R"))



perform_lca <- function(data, n_classes) {
  # Perform Latent Class Analysis (LCA) and return model statistics.
  #
  # Args:
  #   data: A data frame containing numeric variables for LCA.
  #   n_classes: Number of latent classes to fit.
  #
  # Returns:
  #   A data frame with:
  #     - Predicted_Class: Predicted latent classes for each row.
  #     - AIC: Akaike Information Criterion.
  #     - BIC: Bayesian Information Criterion.
  #     - G_squared: Likelihood ratio/deviance statistic (G²).
  #     - X_squared: Chi-square goodness of fit statistic (X²).

  # Validate inputs
  if (!is.data.frame(data) || !all(sapply(data, is.numeric))) {
    stop("Data for LCA must be a numeric data frame.")
  }

  # Create LCA formula
  lca_formula <- as.formula(paste("cbind(", paste(names(data), collapse = ", "), ") ~ 1"))

  # Perform LCA
  lca_result <- poLCA::poLCA(lca_formula, data = data, nclass = n_classes, na.rm = TRUE)

  # Extract results
  data.frame(
    Predicted_Class = lca_result$predclass,
    AIC = lca_result$aic,
    BIC = lca_result$bic,
    Gsq = lca_result$Gsq,
    Chisq = lca_result$Chisq
  )
}


run_latent_class_analysis <- function(data, group_column, n_classes = 5, columns_to_lca = NULL) {
  # Perform latent class analysis (LCA) for grouped data.
  #
  # Args:
  #   data: A data frame with grouped data.
  #   group_column: The column to group by for LCA.
  #   n_classes: Number of latent classes to fit.
  #   columns_to_lca: A character vector of column names to use for LCA.
  # Returns:
  #   A data frame with "ID", LCA class, AIC, and BIC.

  # Validate inputs
  if (!"ID" %in% names(data)) stop("Data must contain an 'ID' column.")
  if (!group_column %in% names(data)) stop(paste("Column", group_column, "not found in data."))

  # If columns_to_lca is not specified, use all columns except "ID" and group_column
  if (is.null(columns_to_lca)) {
    columns_to_lca <- setdiff(names(data), c("ID", group_column))
  }

  # Validate columns_to_lca
  if (!all(columns_to_lca %in% names(data))) {
    invalid_cols <- columns_to_lca[!columns_to_lca %in% names(data)]
    stop("The following columns in 'columns_to_lca' do not exist in the data: ", paste(invalid_cols, collapse = ", "))
  }

  # Get unique group values
  group_values <- unique(data[[group_column]])
  final_results <- data.frame()

  # Loop through group values and perform LCA
  for (group_value in group_values) {
    # Filter rows for the current group
    filtered_rows <- data %>% filter(.data[[group_column]] == group_value)

    # Extract ID column and subset columns for LCA
    id_column <- filtered_rows$ID
    filtered_rows <- filtered_rows[, columns_to_lca, drop = FALSE]

    # Perform LCA
    lca_result <- perform_lca(filtered_rows, n_classes)

    # Sort classes by frequency
    class_frequencies <- table(lca_result$Predicted_Class)
    sorted_classes <- as.integer(names(sort(class_frequencies, decreasing = TRUE)))
    remapped_classes <- match(lca_result$Predicted_Class, sorted_classes)

    # Create a temporary data frame for the group
    temp_results <- data.frame(
      ID = id_column,
      Predicted_Class = remapped_classes,
      AIC = lca_result$AIC[1],
      BIC = lca_result$BIC[1],
      Gsq = lca_result$Gsq[1],
      Chisq = lca_result$Chisq[1]
    )

    # Append to final results
    final_results <- rbind(final_results, temp_results)
  }

  # Rename columns with the required naming convention
  colnames(final_results) <- c(
    "ID",
    paste0("LCA_", group_column, "_class"),
    paste0("LCA_", group_column, "_AIC"),
    paste0("LCA_", group_column, "_BIC"),
    paste0("LCA_", group_column, "_Gsq"),
    paste0("LCA_", group_column, "_Chisq")
  )

  return(final_results)
}

print(DATA_DIR)

# File paths
encoded_data_path <- file.path(DATA_DIR, "processed", "encoded_data.csv")
group_set_path <- file.path(DATA_DIR, "processed", "group_set.csv")

# Read data
encoded_data <- read_csv(encoded_data_path)
group_set <- read_csv(group_set_path)

# Combine feature prefixes
prefixes_to_lca <- unique(c(MOMENT_OF_SUICIDE_FEATURES, SOCIO_DEMOGRAPHIC_FEATURES))

# Select columns matching prefixes
columns_to_lca <- grep(paste0("^(", paste(prefixes_to_lca, collapse = "|"), ")"),
  names(encoded_data),
  value = TRUE
)

# Include "ID" column
selected_columns <- c(columns_to_lca, "ID")
encoded_data <- encoded_data[, selected_columns]

# Add missing columns from group_set to encoded_data
new_columns <- setdiff(names(group_set), names(encoded_data))
if (length(new_columns) > 0) {
  encoded_data <- left_join(
    encoded_data,
    group_set %>% dplyr::select(all_of(c("ID", new_columns))),
    by = "ID"
  )
} else {
  message("No new columns to join.")
}

# Modify values: 0 -> 1, 1 -> 2
encoded_data <- encoded_data %>%
  mutate(across(all_of(columns_to_lca), ~ if_else(. == 0, 1, if_else(. == 1, 2, .))))

# ================================================================================
# Execute LCA
# ================================================================================
group_columns <- c("Group_AF", "Group_AG", "Group_AGF")

for (group_column in group_columns) {
  # Define columns to exclude based on the group column
  excluded_columns <- switch(group_column,
    "Group_AF" = "Fatal",
    "Group_AG" = "Gender",
    "Group_AGF" = c("Fatal", "Gender"),
    NULL # Default case if no exclusions are needed
  )

  # Select columns to use in LCA
  selected_columns <- columns_to_lca[!columns_to_lca %in% excluded_columns]

  # Run latent class analysis
  lca_classes <- run_latent_class_analysis(
    data = encoded_data,
    group_column = group_column,
    n_classes = 5,
    columns_to_lca = selected_columns
  )

  # Save results
  output_path <- file.path(DATA_DIR, "processed", "lca_group_results.xlsx")
  write_excel(output_path, lca_classes, sheet_name = group_column, if_sheet_exists = "replace")
}
