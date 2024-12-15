library(dplyr)
library(here)
#install.packages("poLCA")
library(poLCA)

#================================================================================
#FOLDER SETTINGS
#================================================================================
# Pobranie ścieżki do bieżącego pliku
current_path <- normalizePath(here())
print(paste("Aktualna ścieżka:", current_path))

# Cofnięcie się dwa poziomy w górę, aby dotrzeć do folderu 'ippan_suicide_study'
parent_folder <- dirname(dirname(current_path))  # Cofnięcie się dwa razy

# Przejście do folderu 'results/poLCA' w folderze 'ippan_suicide_study'
results_folder <- file.path(parent_folder, "results", "poLCA")

print(paste("Ścieżka do folderu 'results/poLCA':", results_folder))



# Funkcja do tworzenia folderu, jeśli nie istnieje
create_folder_if_not_exists <- function(folder_path) {
  
  # Sprawdzenie, czy folder istnieje
  if (!dir.exists(folder_path)) {
    print(paste("Folder does not exist. Creating folder:", folder_path))
    
    # Tworzenie folderu, jeśli nie istnieje
    dir.create(folder_path, recursive = TRUE)
    
    # Sprawdzenie, czy folder został poprawnie utworzony
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
#READ DATA
#================================================================================

# Define the path to the CSV file
data_path <- normalizePath(file.path(here(), "..", "..", "data", "encoded", "encoded_data.csv"))

# Read the CSV file into a data frame
encoded_data <- read.csv(data_path, header = TRUE, sep = ",", stringsAsFactors = FALSE)

encoded_data[] <- lapply(encoded_data, function(x) {
  if (all(x %in% c("True", "False"))) {  # Check if the column contains only "True" or "False"
    return(as.integer(x == "True"))  # Convert "True" to 1, "False" to 0
  } else {
    return(x)  # Leave other columns unchanged
  }
})

selected_data <- encoded_data


# Lista prefiksów kolumn do wybrania
selected_prefixes <- c('Income', 'Method', 'Education',
                       'WorkInfo', 'Substance', 'Place', 'Marital', 'Context', 'AbuseInfo', 'Gender', 'Fatal')

# Wybieranie kolumn, których nazwy zaczynają się od dowolnego z prefiksów
selected_columns <- grep(paste0("^(", paste(selected_prefixes, collapse = "|"), ")"), 
                         names(encoded_data), value = TRUE)

# Tworzenie nowego zbioru danych zawierającego tylko wybrane kolumny
selected_data <- selected_data[, selected_columns]

# Recode the data to start from 1
selected_data <- selected_data + 1

# Add ID column
selected_data$ID <- encoded_data$ID

#================================================================================
#
#================================================================================




#================================================================================
#poLCA for Group
#================================================================================
data <- selected_data

# Choose Group column
group_column <- "Group_AGF"
predicted_class_column <- paste("Predicted_Class", group_column, sep = "_")



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

group_values <- unique(data[[group_column]])

lca_classes <- data.frame(ID = encoded_data$ID)
lca_classes[[predicted_class_column]] <- NA



for (group_value in group_values) {
  
  filtered_rows <- data %>% filter(.data[[group_column]] == group_value)
  
  # Pobranie kolumny ID do późniejszego dopasowania
  id_column <- filtered_rows$ID
  
  # Usunięcie kolumn związanych z Group i ID
  filtered_rows <- filtered_rows[, !names(filtered_rows) %in% c(group_column, "ID"), drop = FALSE]
  
  # Stworzenie formuły do LCA
  lca_formula <- as.formula(paste("cbind(", paste(colnames(filtered_rows), collapse = ", "), ") ~ 1"))
  
  # Wykonanie analizy ukrytych klas
  lca_result <- poLCA(lca_formula, data = filtered_rows, nclass = 5, na.rm = TRUE)
  
  # Przypisanie klasy do odpowiednich wierszy w lca_classes na podstawie ID
  lca_classes[[predicted_class_column]][lca_classes$ID %in% id_column] <- lca_result$predclass
  
  # Generowanie dynamicznej nazwy pliku
  file_name <- paste0(group_value, ".txt")
  
  folder_name <- file.path(results_folder, group_column)
  create_folder_if_not_exists(folder_name)
  
  # Tworzenie pełnej ścieżki do pliku w folderze 'results'
  file_path <- file.path(folder_name, file_name)
  
  # Zapis wyników do pliku
  sink(file_path)  # Przekierowanie wyjścia do pliku
  
  # Zapis nagłówka do pliku
  cat("#================================================================================\n")
  cat(paste("#poLCA for", group_value, "\n"))
  cat("#================================================================================\n")
  
  # Zapis wyników LCA do pliku
  print(lca_result)
  
  # Zapis liczności klas do pliku
  cat("\n# Class counts\n")
  print(table(lca_result$predclass))
  
  # Zakończenie przekierowania do pliku
  sink()
  
  # Opcjonalne: potwierdzenie, że plik został zapisany
  print(paste("Saved in:", file_name))

}
file_name <- paste0(group_column, ".csv")
output_data_path <- file.path(results_folder, file_name)
write.csv(lca_classes, output_data_path, row.names = FALSE)
