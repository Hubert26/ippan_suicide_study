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

data <- encoded_data


# Lista prefiksów kolumn do wybrania
selected_prefixes <- c('Fatal', 'Income', 'Method', 'Education',
                       'WorkInfo', 'Substance', 'Place', 'Marital', 'Context', 'Group')

# Wybieranie kolumn, których nazwy zaczynają się od dowolnego z prefiksów
selected_columns <- grep(paste0("^(", paste(selected_prefixes, collapse = "|"), ")"), 
                         names(data), value = TRUE)

# Tworzenie nowego zbioru danych zawierającego tylko wybrane kolumny
data <- data[, selected_columns]

# Recode the data to start from 1
data <- data + 1

# Add ID column
data$ID <- encoded_data$ID
#================================================================================
#
#================================================================================




#================================================================================
#poLCA for GroupAge and Gender => Group
#================================================================================

group <- grep(paste0("^(", paste(c("Group_"), collapse = "|"), ")"), 
                       names(data), value = TRUE)

group_columns <- group

lca_classes <- data.frame(ID = encoded_data$ID, Predicted_Class = NA)

# Sprawdzenie, czy kolumny Group
if (!all(group_columns %in% names(data))) {
  print("Nie ma tych kolumn")
} else {
  print("Kolumny istnieją, kontynuujemy przetwarzanie.")
}



# Pętla po kolumnach z Group
for (group_column in group_columns) {
  
  # Filtrowanie wierszy, gdzie w danej kolumnie Group wartość wynosi 2
  filtered_rows <- data %>% filter(.data[[group_column]] == 2)
  
  # Pobranie kolumny ID do późniejszego dopasowania
  id_column <- filtered_rows$ID
  
  # Usunięcie kolumn związanych z Group i ID
  filtered_rows <- filtered_rows[, !names(filtered_rows) %in% c(group_columns, "ID"), drop = FALSE]
  
  # Stworzenie formuły do LCA
  lca_formula <- as.formula(paste("cbind(", paste(colnames(filtered_rows), collapse = ", "), ") ~ 1"))
  
  # Wykonanie analizy ukrytych klas
  lca_result <- poLCA(lca_formula, data = filtered_rows, nclass = 5, na.rm = TRUE)
  
  # Przypisanie klasy do odpowiednich wierszy w lca_classes na podstawie ID
  lca_classes$Predicted_Class[lca_classes$ID %in% id_column] <- lca_result$predclass
  
  # Generowanie dynamicznej nazwy pliku
  file_name <- paste0("LCA_result_", group_column, ".txt")
  
  folder_name <- file.path(results_folder, "Group")
  create_folder_if_not_exists(folder_name)
  
  # Tworzenie pełnej ścieżki do pliku w folderze 'results'
  file_path <- file.path(folder_name, file_name)
  
  # Zapis wyników do pliku
  sink(file_path)  # Przekierowanie wyjścia do pliku
  
  # Zapis nagłówka do pliku
  cat("#================================================================================\n")
  cat(paste("#poLCA for", group_column, "\n"))
  cat("#================================================================================\n")
  
  # Zapis wyników LCA do pliku
  print(lca_result)
  
  # Zapis liczności klas do pliku
  cat("\n# Class counts\n")
  print(table(lca_result$predclass))
  
  # Zakończenie przekierowania do pliku
  sink()
  
  # Opcjonalne: potwierdzenie, że plik został zapisany
  print(paste("Wyniki zapisano do pliku:", file_name))

}

output_data_path <- file.path(results_folder, "lca_classes.csv")
write.csv(lca_classes, output_data_path, row.names = FALSE)





#================================================================================
#poLCA for Fatal, GroupAge and Gender
#================================================================================

group <- grep(paste0("^(", paste(c("Group_"), collapse = "|"), ")"), 
              names(data), value = TRUE)

group_columns <- group

fatal_column <- c("Fatal")

# Pobierz unikalne wartości z kolumny Gender
fatal_values <- unique(data[, fatal_column])

# Sprawdzenie, czy kolumny Group i Fatal nie istnieją w danych
if (!all(group_columns %in% names(data)) | !all(fatal_column %in% names(data))){
  print("Nie ma tych kolumn")
} else {
  print("Kolumny istnieją, kontynuujemy przetwarzanie.")
}



# Pętla po kolumnach z GroupAge
for (group_column in group_columns) {

  # Pętla po unikalnych wartościach z kolumny Fatal
  for (fatal_value in fatal_values) {
    
    # Filtrowanie wierszy, gdzie w danej kolumnie GroupAge wartość wynosi 2 i w kolumnie Gender jest aktualna wartość gender_value
    filtered_rows <- data[data[, group_column] == 2 & data[, fatal_column] == fatal_value, ]
    
    # Usunięcie kolumn związanych z GroupAge, Gender, Fatal
    filtered_rows <- filtered_rows[, setdiff(names(filtered_rows), c(group, fatal_column)), drop = FALSE]
    
    # Perform Latent Class Analysis
    lca_formula <- as.formula(paste("cbind(", paste(colnames(filtered_rows), collapse = ", "), ") ~ 1"))
    lca_result <- poLCA(lca_formula, data = filtered_rows, nclass = 5, na.rm = TRUE)
    
    # Generowanie dynamicznej nazwy pliku
    file_name <- paste0("LCA_result_", group_column, fatal_value-1, ".txt")
    
    folder_name <- file.path(results_folder, "GroupFatal")
    create_folder_if_not_exists(folder_name)
    
    # Tworzenie pełnej ścieżki do pliku w folderze 'results'
    file_path <- file.path(folder_name, file_name)
    
    # Zapis wyników do pliku
    sink(file_path)  # Przekierowanie wyjścia do pliku
    
    # Zapis nagłówka do pliku
    cat("#================================================================================\n")
    cat(paste("#poLCA for", group_column, fatal_value-1, "\n"))
    cat("#================================================================================\n")
    
    # Zapis wyników LCA do pliku
    print(lca_result)
    
    # Zapis liczności klas do pliku
    cat("\n# Class counts\n")
    print(table(lca_result$predclass))
    
#    cat("\n# Posterior probabilities for each class (per observation):\n")
#    print(lca_result$posterior)
    
    # Zakończenie przekierowania do pliku
    sink()
    
    # Opcjonalne: potwierdzenie, że plik został zapisany
    print(paste("Wyniki zapisano do pliku:", file_name))
  }

}



#================================================================================
#poLCA for Fatal
#================================================================================
group <- grep(paste0("^(", paste(c("Group_"), collapse = "|"), ")"), 
              names(data), value = TRUE)

fatal_column <- c("Fatal")

# Pobierz unikalne wartości z kolumny Fatal
fatal_values <- unique(data[, fatal_column])

# Inicjalizacja kolumny Predicted_Class w encoded_data jako NA
encoded_data$Predicted_Class <- NA

# Pętla po unikalnych wartościach z kolumny Fatal
for (fatal_value in fatal_values) {
  
  # Filtrowanie wierszy, gdzie w danej kolumnie Fatal wartość wynosi aktualny fatal_value
  filtered_rows <- data[data[, fatal_column] == fatal_value, ]
  
  # Usunięcie kolumn związanych z Fatal i Group
  filtered_rows <- filtered_rows[, setdiff(names(filtered_rows), c(group, fatal_column)), drop = FALSE]
  
  # Sprawdź, czy są jakieś dane do analizy
  if (nrow(filtered_rows) > 0) {
    # Perform Latent Class Analysis
    lca_formula <- as.formula(paste("cbind(", paste(colnames(filtered_rows), collapse = ", "), ") ~ 1"))
    lca_result <- poLCA(lca_formula, data = filtered_rows, nclass = 5, na.rm = TRUE)
    
    # Zapisanie klasy do danych
    # Dodaj kolumnę predclass do oryginalnych danych (filtrując odpowiednie wiersze)
    encoded_data$Predicted_Class[encoded_data[, fatal_column] == fatal_value] <- lca_result$predclass
    
    # Generowanie dynamicznej nazwy pliku
    file_name <- paste0("LCA_result_Fatal_", fatal_value - 1, ".txt")
    
    folder_name <- file.path(results_folder, "Fatal")
    create_folder_if_not_exists(folder_name)
    
    # Tworzenie pełnej ścieżki do pliku w folderze 'results'
    file_path <- file.path(folder_name, file_name)
    
    # Zapis wyników do pliku
    sink(file_path)  # Przekierowanie wyjścia do pliku
    
    # Zapis nagłówka do pliku
    cat("#================================================================================\n")
    cat(paste("#poLCA for Fatal", fatal_value - 1, "\n"))
    cat("#================================================================================\n")
    
    # Zapis wyników LCA do pliku
    print(lca_result)
    
    # Zapis liczności klas do pliku
    cat("\n# Class counts\n")
    print(table(lca_result$predclass))
    
    # Zakończenie przekierowania do pliku
    sink()
    
    # Opcjonalne: potwierdzenie, że plik został zapisany
    print(paste("Wyniki zapisano do pliku:", file_name))
  } else {
    warning(paste("Brak danych do analizy dla Fatal:", fatal_value))
  }
}

# Opcjonalnie: zapisz zaktualizowane dane do nowego pliku CSV
output_data_path <- file.path(results_folder, "encoded_data_LCU.csv")
write.csv(encoded_data, output_data_path, row.names = FALSE)
