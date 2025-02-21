# --- Konfiguracja styler ---
options(styler.colored_print.vertical = FALSE)
options(styler.strict = TRUE)
options(styler.line_length = 88)

# --- Konfiguracja Language Server dla VSCode ---
# Automatyczna instalacja i konfiguracja pakietu 'languageserver'
if (!requireNamespace("languageserver", quietly = TRUE)) {
  install.packages("languageserver")
}

# Automatyczne ładowanie pakietu 'languageserver'
if (interactive()) {
  library(languageserver)
}

# Optymalizacja wydajności Language Servera
options(
  languageserver.formatting_style = "styler",  # Używanie styler do formatowania
  languageserver.max_workspace_size = 1e+06,   # Maksymalny rozmiar workspace
  languageserver.diagnostics = TRUE,           # Diagnostyka błędów
  languageserver.completion = TRUE,             # Autouzupełnianie
  languageserver.signature = TRUE,              # Podpowiedzi argumentów funkcji
  languageserver.hover = TRUE,                  # Podpowiedzi dokumentacyjne
  languageserver.snippet_support = TRUE,         # Wsparcie dla snippetów
  languageserver.on_attach = function(client, workspace) {
    cat("Language Server Attached\n")
  }
)

# --- Poprawki dla VSCode ---
# Poprawki dla zagnieżdżonych wywołań
options(
  r.langserver.validate_on_save = TRUE,
  r.langserver.validate_on_change = TRUE
)

# --- Ustawienia dla lintr ---
# Ignorowanie niektórych linterów, np. dla nazw zmiennych
if (requireNamespace("lintr", quietly = TRUE)) {
  lintr::use_lintr(type = "tidyverse")
}
