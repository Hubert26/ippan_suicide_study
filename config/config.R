library(dotenv)
library(yaml)
library(fs)

# Load environment variables from the .env file
dotenv::load_dot_env()

# Get the PROJECT_ROOT path
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "/") # Default to "/" for Docker

if (!dir_exists(PROJECT_ROOT)) {
    stop(paste("PROJECT_ROOT is not a valid path:", PROJECT_ROOT))
}

# Load settings from `settings.yaml`
settings <- yaml::read_yaml(file.path(PROJECT_ROOT, "settings.yaml"))

# Assign paths
DATA_DIR <- file.path(PROJECT_ROOT, settings$paths$data)
RESULTS_DIR <- file.path(PROJECT_ROOT, settings$paths$results)
PLOTS_DIR <- file.path(PROJECT_ROOT, settings$paths$plots)
MOMENT_OF_SUICIDE_FEATURES <- settings$moment_of_suicide_features
SOCIO_DEMOGRAPHIC_FEATURES <- settings$socio_demographic_features

# Ensure directories exist before proceeding
dirs_to_create <- c(DATA_DIR, RESULTS_DIR, PLOTS_DIR)
for (dir in dirs_to_create) {
    if (!dir_exists(dir)) {
        dir_create(dir, recursive = TRUE)
    }
}

# Get the name of the current script safely
get_script_name <- function() {
    args <- commandArgs(trailingOnly = FALSE)
    script_name <- basename(tail(args, n = 1))
    return(script_name)
}

# Print configuration for debugging
print_config <- function() {
    cat("PROJECT_ROOT:", PROJECT_ROOT, "\n")
    cat("DATA_DIR:", DATA_DIR, "\n")
    cat("RESULTS_DIR:", RESULTS_DIR, "\n")
    cat("PLOTS_DIR:", PLOTS_DIR, "\n")
    cat("MOMENT_OF_SUICIDE_FEATURES:", MOMENT_OF_SUICIDE_FEATURES, "\n")
    cat("SOCIO_DEMOGRAPHIC_FEATURES:", SOCIO_DEMOGRAPHIC_FEATURES, "\n")
}

# print_config()
