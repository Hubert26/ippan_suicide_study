from config.config import (
    COLUMN_MAPPINGS_2013_2022,
    COLUMN_MAPPINGS_2023,
    DATA_DIR,
    VALUE_MAPPINGS_2013_2022,
    VALUE_MAPPINGS_2023,
)
from logs.logger import logger
from src.data_processing.step_01_data_mapping import run_data_mapping
from src.data_processing.step_02_data_imputation import run_data_imputation
from src.data_processing.step_03_data_encoding import run_data_encoding
from src.data_processing.step_03_group_mapping import run_group_mapping
from src.utils.utils import read_csv, read_excel


def main():
    """
    Main function to control the data processing pipeline.
    It executes the data mapping and imputation processes without saving files.
    """
    logger.info("Starting data processing pipeline...")

    # Load raw data for 2023
    excel_file_path = DATA_DIR / "raw" / "Samobojstwa_2023.xlsx"
    logger.info(f"Loading raw data from: {excel_file_path}")
    df_raw_2023 = read_excel(excel_file_path)
    logger.debug(f"Loaded data shape (2023): {df_raw_2023.shape}")

    # Load raw data for 2013-2022
    csv_file_path = DATA_DIR / "raw" / "final_samobojstwa_2013_2022.csv"
    logger.info(f"Loading raw data from: {csv_file_path}")
    df_raw_2013_2022 = read_csv(csv_file_path, delimiter=",", low_memory=False)
    logger.debug(f"Loaded data shape (2013-2022): {df_raw_2013_2022.shape}")

    # Step 01: Data Mapping
    logger.info("Step 01: Running data mapping...")
    df_mapped = run_data_mapping(
        df_raw_2023,
        df_raw_2013_2022,
        column_mappings_2023=COLUMN_MAPPINGS_2023,
        value_mappings_2023=VALUE_MAPPINGS_2023,
        column_mappings_2013_2022=COLUMN_MAPPINGS_2013_2022,
        value_mappings_2013_2022=VALUE_MAPPINGS_2013_2022,
    )
    logger.info("Data mapping completed.")
    logger.debug(f"Mapped data shape: {df_mapped.shape}")
    logger.debug(f"Mapped data columns: {df_mapped.columns.tolist()}")

    # Step 02: Data Imputation
    logger.info("Step 02: Running data imputation...")
    df_imputed = run_data_imputation(df_mapped)
    logger.info("Data imputation completed.")
    logger.debug(f"Imputed data shape: {df_imputed.shape}")
    logger.debug(f"Imputed data columns: {df_imputed.columns.tolist()}")

    # Step 03: Group Mapping
    logger.info("Step 03: Running group mapping...")
    group_set = run_group_mapping(df_imputed)
    logger.info("Group mapping completed.")
    logger.debug(f"Group set: {group_set}")

    # Step 03: Data Encoding
    logger.info("Step 03: Running data encoding...")
    df_encoded = run_data_encoding(df_imputed)
    logger.info("Data encoding completed.")
    logger.debug(f"Encoded data shape: {df_encoded.shape}")
    logger.debug(f"Encoded data columns: {df_encoded.columns.tolist()}")

    logger.info("Data processing pipeline completed successfully.")


if __name__ == "__main__":
    main()
