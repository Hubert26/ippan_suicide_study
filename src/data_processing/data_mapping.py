"""Data mapping module for standardizing and transforming suicide study dataset.
Handles data from both 2023 and 2013-2022 periods."""

import sys
from pathlib import Path
from dotenv import dotenv_values
import pandas as pd
from typing import List, Dict, Optional

# Load environment variables from the .env file
env_vars = dotenv_values()  # Load variables from the .env file

# Get the workspace path from the environment variables
WORKSPACE_PATH = Path(env_vars.get("WORKSPACE_PATH"))  # Fetch WORKSPACE_PATH from .env

if not WORKSPACE_PATH:
    raise ValueError("WORKSPACE_PATH is not defined in the .env file or is empty.")

# Add the WORKSPACE_PATH folder to the Python path
sys.path.append(str(WORKSPACE_PATH))

# Import custom utility functions
from src.config.utils import read_csv, write_csv, read_excel, split_string

DATA_DIR = Path(env_vars["DATA_DIR"])
MOMENT_OF_SUICIDE_FEATURES = split_string(env_vars["MOMENT_OF_SUICIDE_FEATURES"])
SOCIO_DEMOGRAPHIC_FEATURES = split_string(env_vars["SOCIO_DEMOGRAPHIC_FEATURES"])

# ================================================================================
# MAPPING DICTIONARIES
# ================================================================================
PREXES_TO_RETAIN = (
    MOMENT_OF_SUICIDE_FEATURES + SOCIO_DEMOGRAPHIC_FEATURES + ["ID", "Date", "AgeGroup"]
)

COLUMN_MAPPINGS_2013_2022 = {
    "ID samobójcy": "ID",
    "Data raportu [RRRRMM]": "Date",
    "Przedział wiekowy": "AgeGroup",
    "Płeć": "Gender",
    "Stan cywilny": "Marital",
    "Wykształcenie": "Education",
    "Informacje o pracy i nauce": "WorkInfo",
    "Źródło utrzymania": "Income",
    "Czy samobójstwo zakończyło się zgonem": "Fatal",
    "Miejsce zamachu": "Place",
    "Sposób popełnienia": "Method",
    "Stan świadomości": "Substance",
    "Informacje dotyczące leczenia z powodu alkoholizmu/narkomanii": "AbuseInfo2",
    "Powód zamachu": "Context1",
    "Powód zamachu 2": "Context2",
    "Powód zamachu 3": "Context3",
    "Powód zamachu 4": "Context4",
    "Informacje o używaniu substancji": "AbuseInfo1",
}

VALUE_MAPPINGS_2013_2022 = {
    "AgeGroup": {
        0: pd.NA,
        1: "07_12",
        2: "13_18",
        3: "19_24",
        4: "25_29",
        5: "30_34",
        6: "35_39",
        7: "40_44",
        8: "45_49",
        9: "50_54",
        10: "55_59",
        11: "60_64",
        12: "65_69",
        13: "70_74",
        14: "75_79",
        15: "80_84",
        16: "85",
    },
    "Gender": {0: pd.NA, 1: "F", 2: "M"},
    "Marital": {
        0: pd.NA,
        1: "Single",
        2: "Cohabitant",
        3: "Married",
        4: "Separated",
        5: "Divorced",
        6: "Widowed",
    },
    "Education": {
        0: pd.NA,
        1: "PrePrimary",
        2: "Primary",
        3: "Secondary",
        4: "Vocational",
        5: "Secondary",
        6: "Secondary",
        7: "Higher",
    },
    "WorkInfo": {
        0: pd.NA,
        1: "Student",
        2: "Student",
        3: "Employed",
        4: "Employed",
        5: "Agriculturalist",
        6: "Employed",
        7: "Employed",
        8: "Employed",
        9: "Unemployed",
    },
    "Income": {0: pd.NA, 1: "Dependent", 2: "Steady", 3: "Benefits", 4: "NoSteady"},
    "Fatal": {0: pd.NA, 1.0: 1, 2.0: 0},
    "Place": {
        0: pd.NA,
        1: "Road",
        2: "UtilitySpaces",
        3: "House",
        4: "Railway",
        5: "Forest",
        6: "House",
        7: "WaterRes",
        8: "Work",
        9: "Institution",
        10: "Isolation",
        11: "PoliceArmy",
        12: "Institution",
        13: "School",
        14: "PoliceArmy",
        15: "Other",
    },
    "Method": {
        0: pd.NA,
        1: "Vehicle",
        2: "Jumping",
        3: "Hanging",
        4: "SelfHarm",
        5: "Schooting",
        6: "SelfHarm",
        7: "SelfHarm",
        8: "Drugs",
        9: "Poisoning",
        10: "Gas",
        11: "Drugs",
        12: "Poisoning",
        13: "Drugs",
        14: "Drugs",
        15: "Drowning",
        16: "SelfHarm",
        17: "Other",
        18: "Other",
    },
    "Substance": {
        0: pd.NA,
        1: "Sober",
        2: "Alco",
        3: "OtherSub",
        4: "OtherSub",
        5: "OtherSub",
        6: "OtherSub",
        7: "AlcoOtherSub",
        8: "AlcoOtherSub",
        9: "AlcoOtherSub",
        10: "AlcoOtherSub",
        11: "AlcoOtherSub",
        12: "OtherSub",
        13: "OtherSub",
        14: "OtherSub",
        15: "OtherSub",
        16: "OtherSub",
        17: "AlcoOtherSub",
        18: "AlcoOtherSub",
    },
    "Context": {
        0: pd.NA,
        1: "HeartBreak",
        2: "MentalHealth",
        3: "FamilyConflict",
        4: "HealthLoss",
        5: "Finances",
        6: "Finances",
        7: "HealthLoss",
        8: "SchoolWork",
        9: "CloseDeath",
        10: "Crime",
        11: "Disability",
        12: "Other",
        13: "HealthLoss",
        14: "HealthLoss",
        15: "SchoolWork",
        16: "Finances",
        17: "SchoolWork",
        18: "Other",
    },
    "AbuseInfo1": {
        0: pd.NA,
        1: "Alco",
        2: "OtherSub",
        3: "OtherSub",
        4: "AlcoOtherSub",
        5: "OtherSub",
        6: "AlcoOtherSub",
        7: "AlcoOtherSub",
    },
    "AbuseInfo2": {0: pd.NA, 1: "Alco", 2: "OtherSub", 3: "AlcoOtherSub"},
}

COLUMN_MAPPINGS_2023 = {
    "ID samobójcy": "ID",
    "Data rejestracji": "Date",
    "Przedział wiekowy": "AgeGroup",
    "Płeć": "Gender",
    "Stan cywilny": "Marital",
    "Wykształcenie": "Education",
    "Informacje o pracy i nauce": "WorkInfo",
    "Źródło utrzymania": "Income",
    "Czy samobójstwo zakończyło się zgonem": "Fatal",
    "Miejsce zamachu": "Place",
    "Sposób popełnienia": "Method",
    "Stan świadomości *": "Substance",
    "Informacje dotyczące leczenia z powodu alkoholizmu/narkomanii": "AbuseInfo1",
    "Powód zamachu *": "Context1",
    "Powód zamachu 2": "Context2",
    "Powód zamachu 3": "Context3",
    "Powód zamachu 4": "Context4",
    "Informacje dotyczące stanu zdrowia *": "AbuseInfo2",
}

VALUE_MAPPINGS_2023 = {
    "AgeGroup": {
        "7-12": "07_12",
        "13-18": "13_18",
        "19-24": "19_24",
        "25-29": "25_29",
        "30-34": "30_34",
        "35-39": "35_39",
        "40-44": "40_44",
        "45-49": "45_49",
        "50-54": "50_54",
        "55-59": "55_59",
        "60-64": "60_64",
        "65-69": "65_69",
        "70-74": "70_74",
        "75-79": "75_79",
        "80-84": "80_84",
        "85+": "85",
        "Nieustalony wiek": pd.NA,
    },
    "Gender": {
        "M": "M",
        "Mężczyzna": "M",
        "K": "F",
        "Kobieta": "F",
    },
    "Marital": {
        "Kawaler/panna": "Single",
        "Żonaty/zamężna": "Married",
        "Wdowiec/wdowa": "Widowed",
        "Rozwiedziony/rozwiedziona": "Divorced",
        "W separacji": "Separated",
        "Konkubent/konkubina": "Cohabiting",
        "Brak danych/nieustalony": pd.NA,
    },
    "Education": {
        "Podstawowe": "Primary",
        "Gimnazjalne": "LowerSecondary",
        "Zasadnicze zawodowe": "Vocational",
        "Średnie": "Secondary",
        "Wyższe": "Higher",
        "Brak danych/nieustalony": pd.NA,
        "Nie dotyczy": pd.NA,
    },
    "WorkInfo": {
        "Brak danych/nieustalono": pd.NA,
        "Uczeń": "Student",
        "Student": "Student",
        "Rolnik": "Agriculturalist",
        "Pracujący na własny rachunek/samodzielna działalność gospodarcza": "Employed",
        "Praca stała": "Employed",
        "Praca dorywcza": "Employed",
        "Bezrobotny": "Unemployed",
    },
    "Income": {
        "Brak danych/nieustalony": pd.NA,
        "Na utrzymaniu innej osoby": "Dependent",
        "Praca": "Steady",
        "Emerytura": "Benefits",
        "Renta": "Benefits",
        "Zasiłek/alimenty": "Benefits",
        "Nie ma stałego źródła utrzymania": "NoSteady",
    },
    "Fatal": {"T": 1, "N": 0},
    "Place": {
        "Droga/ulica/chodnik": "Road",
        "Zabudowania gospodarcze": "UtilitySpaces",
        "Mieszkanie/dom": "House",
        "Teren kolei/tory": "Railway",
        "Park, las": "Forest",
        "Garaż/piwnica/strych": "House",
        "Rzeka/jezioro/inny zbiornik wodny": "WaterRes",
        "Zakład pracy": "Work",
        "Placówka lecznicza lub sanatoryjna": "Institution",
        "Miejsce prawnej izolacji": "Isolation",
        "Obiekt wojskowy": "PoliceArmy",
        "Placówka wychowawczo-opiekuńcza": "Institution",
        "Szkoła/uczelnia": "School",
        "Obiekt policyjny": "PoliceArmy",
        "Inne": "Other",
    },
    "Method": {
        "Rzucenie się pod pojazd w ruchu": "Vehicle",
        "Rzucenie się z wysokości": "Jumping",
        "Powieszenie się": "Hanging",
        "Uszkodzenie układu krwionośnego": "SelfHarm",
        "Zastrzelenie się/użycie broni palnej": "Shooting",
        "Samookaleczenie powierzchowne": "SelfHarm",
        "Zażycie środków nasennych/leków psychotropowych": "Drugs",
        "Zatrucie gazem/spalinami": "Gas",
        "Zażycie innych leków": "Drugs",
        "Zatrucie środkami chemicznymi/toksycznymi": "Poisoning",
        "Zatrucie środkami odurzającymi": "Drugs",
        "Zatrucie dopalaczami": "Drugs",
        "Utonięcie/utopienie się": "Drowning",
        "Samopodpalenie": "SelfHarm",
        "Uduszenie się": "Other",
        "Inny": "Other",
    },
    "Substance": {
        "Brak danych/nieustalony": pd.NA,
        "Trzeźwy(a)": "Sober",
        "Pod wpływem alkoholu": "Alco",
        "Pod wpływem zastępczych środków/substancji (dopalaczy)": "OtherSub",
        "Pod wpływem leków": "OtherSub",
        "Pod wpływem środków odurzających": "OtherSub",
        "Pod wpływem alkoholu i zastępczych środków/substancji (dopalaczy)": "AlcoOtherSub",
        "Pod wpływem alkoholu zastępczych środków/substancji (dopalaczy)": "AlcoOtherSub",
        "Pod wpływem alkoholu i leków": "AlcoOtherSub",
        "Pod wpływem alkoholu i środków odurzających": "AlcoOtherSub",
        "Pod wpływem leków i środków odurzających": "OtherSub",
        "Pod wpływem alkoholu, leków i środków odurzających": "AlcoOtherSub",
    },
    "Context": {
        "Nieustalony": pd.NA,
        "Zawód miłosny": "HeartBreak",
        "Leczony(a) psychiatrycznie": "MentalHealth",
        "Nieporozumienie rodzinne/przemoc w rodzinie": "FamilyConflict",
        "Nosiciel wirusa HIV, chory na AIDS": "HealthLoss",
        "Nagła utrata źródła utrzymania": "Finances",
        "Złe warunki ekonomiczne/długi": "Finances",
        "Choroba psychiczna/zaburzenia psychiczne": "MentalHealth",
        "Problemy w szkole lub pracy": "SchoolWork",
        "Śmierć bliskiej osoby": "CloseDeath",
        "Dokonanie przestępstwa lub wykroczenia": "Crime",
        "Trwałe kalectwo": "Disability",
        "Niepożądana ciąża": "Other",
        "Choroba fizyczna": "HealthLoss",
        "Pogorszenie lub nagła utrata zdrowia": "HealthLoss",
        "Konflikt z osobami spoza rodziny": "SchoolWork",
        "Zagrożenie lub utrata miejsca zamieszkania": "Finances",
        "Mobbing, cybermobbing, znęcanie": "SchoolWork",
        "Inny niewymieniony powyżej": "Other",
    },
    "AbuseInfo1": {
        "Leczony(a) psychiatrycznie": pd.NA,
        "Nadużywał(a) alkoholu": "Alco",
        "Leczony(a) z powodu alkoholizmu": "Alco",
        "Leczony(a) z powodu narkomanii": "OtherSub",
        "Leczony(a) z powodu alkoholizmu i narkomanii": "AlcoOtherSub",
    },
    "AbuseInfo2": {
        "Brak danych/nieustalono": pd.NA,
        "Nadużywał(a) alkoholu": "Alco",
        "Leczony(a) psychiatrycznie": pd.NA,
        "Leczony(a) z powodu alkoholizmu": "Alco",
        "Choroba fizyczna": pd.NA,
        "Trwałe kalectwo": pd.NA,
        "Zatrzymany(a) w izbie wytrzeźwień": "Alco",
        "Nadużywał(a) alkoholu i narkotyków": "AlcoOtherSub",
        "Nadużywał(a) alkoholu i nakrotyków": "AlcoOtherSub",
        "Leczony(a) z powodu narkomanii": "OtherSub",
        "Używał dopalaczy i narkotyków": "OtherSub",
        "Nadużywał(a) alkoholu i narkotykó": "OtherSub",
        "Nadużywał(a) alkoholu, dopalaczy i narkotyków": "AlcoOtherSub",
        "Nadużywał(a) alkoholu, narkotyków i dopalaczy": "AlcoOtherSub",
        "Nadużywał(a) alkoholu i dopalaczy": "AlcoOtherSub",
        "Używał dopalaczy": "OtherSub",
        "Nadużywał(a) alkoholu, dopalaczy, narkotyków": "AlcoOtherSub",
        "Leczony(a) psychiatrycznie, nadużywał(a) alkoholu": "Alco",
    },
}


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================


def map_columns(df: pd.DataFrame, column_mappings: Dict[str, str]) -> pd.DataFrame:
    """Rename columns in the DataFrame based on a mapping dictionary."""
    df.rename(columns=column_mappings, inplace=True)
    return df


def map_features(
    df: pd.DataFrame, column_mappings: Dict[str, str], value_mappings: Dict[str, Dict]
) -> pd.DataFrame:
    """Apply value mappings to DataFrame columns."""
    for old_col, new_col in column_mappings.items():
        if new_col in value_mappings and new_col in df.columns:
            # Convert column to object to handle mixed types
            df[new_col] = df[new_col].astype(object)

            # Apply mapping
            mapped = df[new_col].map(value_mappings[new_col])

            # Replace invalid values with pd.NA to ensure compatibility
            mapped = mapped.where(mapped.notna(), pd.NA)

            # Convert to object type
            df.loc[:, new_col] = mapped.astype(object)
    return df


def merge_columns(
    df: pd.DataFrame, columns: List[str], output_column: str
) -> pd.DataFrame:
    """
    Merge multiple columns into a single column, prioritizing non-null values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of column names to merge.
        output_column (str): Name of the resulting merged column.

    Returns:
        pd.DataFrame: Updated DataFrame with the merged column.
    """
    if not columns:
        return df

    df = df.copy()

    # Merge columns and drop originals
    merged_column = df[columns].bfill(axis=1).iloc[:, 0]
    df.loc[:, output_column] = merged_column
    df.drop(columns=columns, inplace=True, errors="ignore")
    return df


def encode_columns_by_prefix(
    df: pd.DataFrame,
    column_prefix: str,
    output_prefix: str,
    value_mapping: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Encode unique values from columns with a specific prefix into binary columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_prefix (str): Prefix of the columns to encode.
        output_prefix (str): Prefix for the resulting binary columns.
        value_mapping (Optional[Dict]): Optional mapping for values before encoding.

    Returns:
        pd.DataFrame: Updated DataFrame with binary-encoded columns.
    """
    target_columns = [col for col in df.columns if col.startswith(column_prefix)]
    if not target_columns:
        return df  # No columns with the specified prefix

    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Apply value mapping if provided
    if value_mapping:
        for col in target_columns:
            if col in df.columns:
                mapped_column = df[col].map(value_mapping)
                # Replace NaN with pd.NA for compatibility with object type
                mapped_column = mapped_column.astype(object).where(
                    mapped_column.notna(), pd.NA
                )
                df.loc[:, col] = mapped_column.astype(object)

    # Get unique non-null values across all target columns
    unique_values = pd.concat(
        [df[col].dropna() for col in target_columns], axis=0
    ).unique()

    for value in unique_values:
        binary_column = f"{output_prefix}_{value}"
        # Use .loc to modify the DataFrame, handling NA values explicitly
        df.loc[:, binary_column] = df[target_columns].apply(
            lambda row: int(value in row.dropna().values), axis=1
        )

    # Drop the original columns
    df.drop(columns=target_columns, inplace=True, errors="ignore")
    return df


def clean_data(
    df: pd.DataFrame,
    value_mappings: Dict[str, Dict],
    prefixes_to_retain: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Clean and preprocess data, including filtering columns based on mappings.

    Args:
        df (pd.DataFrame): Input DataFrame.
        value_mappings (Dict[str, Dict]): Mapping dictionary for values.
        column_mappings (Optional[Dict[str, str]]): Mapping dictionary for column names.

    Returns:
        pd.DataFrame: Cleaned and filtered DataFrame.
    """
    df = df.copy()

    # Drop rows with empty or NaN IDs
    if "ID" in df.columns:
        df = df.loc[~df["ID"].isna() & (df["ID"].str.strip() != "")]

    # Convert Date to datetime and extract Year/Month
    if "Date" in df.columns:
        # First, try to convert 'YYYYMM' format to datetime
        date_column_YYYYMM = pd.to_datetime(df["Date"], format="%Y%m", errors="coerce")
        # Then, convert any remaining valid date strings to datetime
        date_column_rest = pd.to_datetime(df["Date"], errors="coerce")
        df.loc[:, "Date"] = date_column_YYYYMM.fillna(date_column_rest)
        df["Date"] = df["Date"].astype("datetime64[ns]")

        if df["Date"].isna().all():
            raise ValueError(
                "All values in 'Date' column are invalid and could not be parsed."
            )

        # Create year and month columns if Date is valid
        # Ensure the Date column is in datetime format
        if pd.api.types.is_datetime64_any_dtype(df["Date"]):
            # Create year and month columns
            df["DateY"] = df["Date"].dt.year.astype("Int64").astype(str).str.zfill(4)
            df["DateM"] = df["Date"].dt.month.astype("Int64").astype(str).str.zfill(2)
        else:
            raise ValueError("Column 'Date' could not be converted to datetime format.")

    # Encode Context columns into binary features
    df = encode_columns_by_prefix(
        df,
        column_prefix="Context",
        output_prefix="Context",
        value_mapping=value_mappings.get("Context"),
    )

    # Merge AbuseInfo columns into a single column
    df = merge_columns(
        df,
        columns=[col for col in df.columns if col.startswith("AbuseInfo")],
        output_column="AbuseInfo",
    )

    # Filter columns to retain only those with prefixes in prefixes_to_retain
    if prefixes_to_retain:
        columns_to_retain = {
            col: col
            for col in df.columns
            if any(col.startswith(prefix) for prefix in prefixes_to_retain)
        }
        mapped_columns = set(columns_to_retain.values())
        current_columns = set(df.columns)
        columns_to_retain = current_columns & mapped_columns  # Keep only mapped columns
        unmapped_columns = current_columns - columns_to_retain  # Find unmapped columns

        # Drop unmapped columns
        df.drop(columns=list(unmapped_columns), inplace=True, errors="ignore")

    return df


def run_data_mapping(
    df_raw_2023: pd.DataFrame,
    df_raw_2013_2022: pd.DataFrame,
    column_mappings_2023: Dict[str, str],
    value_mappings_2023: Dict[str, Dict],
    column_mappings_2013_2022: Dict[str, str],
    value_mappings_2013_2022: Dict[str, Dict],
) -> pd.DataFrame:
    """Run data mapping for 2023 and 2013-2022 datasets and combine them."""
    df_2023 = map_columns(df_raw_2023, column_mappings_2023)
    df_2023 = map_features(df_2023, column_mappings_2023, value_mappings_2023)
    df_2023 = clean_data(df_2023, value_mappings_2023, PREXES_TO_RETAIN)

    df_2013_2022 = map_columns(df_raw_2013_2022, column_mappings_2013_2022)
    df_2013_2022 = map_features(
        df_2013_2022, column_mappings_2013_2022, value_mappings_2013_2022
    )
    df_2013_2022 = clean_data(df_2013_2022, value_mappings_2013_2022, PREXES_TO_RETAIN)

    df_combined = pd.concat([df_2023, df_2013_2022], ignore_index=True)

    df_combined.drop(columns=["ID"], inplace=True)  # Drop the original "ID" column
    df_combined.reset_index(drop=False, inplace=True)  # Reset the index
    df_combined.rename(
        columns={"index": "ID"}, inplace=True
    )  # Rename the new index column to "ID"
    return df_combined


# ================================================================================
# PROCESS DATASETS
# ================================================================================
if __name__ == "__main__":
    excel_file_path = Path(DATA_DIR) / "raw" / "Samobojstwa_2023.xlsx"
    df_raw_2023 = read_excel(excel_file_path)

    csv_file_path = Path(DATA_DIR) / "raw" / "final_samobojstwa_2013_2022.csv"
    df_raw_2013_2022 = read_csv(csv_file_path, delimiter=",", low_memory=False)

    df_mapped = run_data_mapping(
        df_raw_2023,
        df_raw_2013_2022,
        COLUMN_MAPPINGS_2023,
        VALUE_MAPPINGS_2023,
        COLUMN_MAPPINGS_2013_2022,
        VALUE_MAPPINGS_2013_2022,
    )

    # Save combined dataset
    output_file_path = Path(DATA_DIR) / "processed"
    write_csv(
        data=df_mapped,
        file_path=output_file_path / "mapped_data.csv",
        index=False,
    )
