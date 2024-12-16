"""Data mapping module for standardizing and transforming suicide study dataset.
Handles data from both 2023 and 2013-2022 periods."""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables from the .env file
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")

# Set output directory
output_file_path = Path(DATA_DIR) / "mapped"
print(f"Output Directory: {output_file_path}")

# ================================================================================
# MAPPING DICTIONARIES
# ================================================================================

# Column name mappings
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

# Value mappings for each column
VALUE_MAPPINGS_2013_2022 = {
    "AgeGroup": {
        0: np.nan,
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
    "Gender": {0: np.nan, 1: "F", 2: "M"},
    "Marital": {
        0: np.nan,
        1: "Single",
        2: "Cohabitant",
        3: "Married",
        4: "Separated",
        5: "Divorced",
        6: "Widowed",
    },
    "Education": {
        0: np.nan,
        1: "PrePrimary",
        2: "Primary",
        3: "Secondary",
        4: "Vocational",
        5: "Secondary",
        6: "Secondary",
        7: "Higher",
    },
    "WorkInfo": {
        0: np.nan,
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
    "Income": {0: np.nan, 1: "Dependent", 2: "Steady", 3: "Benefits", 4: "NoSteady"},
    "Fatal": {0: np.nan, 1.0: 1, 2.0: 0},
    "Place": {
        0: np.nan,
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
        0: np.nan,
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
        0: np.nan,
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
        0: np.nan,
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
        0: np.nan,
        1: "Alco",
        2: "OtherSub",
        3: "OtherSub",
        4: "AlcoOtherSub",
        5: "OtherSub",
        6: "AlcoOtherSub",
        7: "AlcoOtherSub",
    },
    "AbuseInfo2": {0: np.nan, 1: "Alco", 2: "OtherSub", 3: "AlcoOtherSub"},
}

# Column name mappings
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

# Value mappings for each column
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
        "Nieustalony wiek": np.nan,
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
        "Brak danych/nieustalony": np.nan,
    },
    "Education": {
        "Podstawowe": "Primary",
        "Gimnazjalne": "LowerSecondary",
        "Zasadnicze zawodowe": "Vocational",
        "Średnie": "Secondary",
        "Wyższe": "Higher",
        "Brak danych/nieustalony": np.nan,
        "Nie dotyczy": np.nan,
    },
    "WorkInfo": {
        "Brak danych/nieustalono": np.nan,
        "Uczeń": "Student",
        "Student": "Student",
        "Rolnik": "Agriculturalist",
        "Pracujący na własny rachunek/samodzielna działalność gospodarcza": "Employed",
        "Praca stała": "Employed",
        "Praca dorywcza": "Employed",
        "Bezrobotny": "Unemployed",
    },
    "Income": {
        "Brak danych/nieustalony": np.nan,
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
        "Brak danych/nieustalony": np.nan,
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
        "Nieustalony": np.nan,
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
        "Leczony(a) psychiatrycznie": np.nan,
        "Nadużywał(a) alkoholu": "Alco",
        "Leczony(a) z powodu alkoholizmu": "Alco",
        "Leczony(a) z powodu narkomanii": "OtherSub",
        "Leczony(a) z powodu alkoholizmu i narkomanii": "AlcoOtherSub",
    },
    "AbuseInfo2": {
        "Brak danych/nieustalono": np.nan,
        "Nadużywał(a) alkoholu": "Alco",
        "Leczony(a) psychiatrycznie": np.nan,
        "Leczony(a) z powodu alkoholizmu": "Alco",
        "Choroba fizyczna": np.nan,
        "Trwałe kalectwo": np.nan,
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
# MAPPING FUNCTIONS
# ================================================================================


def map_column(df: pd.DataFrame, old_col: str, new_col: str) -> pd.DataFrame:
    """Rename a column in the DataFrame.

    Args:
        df: Input DataFrame
        old_col: Original column name
        new_col: New column name

    Returns:
        DataFrame with renamed column
    """
    if old_col in df.columns:
        df.rename(columns={old_col: new_col}, inplace=True)
    return df


def map_columns(df: pd.DataFrame, column_mappings: dict) -> pd.DataFrame:
    """Rename multiple columns in the DataFrame based on a mapping dictionary.

    Args:
        df: Input DataFrame
        column_mappings: Dictionary mapping old column names to new ones

    Returns:
        DataFrame with renamed columns
    """
    for old_col, new_col in column_mappings.items():
        df = map_column(df, old_col, new_col)
    return df


def load_data(file_path: str, is_excel: bool = True):
    # Load dataset from the specified file path.
    if is_excel:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path, low_memory=False)
    return df


def map_features(
    df: pd.DataFrame, column_mappings: dict, value_mappings: dict
) -> pd.DataFrame:
    """Map multiple features using provided column and value mappings."""
    for old_col, new_col in column_mappings.items():
        if new_col in value_mappings:
            # Check if value_mappings[new_col] is a dictionary
            if isinstance(value_mappings[new_col], dict):
                if new_col in df.columns:  # Ensure new_col exists in the DataFrame
                    df[new_col] = df[new_col].map(value_mappings[new_col])

    # Replace values in Context columns
    context_columns = ["Context1", "Context2", "Context3", "Context4"]
    if "Context" in value_mappings:
        for context_col in context_columns:
            if context_col in df.columns:
                df[context_col] = df[context_col].map(value_mappings["Context"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Clean ID column
    if "ID" in df.columns:
        df = df.loc[
            ~df["ID"].isna() & (df["ID"].str.strip() != "")
        ].copy()  # Create a copy to avoid warnings

    # Process dates if present
    if "Date" in df.columns:
        # First, try to convert 'YYYYMM' format to datetime
        df["Date"] = pd.to_datetime(
            df["Date"], format="%Y%m", errors="coerce"
        )  # Coerce invalid formats to NaT

        # Then, convert any remaining valid date strings to datetime
        df["Date"] = pd.to_datetime(
            df["Date"], errors="coerce"
        )  # Convert valid date strings to datetime

        # Create year and month columns if Date is valid
        if df["Date"].notna().any():
            df.loc[:, "DateY"] = df["Date"].dt.strftime("%Y")
            df.loc[:, "DateM"] = df["Date"].dt.strftime("%m")
            # Combine and convert to datetime
            df.loc[:, "Date"] = pd.to_datetime(
                df["DateM"] + "." + df["DateY"], format="%m.%Y", errors="coerce"
            )

    # Find context columns dynamically
    context_columns = [col for col in df.columns if col.startswith(("Context"))]

    # Extract unique context values
    context_values = set()
    for col in context_columns:
        if col in df.columns:
            context_values.update(df[col].dropna().unique())

    # Create new binary columns for each unique context value
    for value in context_values:
        column_name = f"Context_{value}"
        df.loc[:, column_name] = df[context_columns].apply(
            lambda row: 1 if value in row.values else 0, axis=1
        )

    # Drop original context columns
    df.drop(columns=context_columns, inplace=True, errors="ignore")

    # Merge AbuseInfo columns
    abuse_info_columns = [col for col in df.columns if col.startswith("AbuseInfo")]
    if abuse_info_columns:
        if len(abuse_info_columns) == 1:
            df.loc[:, "AbuseInfo"] = df[
                abuse_info_columns[0]
            ]  # Directly assign the single column
        else:
            df.loc[:, "AbuseInfo"] = (
                df[abuse_info_columns].bfill(axis=1).iloc[:, 0]
            )  # Use backfill to fill NaNs
        df.drop(columns=abuse_info_columns, inplace=True, errors="ignore")

    return df


# ================================================================================
# PROCESS DATASETS
# ================================================================================


def main():
    # Process 2023 Dataset
    excel_file_path = Path(DATA_DIR) / "raw" / "Samobojstwa_2023.xlsx"
    df_raw_2023 = load_data(excel_file_path, is_excel=True)
    df_raw_2023 = map_columns(df_raw_2023, COLUMN_MAPPINGS_2023)
    df_raw_2023 = map_features(df_raw_2023, COLUMN_MAPPINGS_2023, VALUE_MAPPINGS_2023)
    df_raw_2023 = clean_data(df_raw_2023)

    # Process 2013-2022 Dataset
    csv_file_path = Path(DATA_DIR) / "raw" / "final_samobojstwa_2013_2022.csv"
    df_raw_2013_2022 = load_data(csv_file_path, is_excel=False)
    df_raw_2013_2022 = map_columns(df_raw_2013_2022, COLUMN_MAPPINGS_2013_2022)
    df_raw_2013_2022 = map_features(
        df_raw_2013_2022, COLUMN_MAPPINGS_2013_2022, VALUE_MAPPINGS_2013_2022
    )
    df_raw_2013_2022 = clean_data(df_raw_2013_2022)

    # Combine datasets
    df_combined = pd.concat([df_raw_2023, df_raw_2013_2022], ignore_index=True)

    # Delete ID column if it exists
    if "ID" in df_combined.columns:
        df_combined.drop(columns=["ID"], inplace=True)

    # Set the index as ID
    df_combined.reset_index(drop=True, inplace=True)  # Reset index to avoid confusion
    df_combined.index.name = "ID"  # Set the index name to 'ID'

    # Define columns to retain based on prefixes
    columns_to_retain = [
        col
        for col in df_combined.columns
        if col.startswith(("AbuseInfo", "Context", "Date"))
    ]

    # Delete unnecessary columns
    current_columns = df_combined.columns.tolist()
    mapped_columns = set(COLUMN_MAPPINGS_2023.values()) | set(
        COLUMN_MAPPINGS_2013_2022.values()
    )

    # Find columns in df that are not in COLUMN_MAPPINGS and are not in columns_to_retain
    unmapped_columns = [
        col
        for col in current_columns
        if col not in mapped_columns and col not in columns_to_retain
    ]

    # Drop the unmapped columns
    df_combined.drop(columns=unmapped_columns, inplace=True, errors="ignore")

    # Save combined dataset
    output_file_path = Path(DATA_DIR) / "mapped"
    output_file_path.mkdir(
        parents=True, exist_ok=True
    )  # Create output directory if it doesn't exist
    df_combined.to_csv(output_file_path / "mapped_data.csv", index=False)


if __name__ == "__main__":
    main()
