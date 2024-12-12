"""Data mapping module for standardizing and transforming suicide study dataset.
Handles data from both 2023 and 2013-2022 periods."""

import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables from the .env file
load_dotenv()

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from config.config import DATA_DIR

pd.options.display.max_columns = None

# Set output directory
output_file_path = DATA_DIR / 'mapped'

#================================================================================
# MAPPING DICTIONARIES
#================================================================================

# Column name mappings
COLUMN_MAPPINGS = {
    'ID samobójcy': 'ID',
    'Data_rejestracji': 'Date',
    'Przedział_wiekowy': 'AgeGroup',
    'Płeć': 'Gender',
    'Stan_cywilny': 'Marital',
    'Wykształcenie': 'Education',
    'Informacje_o_pracy_i_nauce': 'WorkInfo',
    'Źródło_utrzymania': 'Income',
    'Czy_samobójstwo_zakończyło_się_zgonem': 'Fatal',
    'Miejsce_zamachu': 'Place',
    'Sposób_popełnienia': 'Method',
    'Stan_świadomości_*': 'Substance',
    'Informacje_dotyczące_leczenia_z_powodu_alkoholizmu/narkomanii': 'AbuseHistory'
}

# Value mappings for each column
VALUE_MAPPINGS = {
    'Gender': {
        'M': 'M',
        'K': 'F'
    },
    'Marital': {
        'Kawaler/panna': 'Single',
        'Żonaty/zamężna': 'Married',
        'Wdowiec/wdowa': 'Widowed',
        'Rozwiedziony/rozwiedziona': 'Divorced',
        'W separacji': 'Separated',
        'Konkubent/konkubina': 'Cohabiting',
        'Brak danych/nieustalony': np.nan
    },
    'Education': {
        'Podstawowe': 'Primary',
        'Gimnazjalne': 'LowerSecondary',
        'Zasadnicze zawodowe': 'Vocational',
        'Średnie': 'Secondary',
        'Wyższe': 'Higher',
        'Brak danych/nieustalony': np.nan,
        'Nie dotyczy': np.nan
    },
    'WorkInfo': {
        'Pracuje': 'Working',
        'Uczy się': 'Student',
        'Pracuje i uczy się': 'WorkingStudent',
        'Nie pracuje i nie uczy się': 'Neither',
        'Brak danych/nieustalony': np.nan,
        'Nie dotyczy': np.nan
    },
    'Income': {
        'Brak danych/nieustalony': np.nan,
        'Na utrzymaniu innej osoby': 'Dependent',
        'Praca': 'Steady',
        'Emerytura': 'Benefits',
        'Renta': 'Benefits',
        'Zasiłek/alimenty': 'Benefits',
        'Nie ma stałego źródła utrzymania': 'NoSteady'
    },
    'Fatal': {
        'T': 1,
        'N': 0
    },
    'Place': {
        'Droga/ulica/chodnik': 'Road',
        'Zabudowania gospodarcze': 'UtilitySpaces',
        'Mieszkanie/dom': 'House',
        'Teren kolei/tory': 'Railway',
        'Park, las': 'Forest',
        'Garaż/piwnica/strych': 'House',
        'Rzeka/jezioro/inny zbiornik wodny': 'WaterRes',
        'Zakład pracy': 'Work',
        'Placówka lecznicza lub sanatoryjna': 'Institution',
        'Miejsce prawnej izolacji': 'Isolation',
        'Obiekt wojskowy': 'PoliceArmy',
        'Placówka wychowawczo-opiekuńcza': 'Institution',
        'Szkoła/uczelnia': 'School',
        'Obiekt policyjny': 'PoliceArmy',
        'Inne': 'Other'
    },
    'Method': {
        'Rzucenie się pod pojazd w ruchu': 'Vehicle',
        'Rzucenie się z wysokości': 'Jumping',
        'Powieszenie się': 'Hanging',
        'Uszkodzenie układu krwionośnego': 'SelfHarm',
        'Zastrzelenie się/użycie broni palnej': 'Shooting',
        'Samookaleczenie powierzchowne': 'SelfHarm',
        'Zażycie środków nasennych/leków psychotropowych': 'Drugs',
        'Zatrucie gazem/spalinami': 'Gas',
        'Zażycie innych leków': 'Drugs',
        'Zatrucie środkami chemicznymi/toksycznymi': 'Poisoning',
        'Zatrucie środkami odurzającymi': 'Drugs',
        'Zatrucie dopalaczami': 'Drugs',
        'Utonięcie/utopienie się': 'Drowning',
        'Samopodpalenie': 'SelfHarm',
        'Uduszenie się': 'Other',
        'Inny': 'Other'
    },
    'Context': {
        'Nieustalony': np.nan,
        'Zawód miłosny': 'HeartBreak',
        'Leczony(a) psychiatrycznie': 'MentalHealth',
        'Nieporozumienie rodzinne/przemoc w rodzinie': 'FamilyConflict',
        'Nosiciel wirusa HIV, chory na AIDS': 'HealthLoss',
        'Nagła utrata źródła utrzymania': 'Finances',
        'Złe warunki ekonomiczne/długi': 'Finances',
        'Choroba psychiczna/zaburzenia psychiczne': 'MentalHealth',
        'Problemy w szkole lub pracy': 'SchoolWork',
        'Śmierć bliskiej osoby': 'CloseDeath',
        'Dokonanie przestępstwa lub wykroczenia': 'Crime',
        'Trwałe kalectwo': 'Disability',
        'Niepożądana ciąża': 'Other',
        'Choroba fizyczna': 'HealthLoss',
        'Pogorszenie lub nagła utrata zdrowia': 'HealthLoss',
        'Konflikt z osobami spoza rodziny': 'SchoolWork',
        'Zagrożenie lub utrata miejsca zamieszkania': 'Finances',
        'Mobbing, cybermobbing, znęcanie': 'SchoolWork',
        'Inny niewymieniony powyżej': 'Other'
    }
}

#================================================================================
# MAPPING FUNCTIONS
#================================================================================

def map_column(df: pd.DataFrame, old_col: str, new_col: str, mapping_dict: dict) -> pd.DataFrame:
    """Map values in a column using provided mapping dictionary and rename the column.
    
    Args:
        df: Input DataFrame
        old_col: Original column name
        new_col: New column name
        mapping_dict: Dictionary mapping old values to new values
    
    Returns:
        DataFrame with mapped and renamed column
    """
    if old_col in df.columns:
        df[old_col] = df[old_col].map(mapping_dict)
        df.rename(columns={old_col: new_col}, inplace=True)
    return df

def map_features(df: pd.DataFrame, column_mappings: dict, value_mappings: dict) -> pd.DataFrame:
    """Map multiple features using provided column and value mappings.
    
    Args:
        df: Input DataFrame
        column_mappings: Dictionary mapping old column names to new ones
        value_mappings: Dictionary of dictionaries with value mappings for each column
    
    Returns:
        DataFrame with all specified columns mapped
    """
    for old_col, new_col in column_mappings.items():
        if new_col in value_mappings:
            df = map_column(df, old_col, new_col, value_mappings[new_col])
        else:
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
    return df

def process_context_features(df: pd.DataFrame, context_mapping: dict) -> pd.DataFrame:
    """Process context-related features and create dummy variables.
    
    Args:
        df: Input DataFrame
        context_mapping: Dictionary mapping context values
    
    Returns:
        DataFrame with processed context features
    """
    # Map multiple context columns
    context_columns = []
    for i, suffix in enumerate(['*', '2', '3', '4']):
        column = f'Powód_zamachu_{suffix}'
        if column in df.columns:
            df[column] = df[column].map(context_mapping)
            new_name = f'Context{i+1 if i > 0 else ""}'
            df.rename(columns={column: new_name}, inplace=True)
            context_columns.append(new_name)

    if context_columns:
        # Create dummy variables for contexts
        df_context = pd.get_dummies(df[context_columns].fillna('Missing'), prefix='', prefix_sep='')
        df = pd.concat([df, df_context], axis=1)
        df.drop(columns=context_columns, inplace=True)

        # Count number of contexts
        df['ContextCount'] = df_context.sum(axis=1)

    return df

def load_and_clean_data(file_path: str, is_excel: bool = True) -> pd.DataFrame:
    """Load and perform initial cleaning of dataset.
    
    Args:
        file_path: Path to the data file
        is_excel: Whether the file is Excel (True) or CSV (False)
    
    Returns:
        Cleaned DataFrame
    """
    # Load data
    df = pd.read_excel(file_path) if is_excel else pd.read_csv(file_path, low_memory=False)
    
    # Clean ID column
    if 'ID samobójcy' in df.columns:
        df.rename(columns={'ID samobójcy': 'ID'}, inplace=True)
        non_numeric_ids = df[df['ID'].str.contains(r'\D', na=False)]
        nan_rows = df[df['ID'].isna()]
        df = df[~df.index.isin(non_numeric_ids.index) & ~df.index.isin(nan_rows.index)]
    
    # Process dates if present
    date_col = 'Data_rejestracji' if 'Data_rejestracji' in df.columns else 'Date'
    if date_col in df.columns:
        df.rename(columns={date_col: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df['DateY'] = df['Date'].dt.strftime('%Y')
        df['DateM'] = df['Date'].dt.strftime('%m')
        df['Date'] = df['DateM'] + '.' + df['DateY']
    
    return df

#================================================================================
# PROCESS DATASETS
#================================================================================

def main():
    # Process 2023 Dataset
    excel_file_path = DATA_DIR / 'raw' / 'Samobojstwa_2023.xlsx'
    df_raw_2023 = load_and_clean_data(excel_file_path, is_excel=True)
    df_raw_2023 = map_features(df_raw_2023, COLUMN_MAPPINGS, VALUE_MAPPINGS)
    df_raw_2023 = process_context_features(df_raw_2023, VALUE_MAPPINGS['Context'])

    # Process 2013-2022 Dataset
    csv_file_path = DATA_DIR / 'raw' / 'final_samobojstwa_2013_2022.csv'
    df_raw_2013_2022 = load_and_clean_data(csv_file_path, is_excel=False)
    df_raw_2013_2022 = map_features(df_raw_2013_2022, COLUMN_MAPPINGS, VALUE_MAPPINGS)
    df_raw_2013_2022 = process_context_features(df_raw_2013_2022, VALUE_MAPPINGS['Context'])

    # Combine datasets and save
    df_combined = pd.concat([df_raw_2023, df_raw_2013_2022], ignore_index=True)
    output_file_path.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist
    df_combined.to_csv(output_file_path / 'mapped_data.csv', index=False)
    
if __name__ == "__main__":
    main()