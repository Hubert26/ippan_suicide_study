# settings.py
from pathlib import Path
from src.utils.utils import load_yaml

SETTINGS_DIR = Path(__file__).parent  # Absolute path to settings/

# DATA_MAPPINGS
DATA_MAPPINGS_PATH = SETTINGS_DIR / "data_mappings.yaml"
DATA_MAPPINGS = load_yaml(DATA_MAPPINGS_PATH)

COLUMN_MAPPINGS_2023 = DATA_MAPPINGS['column_mappings']['2023']
VALUE_MAPPINGS_2023 = DATA_MAPPINGS['value_mappings']['2023']
COLUMN_MAPPINGS_2013_2022 = DATA_MAPPINGS['column_mappings']['2013_2022']
VALUE_MAPPINGS_2013_2022 = DATA_MAPPINGS['value_mappings']['2013_2022']

def print_data_mappings():
    print("COLUMN_MAPPINGS_2013_2022:", COLUMN_MAPPINGS_2013_2022, "\n")
    print("COLUMN_MAPPINGS_2023:", COLUMN_MAPPINGS_2023, "\n")
    print("VALUE_MAPPINGS_2013_2022:", VALUE_MAPPINGS_2013_2022, "\n")
    print("VALUE_MAPPINGS_2023:", VALUE_MAPPINGS_2023, "\n")
  
# FEATURES
FEATURES_PATH = SETTINGS_DIR / "features.yaml"
FEATURES = load_yaml(FEATURES_PATH)

MOMENT_OF_SUICIDE_FEATURES = FEATURES['moment_of_suicide_features']
SOCIO_DEMOGRAPHIC_FEATURES = FEATURES['socio_demographic_features']
  
def print_features():
    print("MOMENT_OF_SUICIDE_FEATURES:", MOMENT_OF_SUICIDE_FEATURES, "\n")
    print("SOCIO_DEMOGRAPHIC_FEATURES:", SOCIO_DEMOGRAPHIC_FEATURES, "\n")
    
# GROUP_MAPPINGS
GROUP_MAPPINGS_PATH = SETTINGS_DIR / "group_mappings.yaml"
GROUP_MAPPINGS = load_yaml(GROUP_MAPPINGS_PATH)

AGE_MAPPING = GROUP_MAPPINGS['age_mapping']
AGE_GENDER_MAPPING = GROUP_MAPPINGS['age_gender_mapping']
AGE_FATALITY_MAPPING = GROUP_MAPPINGS['age_fatality_mapping']
AGE_GENDER_FATALITY_MAPPING = GROUP_MAPPINGS['age_gender_fatality_mapping']

def print_group_mappings():
    print("AGE_MAPPING:", AGE_MAPPING, "\n")
    print("AGE_GENDER_MAPPING:", AGE_GENDER_MAPPING, "\n")
    print("AGE_FATALITY_MAPPING:", AGE_FATALITY_MAPPING, "\n")
    print("AGE_GENDER_FATALITY_MAPPING:", AGE_GENDER_FATALITY_MAPPING, "\n")
    
if __name__ == "__main__":
    #print_data_mappings()
    #print_features()
    print_group_mappings()