# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:12:51 2024

@author: huber
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

from config import *
from utils.dataframe_utils import read_csv_file, write_to_csv



#%%
csv_file_path = DATA_DIR / 'imputed' / 'imputed_data.csv'
df_imputed = read_csv_file(csv_file_path, delimiter=',', low_memory=False)

#%%
df_imputed['GroupAge'] = df_imputed['GroupAge2']
columns_to_drop = ['Date', 'DateM', 'GroupAge1', 'GroupAge2', 'CountContext']
df_imputed.drop(columns_to_drop, inplace=True, axis=1)

#%%
conditions = [
    (df_imputed['GroupAge'] == "00_18") & (df_imputed['Gender'] == 0),  # A
    (df_imputed['GroupAge'] == "00_18") & (df_imputed['Gender'] == 1),  # B
    (df_imputed['GroupAge'] == "19_34") & (df_imputed['Gender'] == 0),  # C
    (df_imputed['GroupAge'] == "19_34") & (df_imputed['Gender'] == 1),  # D
    (df_imputed['GroupAge'] == "35_64") & (df_imputed['Gender'] == 0),  # E
    (df_imputed['GroupAge'] == "35_64") & (df_imputed['Gender'] == 1),  # F
    (df_imputed['GroupAge'] == "65") & (df_imputed['Gender'] == 0),  # G
    (df_imputed['GroupAge'] == "65") & (df_imputed['Gender'] == 1)  # H
]

choices = ["A", "B", "C", "D", "E", "F", "G", "H"]

df_imputed['Group'] = np.select(conditions, choices, default=np.nan)

#%%
#Saveing
file_name = 'final_feature_set.csv'
output_file_path = DATA_DIR / 'prepped'
write_to_csv(df_imputed, output_file_path / file_name, index=False)
#%%



#%%
#================================================================================
# One-Hot Encoding
#================================================================================
df_imputed['DateY'] = df_imputed['DateY'].astype(str)

id_column = df_imputed['ID']
df_imputed = df_imputed.drop(columns=['ID'])

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df_imputed, drop_first=False)

#%%
df_encoded['ID'] = id_column

#%%
bool_columns = [
    'Fatal',
    'Gender',
    'Context_Other',
    'Context_FamilyConflict',
    'Context_HeartBreak',
    'Context_Finances',
    'Context_SchoolWork',
    'Context_CloseDeath',
    'Context_Crime',
    'Context_Disability',
    'Context_MentalHealth',
    'Context_HealthLoss'
]

# Convert these columns to boolean type
df_encoded[bool_columns] = df_encoded[bool_columns].astype(bool)

#%%
#Saveing
file_name = 'encoded_data.csv'
output_file_path = DATA_DIR / 'encoded'
write_to_csv(df_encoded, output_file_path / file_name, index=False)