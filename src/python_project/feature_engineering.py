# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:12:51 2024

@author: huber
"""

from pathlib import Path
import sys
import pandas as pd

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from utils.dataframe_utils import read_csv_file, write_to_csv


#%%
current_working_directory = Path.cwd()
grandparent_directory = current_working_directory.parent.parent
output_file_path = grandparent_directory / 'data' / 'encoded'

#%%
csv_file_path = grandparent_directory / 'data' / 'imputed' / 'imputed_data.csv'
df_imputed = read_csv_file(csv_file_path, delimiter=',', low_memory=False, index_col=None, dtype={'DateY': str, 'DateM': str})

#%%
df_imputed['GroupAge'] = df_imputed['GroupAge2']
columns_to_drop = ['ID', 'Date', 'DateM', 'DateY', 'GroupAge1', 'GroupAge2', 'CountContext']
df_imputed.drop(columns_to_drop, inplace=True, axis=1)
    
#%%
# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df_imputed, drop_first=True)

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
write_to_csv(df_encoded, output_file_path / file_name, index=False)