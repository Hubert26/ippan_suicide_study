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
#Group_AG
conditions = [
    (df_imputed['GroupAge'] == "00_18") & (df_imputed['Gender'] == 0),  
    (df_imputed['GroupAge'] == "00_18") & (df_imputed['Gender'] == 1),
    (df_imputed['GroupAge'] == "19_34") & (df_imputed['Gender'] == 0),
    (df_imputed['GroupAge'] == "19_34") & (df_imputed['Gender'] == 1),
    (df_imputed['GroupAge'] == "35_64") & (df_imputed['Gender'] == 0),
    (df_imputed['GroupAge'] == "35_64") & (df_imputed['Gender'] == 1),
    (df_imputed['GroupAge'] == "65") & (df_imputed['Gender'] == 0),
    (df_imputed['GroupAge'] == "65") & (df_imputed['Gender'] == 1)
]

choices = ["00_18F", "00_18M", "19_34F", "19_34M", "35_64F", "35_64M", "65F", "65M"]

df_imputed['Group_AG'] = np.select(conditions, choices, default=np.nan)

#%%
#Group_AF
conditions = [
    (df_imputed['GroupAge'] == "00_18") & (df_imputed['Fatal'] == 0),  
    (df_imputed['GroupAge'] == "00_18") & (df_imputed['Fatal'] == 1),
    (df_imputed['GroupAge'] == "19_34") & (df_imputed['Fatal'] == 0),
    (df_imputed['GroupAge'] == "19_34") & (df_imputed['Fatal'] == 1),
    (df_imputed['GroupAge'] == "35_64") & (df_imputed['Fatal'] == 0),
    (df_imputed['GroupAge'] == "35_64") & (df_imputed['Fatal'] == 1),
    (df_imputed['GroupAge'] == "65") & (df_imputed['Fatal'] == 0),
    (df_imputed['GroupAge'] == "65") & (df_imputed['Fatal'] == 1)
]

choices = ["00_18False", "00_18True", "19_34False", "19_34True", "35_64False", "35_64True", "65False", "65True"]

df_imputed['Group_AF'] = np.select(conditions, choices, default=np.nan)

#%%
#Group_AGF
conditions = [
    (df_imputed['GroupAge'] == "00_18") & (df_imputed['Gender'] == 0) & (df_imputed['Fatal'] == 0),
    (df_imputed['GroupAge'] == "00_18") & (df_imputed['Gender'] == 0) & (df_imputed['Fatal'] == 1),
    (df_imputed['GroupAge'] == "00_18") & (df_imputed['Gender'] == 1) & (df_imputed['Fatal'] == 0),
    (df_imputed['GroupAge'] == "00_18") & (df_imputed['Gender'] == 1) & (df_imputed['Fatal'] == 1),
    (df_imputed['GroupAge'] == "19_34") & (df_imputed['Gender'] == 0) & (df_imputed['Fatal'] == 0),
    (df_imputed['GroupAge'] == "19_34") & (df_imputed['Gender'] == 0) & (df_imputed['Fatal'] == 1),
    (df_imputed['GroupAge'] == "19_34") & (df_imputed['Gender'] == 1) & (df_imputed['Fatal'] == 0),
    (df_imputed['GroupAge'] == "19_34") & (df_imputed['Gender'] == 1) & (df_imputed['Fatal'] == 1),
    (df_imputed['GroupAge'] == "35_64") & (df_imputed['Gender'] == 0) & (df_imputed['Fatal'] == 0),
    (df_imputed['GroupAge'] == "35_64") & (df_imputed['Gender'] == 0) & (df_imputed['Fatal'] == 1),
    (df_imputed['GroupAge'] == "35_64") & (df_imputed['Gender'] == 1) & (df_imputed['Fatal'] == 0),
    (df_imputed['GroupAge'] == "35_64") & (df_imputed['Gender'] == 1) & (df_imputed['Fatal'] == 1),
    (df_imputed['GroupAge'] == "65") & (df_imputed['Gender'] == 0) & (df_imputed['Fatal'] == 0),
    (df_imputed['GroupAge'] == "65") & (df_imputed['Gender'] == 0) & (df_imputed['Fatal'] == 1),
    (df_imputed['GroupAge'] == "65") & (df_imputed['Gender'] == 1) & (df_imputed['Fatal'] == 0),
    (df_imputed['GroupAge'] == "65") & (df_imputed['Gender'] == 1) & (df_imputed['Fatal'] == 1),
]

choices = ["00_18FFalse", "00_18FTrue", "00_18MFalse", "00_18MTrue",
           "19_34FFalse", "19_34FTrue", "19_34MFalse", "19_34MTrue",
           "35_64FFalse", "35_64FTrue", "35_64MFalse", "35_64MTrue",
           "65FFalse", "65FTrue", "65MFalse", "65MTrue"
           ]

df_imputed['Group_AGF'] = np.select(conditions, choices, default=np.nan)

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
columns_to_encode = [
    'Fatal',
    'AbuseInfo',
    'Gender',
    'Income',
    'Method',
    'Education',
    'WorkInfo',
    'Substance',
    'Place',
    'Marital',
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

columns_to_merge = [
    'ID',
    'Group_AG',
    'Group_AF',
    'Group_AGF',
    'DateY'
    ]

df_encoded = df_imputed[columns_to_encode]

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df_encoded, drop_first=False)

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

# Merge additional columns
df_encoded[columns_to_merge] = df_imputed[columns_to_merge].copy()


#%%
#Saveing
file_name = 'encoded_data.csv'
output_file_path = DATA_DIR / 'encoded'
write_to_csv(df_encoded, output_file_path / file_name, index=False)