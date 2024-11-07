# -*- coding: utf-8 -*-
"""data_preparation.ipynb"""


import pandas as pd
import numpy as np
from pathlib import Path
import sys

from config import *
from utils.dataframe_utils import read_excel_file, read_csv_file, write_to_csv

pd.options.display.max_columns = None

#%%
output_file_path = DATA_DIR / 'mapped'

#%%
"""# 2023"""
excel_file_path = DATA_DIR / 'raw' / 'Samobojstwa_2023.xlsx'
df_raw_2023 = read_excel_file(excel_file_path)

#%%

"""Changing column names"""
df_raw_2023.rename(columns={'ID samobójcy': 'ID'}, inplace=True)

df_raw_2023.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

df_raw_2023.rename(columns={'Data_rejestracji': 'Date'}, inplace=True)

#%%
"""ID"""
non_numeric_ids = df_raw_2023[df_raw_2023['ID'].str.contains(r'\D', na=False)]
nan_rows = df_raw_2023[df_raw_2023['ID'].isna()]

df_raw_2023 = df_raw_2023[~df_raw_2023.index.isin(non_numeric_ids.index)]

#%%
"""Daty"""
# Convert the 'Date' column to datetime format
df_raw_2023['Date'] = pd.to_datetime(df_raw_2023['Date'])

# Extract year and month into separate columns
df_raw_2023['DateY'] = df_raw_2023['Date'].dt.strftime('%Y')
df_raw_2023['DateM'] = df_raw_2023['Date'].dt.strftime('%m')

# Combine month and year into a single column in MM.YYYY format
df_raw_2023['Date'] = df_raw_2023['DateM'] + '.' + df_raw_2023['DateY']

#%%
"""Mapowanie"""

#%%
"""Przedział_wiekowy"""


column = 'Przedział_wiekowy'
df_raw_2023.rename(columns={column: 'GroupAge1'}, inplace=True)

df_raw_2023['GroupAge1'].unique()

column = 'GroupAge1'
mapping = {
    "7-12": '07_12',
    '13-18': '13_18',
    '19-24': '19_24',
    '25-29': '25_29',
    '30-34': '30_34',
    '35-39': '35_39',
    '40-44': '40_44',
    '45-49': '45_49',
    '50-54': '50_54',
    '55-59': '55_59',
    '60-64': '60_64',
    '65-69': '65_69',
    '70-74': '70_74',
    '75-79': '75_79',
    '80-84': '80_84',
    '85+': '85',
    'Nieustalony wiek': np.nan
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)

checking = df_raw_2023[column].unique()

#%%
"""Age"""

df_raw_2023['GroupAge2'] = df_raw_2023['GroupAge1']

df_raw_2023['GroupAge2'].unique()

column = 'GroupAge2'
mapping = {
    "07_12": '00_18',
    '13_18': '00_18',
    '19_24': '19_34',
    '25_29': '19_34',
    '30_34': '19_34',
    '35_39': '35_64',
    '40_44': '35_64',
    '45_49': '35_64',
    '50_54': '35_64',
    '55_59': '35_64',
    '60_64': '35_64',
    '65_69': '65',
    '70_74': '65',
    '75_79': '65',
    '80_84': '65',
    '85': '65'
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)

#%%
"""Płeć"""

df_raw_2023['Płeć'].unique()

df_raw_2023.rename(columns={'Płeć': 'Gender'}, inplace=True)

column = 'Gender'
mapping = {
    "Kobieta": '0',
    "Mężczyzna": '1'
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)

checking = df_raw_2023[column].unique()

#%%
"""Stan_cywilny"""

df_raw_2023['Stan_cywilny'].unique()

column = 'Stan_cywilny'
mapping = {
    "Brak danych/nieustalony": np.nan,
    "Kawaler/panna": 'Single',
    "Konkubent/konkubina": 'Cohabitant',
    "Żonaty/zamężna": 'Married',
    "Separowany/separowana": 'Separated',
    "Rozwiedziony/rozwiedziona": 'Divorced',
    "Wdowiec/wdowa": 'Widowed'
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'Marital'}, inplace=True)

checking = df_raw_2023['Marital'].unique()

#%%
"""Wykształcenie"""

df_raw_2023['Wykształcenie'].unique()

column = 'Wykształcenie'
mapping = {
    'Brak danych/nieustalone': np.nan,
    'Podstawowe niepełne': 'PrePrimary',
    'Podstawowe': 'Primary',
    'Gimnazjalne': 'Secondary',
    'Zasadnicze zawodowe': 'Vocational',
    'Średnie': 'Secondary',
    'Policealne': 'Secondary',
    'Wyższe': 'Higher'
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'Education'}, inplace=True)

checking = df_raw_2023['Education'].unique()

#%%
"""Informacje_o_pracy_i_nauce"""

df_raw_2023['Informacje_o_pracy_i_nauce'].unique()

column = 'Informacje_o_pracy_i_nauce'
mapping = {
    'Brak danych/nieustalono': np.nan,
    'Uczeń': 'Student',
    'Student': 'Student',
    'Rolnik': 'Agriculturalist',
    'Pracujący na własny rachunek/samodzielna działalność gospodarcza': 'Employed',
    'Praca stała': 'Employed',
    'Praca dorywcza': 'Employed',
    'Bezrobotny': 'Unemployed'
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'WorkInfo'}, inplace=True)

checking = df_raw_2023['WorkInfo'].unique()

#%%
"""Źródło_utrzymania"""

df_raw_2023['Źródło_utrzymania'].unique()

column = 'Źródło_utrzymania'
mapping = {
    'Brak danych/nieustalony': np.nan,
    'Na utrzymaniu innej osoby': 'Dependent',
    'Praca': 'Steady',
    'Emerytura': 'Benefits',
    'Renta': 'Benefits',
    'Zasiłek/alimenty': 'Benefits',
    'Nie ma stałego źródła utrzymania': 'NoSteady'
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'Income'}, inplace=True)

checking = df_raw_2023['Income'].unique()

#%%
"""###Czy_samobójstwo_zakończyło_się_zgonem"""

df_raw_2023['Czy_samobójstwo_zakończyło_się_zgonem'].unique()

column = 'Czy_samobójstwo_zakończyło_się_zgonem'

mapping = {
    'T': 1,
    'N': 0
}
df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'Fatal'}, inplace=True)

checking = df_raw_2023['Fatal'].unique()

#%%
"""###Miejsce_zamachu"""

df_raw_2023['Miejsce_zamachu'].unique()

column = 'Miejsce_zamachu'
mapping = {
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
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'Place'}, inplace=True)

checking = df_raw_2023['Place'].unique()

#%%
"""###Sposób_popełnienia"""

df_raw_2023['Sposób_popełnienia'].unique()

column = 'Sposób_popełnienia'
mapping = {
    'Rzucenie się pod pojazd w ruchu': 'Vehicle',
    'Rzucenie się z wysokości': 'Jumping',
    'Powieszenie się': 'Hanging',
    'Uszkodzenie układu krwionośnego': 'SelfHarm',
    'Zastrzelenie się/użycie broni palnej': 'Schooting',
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
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'Method'}, inplace=True)

checking = df_raw_2023['Method'].unique()

#%%
"""###Powód_zamachu"""

list(set(df_raw_2023['Powód_zamachu_*'].unique()) | set(df_raw_2023['Powód_zamachu_2'].unique()) | set(df_raw_2023['Powód_zamachu_3'].unique()) | set(df_raw_2023['Powód_zamachu_4'].unique()))

column = 'Powód_zamachu_*'
mapping = {
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

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'Context'}, inplace=True)

#%%
"""###Powód_zamachu_2"""

column = 'Powód_zamachu_2'
mapping = {
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

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'Context2'}, inplace=True)

#%%
"""###Powód_zamachu_3"""

column = 'Powód_zamachu_3'
mapping = {
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

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'Context3'}, inplace=True)

#%%
"""###Powód_zamachu_4"""

column = 'Powód_zamachu_4'
mapping = {
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

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'Context4'}, inplace=True)

list(set(df_raw_2023['Context'].unique()) | set(df_raw_2023['Context2'].unique()) | set(df_raw_2023['Context3'].unique()) | set(df_raw_2023['Context4'].unique()))

#%%
"""###Stan_świadomości"""

df_raw_2023['Stan_świadomości_*'].unique()

column = 'Stan_świadomości_*'
mapping = {
    'Brak danych/nieustalony': np.nan,
    'Trzeźwy(a)': 'Sober',
    'Pod wpływem alkoholu': 'Alco',
    'Pod wpływem zastępczych środków/substancji (dopalaczy)': 'OtherSub',
    'Pod wpływem leków': 'OtherSub',
    'Pod wpływem środków odurzających': 'OtherSub',
    'Pod wpływem alkoholu i zastępczych środków/substancji (dopalaczy)': 'AlcoOtherSub',
    'Pod wpływem alkoholu zastępczych środków/substancji (dopalaczy)': 'AlcoOtherSub',
    'Pod wpływem alkoholu i leków': 'AlcoOtherSub',
    'Pod wpływem alkoholu i środków odurzających': 'AlcoOtherSub',
    'Pod wpływem leków i środków odurzających': 'OtherSub',
    'Pod wpływem alkoholu, leków i środków odurzających': 'AlcoOtherSub'
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'Substance'}, inplace=True)

checking = df_raw_2023['Substance'].unique()

#%%
"""###Informacje_dotyczące_leczenia_z_powodu_alkoholizmu/narkomanii"""

df_raw_2023['Informacje_dotyczące_leczenia_z_powodu_alkoholizmu/narkomanii'].unique()

column = 'Informacje_dotyczące_leczenia_z_powodu_alkoholizmu/narkomanii'
mapping = {
    'Leczony(a) psychiatrycznie': np.nan,
    'Nadużywał(a) alkoholu': 'Alco',
    'Leczony(a) z powodu alkoholizmu': 'Alco',
    'Leczony(a) z powodu narkomanii': 'OtherSub',
    'Leczony(a) z powodu alkoholizmu i narkomanii': 'AlcoOtherSub'
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)
df_raw_2023.rename(columns={column: 'AbuseInfo'}, inplace=True)

df_raw_2023['AbuseInfo'].unique()

df_raw_2023['Informacje_dotyczące_stanu_zdrowia_*'].unique()

column = 'Informacje_dotyczące_stanu_zdrowia_*'
mapping = {
    'Brak danych/nieustalono': np.nan,
    'Nadużywał(a) alkoholu': 'Alco',
    'Leczony(a) psychiatrycznie': np.nan,
    'Leczony(a) z powodu alkoholizmu': 'Alco',
    'Choroba fizyczna': np.nan,
    'Trwałe kalectwo': np.nan,
    'Zatrzymany(a) w izbie wytrzeźwień': 'Alco',
    'Nadużywał(a) alkoholu i narkotyków': 'AlcoOtherSub',
    'Nadużywał(a) alkoholu i nakrotyków': 'AlcoOtherSub',
    'Leczony(a) z powodu narkomanii': 'OtherSub',
    'Używał dopalaczy i narkotyków': 'OtherSub',
    'Nadużywał(a) alkoholu i narkotykó': 'OtherSub',
    'Nadużywał(a) alkoholu, dopalaczy i narkotyków': 'AlcoOtherSub',
    'Nadużywał(a) alkoholu, narkotyków i dopalaczy': 'AlcoOtherSub',
    'Nadużywał(a) alkoholu i dopalaczy': 'AlcoOtherSub',
    'Używał dopalaczy': 'OtherSub',
    'Nadużywał(a) alkoholu, dopalaczy, narkotyków': 'AlcoOtherSub',
    'Leczony(a) psychiatrycznie, nadużywał(a) alkoholu': 'Alco',
}

df_raw_2023[column] = df_raw_2023[column].map(mapping)

df_raw_2023['AbuseInfo'] = df_raw_2023['AbuseInfo'].fillna(df_raw_2023['Informacje_dotyczące_stanu_zdrowia_*'])
df_raw_2023.drop(columns='Informacje_dotyczące_stanu_zdrowia_*', inplace=True)

#%%
"""##Powód zamachu - dummies"""

columns_to_drop = ['Context', 'Context2', 'Context3', 'Context4']
df_context_2023 = df_raw_2023[columns_to_drop]
df_raw_2023.drop(columns=columns_to_drop, inplace=True)

context_values = (
    set(df_context_2023['Context'].unique()) |
    set(df_context_2023['Context2'].unique()) |
    set(df_context_2023['Context3'].unique()) |
    set(df_context_2023['Context4'].unique())
)

for value in context_values:
    column_name = 'Context_' + str(value)
    df_context_2023[column_name] = df_context_2023.apply(lambda row: 1 if value in row.values else 0, axis=1)

columns_to_drop = ['Context', 'Context2', 'Context3', 'Context4', 'Context_nan']
df_context_2023.drop(columns=columns_to_drop, inplace=True)

#%%
"""##Liczba_powodów_zamachu"""

df_raw_2023['CountContext'] = df_context_2023.sum(axis=1)

#%%







#%%
"""# 2013_2022"""
csv_file_path = DATA_DIR / 'raw' / 'final_samobojstwa_2013_2022.csv'
df_raw_2013_2022 = read_csv_file(csv_file_path, low_memory = False)

#%%
"""##Zamiana nazw kolumn"""
df_raw_2013_2022.rename(columns={'ID samobójcy': 'ID'}, inplace=True)

# Zamiana spacji na podkreślenia w nazwach kolumn
df_raw_2013_2022.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

df_raw_2013_2022.rename(columns={'Data_raportu_[RRRRMM]': 'Date'}, inplace=True)

#%%
"""ID"""
non_numeric_ids = df_raw_2013_2022[df_raw_2013_2022['ID'].str.contains(r'\D', na=False)]
nan_rows = df_raw_2013_2022[df_raw_2013_2022['ID'].isna()]

#%%


#%%
"""Daty"""
df_raw_2013_2022['DateY'] = df_raw_2013_2022['Date'].apply(lambda x: x[0:4] if len(x) > 8 else x[:4])
df_raw_2013_2022['DateM'] = df_raw_2013_2022['Date'].apply(lambda x: x[5:7] if len(x) > 8 else x[4:6])

# Combine month and year into a single column in MM.YYYY format
df_raw_2013_2022['Date'] = df_raw_2013_2022['DateM'] + '.' + df_raw_2013_2022['DateY']


#%%
"""Mapowanie"""

"""Przedział_wiekowy"""

column = 'Przedział_wiekowy'
mapping = {
    0: np.nan,
    1: '07_12',
    2: '13_18',
    3: '19_24',
    4: '25_29',
    5: '30_34',
    6: '35_39',
    7: '40_44',
    8: '45_49',
    9: '50_54',
    10: '55_59',
    11: '60_64',
    12: '65_69',
    13: '70_74',
    14: '75_79',
    15: '80_84',
    16: '85'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'GroupAge1'}, inplace=True)

df_raw_2013_2022['GroupAge2']=df_raw_2013_2022['GroupAge1']

column = 'GroupAge2'
mapping = {
    "07_12": '00_18',
    '13_18': '00_18',
    '19_24': '19_34',
    '25_29': '19_34',
    '30_34': '19_34',
    '35_39': '35_64',
    '40_44': '35_64',
    '45_49': '35_64',
    '50_54': '35_64',
    '55_59': '35_64',
    '60_64': '35_64',
    '65_69': '65',
    '70_74': '65',
    '75_79': '65',
    '80_84': '65',
    '85': '65'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)

checking = df_raw_2013_2022['GroupAge2'].unique()

#%%
"""###Płeć"""

column = 'Płeć'
mapping = {
    0: np.nan,
    1: '0',
    2: '1'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Gender'}, inplace=True)

checking = df_raw_2013_2022['Gender'].unique()

#%%
"""###Stan_cywilny"""

column = 'Stan_cywilny'
mapping = {
    0: np.nan,
    1: 'Single',
    2: 'Cohabitant',
    3: 'Married',
    4: 'Separated',
    5: 'Divorced',
    6: 'Widowed'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Marital'}, inplace=True)

checking = df_raw_2013_2022['Marital'].unique()

#%%
"""###Wykształcenie"""

column = 'Wykształcenie'
mapping = {
    0: np.nan,
    1: 'PrePrimary',
    2: 'Primary',
    3: 'Secondary',
    4: 'Vocational',
    5: 'Secondary',
    6: 'Secondary',
    7: 'Higher'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Education'}, inplace=True)

checking = df_raw_2013_2022['Education'].unique()

#%%
"""###Informacje_o_pracy_i_nauce"""

column = 'Informacje_o_pracy_i_nauce'
mapping = {
    0: np.nan,
    1: 'Student',
    2: 'Student',
    3: 'Employed',
    4: 'Employed',
    5: 'Agriculturalist',
    6: 'Employed',
    7: 'Employed',
    8: 'Employed',
    9: 'Unemployed'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'WorkInfo'}, inplace=True)

checking = df_raw_2013_2022['WorkInfo'].unique()

#%%
"""###Źródło_utrzymania"""

column = 'Źródło_utrzymania'
mapping = {
    0: np.nan,
    1: 'Dependent',
    2: 'Steady',
    3: 'Benefits',
    4: 'NoSteady'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Income'}, inplace=True)

checking = df_raw_2013_2022['Income'].unique()

#%%
"""###Czy_samobójstwo_zakończyło_się_zgonem"""

column = 'Czy_samobójstwo_zakończyło_się_zgonem'

mapping = {
    0: np.nan,
    1: 1,
    2: 0
}
df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Fatal'}, inplace=True)

checking = df_raw_2013_2022['Fatal'].unique()

#%%
"""###Miejsce_zamachu"""

column = 'Miejsce_zamachu'
mapping = {
    0: np.nan,
    1: 'Road',
    2: 'UtilitySpaces',
    3: 'House',
    4: 'Railway',
    5: 'Forest',
    6: 'House',
    7: 'WaterRes',
    8: 'Work',
    9: 'Institution',
    10: 'Isolation',
    11: 'PoliceArmy',
    12: 'Institution',
    13: 'School',
    14: 'PoliceArmy',
    15: 'Other'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Place'}, inplace=True)

checking = df_raw_2013_2022['Place'].unique()

#%%
"""###Sposób_popełnienia"""

column = 'Sposób_popełnienia'
mapping = {
    0: np.nan,
    1: 'Vehicle',
    2: 'Jumping',
    3: 'Hanging',
    4: 'SelfHarm',
    5: 'Schooting',
    6: 'SelfHarm',
    7: 'SelfHarm',
    8: 'Drugs',
    9: 'Poisoning',
    10: 'Gas',
    11: 'Drugs',
    12: 'Poisoning',
    13: 'Drugs',
    14: 'Drugs',
    15: 'Drowning',
    16: 'SelfHarm',
    17: 'Other',
    18: 'Other'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Method'}, inplace=True)

checking = df_raw_2013_2022['Method'].unique()

#%%
"""###Powód_zamachu"""

column = 'Powód_zamachu'
mapping = {
    0: np.nan,
    1: 'HeartBreak',
    2: 'MentalHealth',
    3: 'FamilyConflict',
    4: 'HealthLoss',
    5: 'Finances',
    6: 'Finances',
    7: 'HealthLoss',
    8: 'SchoolWork',
    9: 'CloseDeath',
    10: 'Crime',
    11: 'Disability',
    12: 'Other',
    13: 'HealthLoss',
    14: 'HealthLoss',
    15: 'SchoolWork',
    16: 'Finances',
    17: 'SchoolWork',
    18: 'Other'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Context'}, inplace=True)

checking = df_raw_2013_2022['Context'].unique()

#%%
"""###Powód_zamachu_2"""

column = 'Powód_zamachu_2'
mapping = {
    0: np.nan,
    1: 'HeartBreak',
    2: 'MentalHealth',
    3: 'FamilyConflict',
    4: 'HealthLoss',
    5: 'Finances',
    6: 'Finances',
    7: 'HealthLoss',
    8: 'SchoolWork',
    9: 'CloseDeath',
    10: 'Crime',
    11: 'Disability',
    12: 'Other',
    13: 'HealthLoss',
    14: 'HealthLoss',
    15: 'SchoolWork',
    16: 'Finances',
    17: 'SchoolWork',
    18: 'Other'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Context2'}, inplace=True)

checking = df_raw_2013_2022['Context2'].unique()

#%%
"""###Powód_zamachu_3"""

column = 'Powód_zamachu_3'
mapping = {
    0: np.nan,
    1: 'HeartBreak',
    2: 'MentalHealth',
    3: 'FamilyConflict',
    4: 'HealthLoss',
    5: 'Finances',
    6: 'Finances',
    7: 'HealthLoss',
    8: 'SchoolWork',
    9: 'CloseDeath',
    10: 'Crime',
    11: 'Disability',
    12: 'Other',
    13: 'HealthLoss',
    14: 'HealthLoss',
    15: 'SchoolWork',
    16: 'Finances',
    17: 'SchoolWork',
    18: 'Other'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Context3'}, inplace=True)

checking = df_raw_2013_2022['Context3'].unique()

#%%
"""###Powód_zamachu_4"""

column = 'Powód_zamachu_4'
mapping = {
    0: np.nan,
    1: 'HeartBreak',
    2: 'MentalHealth',
    3: 'FamilyConflict',
    4: 'HealthLoss',
    5: 'Finances',
    6: 'Finances',
    7: 'HealthLoss',
    8: 'SchoolWork',
    9: 'CloseDeath',
    10: 'Crime',
    11: 'Disability',
    12: 'Other',
    13: 'HealthLoss',
    14: 'HealthLoss',
    15: 'SchoolWork',
    16: 'Finances',
    17: 'SchoolWork',
    18: 'Other'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Context4'}, inplace=True)

checking = df_raw_2013_2022['Context4'].unique()

#%%
"""###Stan_świadomości"""

column = 'Stan_świadomości'
mapping = {
    0: np.nan,
    1: 'Sober',
    2: 'Alco',
    3: 'OtherSub',
    4: 'OtherSub',
    5: 'OtherSub',
    6: 'OtherSub',
    7: 'AlcoOtherSub',
    8: 'AlcoOtherSub',
    9: 'AlcoOtherSub',
    10: 'AlcoOtherSub',
    11: 'AlcoOtherSub',
    12: 'OtherSub',
    13: 'OtherSub',
    14: 'OtherSub',
    15: 'OtherSub',
    16: 'OtherSub',
    17: 'AlcoOtherSub',
    18: 'AlcoOtherSub'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'Substance'}, inplace=True)

checking = df_raw_2013_2022['Substance'].unique()

#%%
"""###Informacje_o_używaniu_substancji"""

column = 'Informacje_o_używaniu_substancji'
mapping = {
    0: np.nan,
    1: 'Alco',
    2: 'OtherSub',
    3: 'OtherSub',
    4: 'AlcoOtherSub',
    5: 'OtherSub',
    6: 'AlcoOtherSub',
    7: 'AlcoOtherSub'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)
df_raw_2013_2022.rename(columns={column: 'AbuseInfo'}, inplace=True)

checking = df_raw_2013_2022['AbuseInfo'].unique()

#%%
"""###Informacje_dotyczące_leczenia_z_powodu_alkoholizmu/narkomanii"""

column = 'Informacje_dotyczące_leczenia_z_powodu_alkoholizmu/narkomanii'
mapping = {
    0: np.nan,
    1: 'Alco',
    2: 'OtherSub',
    3: 'AlcoOtherSub'
}

df_raw_2013_2022[column] = df_raw_2013_2022[column].map(mapping)

df_raw_2013_2022['Informacje_dotyczące_leczenia_z_powodu_alkoholizmu/narkomanii'].unique()

df_raw_2013_2022['AbuseInfo'] = df_raw_2013_2022['AbuseInfo'].fillna(df_raw_2013_2022['Informacje_dotyczące_leczenia_z_powodu_alkoholizmu/narkomanii'])
df_raw_2013_2022.drop(columns='Informacje_dotyczące_leczenia_z_powodu_alkoholizmu/narkomanii', inplace=True)

#%%
"""##Powód zamachu - dummies"""

columns_to_drop = ['Context', 'Context2', 'Context3', 'Context4']
df_context_2013_2022 = df_raw_2013_2022[columns_to_drop]
df_raw_2013_2022.drop(columns=columns_to_drop, inplace=True)

context_values = (
    set(df_context_2013_2022['Context'].unique()) |
    set(df_context_2013_2022['Context2'].unique()) |
    set(df_context_2013_2022['Context3'].unique()) |
    set(df_context_2013_2022['Context4'].unique())
)

for value in context_values:
    column_name = 'Context_' + str(value)
    df_context_2013_2022[column_name] = df_context_2013_2022.apply(lambda row: 1 if value in row.values else 0, axis=1)

columns_to_drop = ['Context', 'Context2', 'Context3', 'Context4', 'Context_nan']
df_context_2013_2022.drop(columns=columns_to_drop, inplace=True)

#%%
"""##Liczba_powodów_zamachu"""

df_raw_2013_2022['CountContext'] = df_context_2013_2022.sum(axis=1)

#%%






#%%
"""Merging"""

"""df_raw"""

common_columns = list(set(df_raw_2023.columns) & set(df_raw_2013_2022.columns))

# Wybierz wspólne kolumny z obu DataFrame'ów
df_raw_2023_selected = df_raw_2023[common_columns]
df_raw_2013_2022_selected = df_raw_2013_2022[common_columns]

df_raw = pd.concat([df_raw_2013_2022_selected, df_raw_2023_selected])

#%%
"""##df_context"""

df_context = pd.concat([df_context_2013_2022, df_context_2023])
df_context.fillna(0, inplace=True)
#%%
out_preped_suicides = pd.concat([df_raw, df_context], axis=1)

#%%
# Resetowanie indeksów i przypisanie nowych indeksów do kolumny ID
out_preped_suicides.reset_index(drop=True, inplace=True)
out_preped_suicides['ID'] = out_preped_suicides.index

#%%
"""#Zapis"""
file_name = 'mapped_data.csv'
write_to_csv(out_preped_suicides, output_file_path / file_name, index=False)