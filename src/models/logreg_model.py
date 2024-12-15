# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:57:12 2024

@author: huber
"""

import pandas as pd
import numpy as np


from utils.dataframe_utils import read_csv_file, filter_dataframe
from utils.file_utils import create_directory
from config import *


# ================================================================================
# Data reading
# ================================================================================
csv_file_path = RESULTS_DIR / "poLCA" / "Group_AG.csv"
lca_classes = read_csv_file(
    csv_file_path, delimiter=",", low_memory=False, index_col=None
)

csv_file_path = DATA_DIR / "encoded" / "encoded_data.csv"
df_encoded = read_csv_file(
    csv_file_path, delimiter=",", low_memory=False, index_col=None
)

# merging
df_encoded = df_encoded.merge(lca_classes, on="ID", how="left")

groups = sorted(list(set(df_encoded["Group_AG"])))


# ================================================================================
# statsmodels
# ================================================================================
import statsmodels.api as sm

results_data = []

for group in groups:
    try:
        # Select Group from Group_AG
        group_data = filter_dataframe(df_encoded, Group_AG=group)

        group_data["Predicted_Class_Group_AG"] = group_data[
            "Predicted_Class_Group_AG"
        ].astype("category")

        # Selects columns
        X = pd.get_dummies(group_data["Predicted_Class_Group_AG"], drop_first=True)
        y = group_data["Fatal"]  # Zmienna zależna

        y = y.astype(int)
        X = X.astype(int)

        X = sm.add_constant(X)
        logreg_model = sm.Logit(y, X)
        result = logreg_model.fit(disp=0)

        # Wyodrębnij istotne informacje z modelu
        for param, coeff in result.params.items():
            results_data.append(
                {
                    "Group": group,
                    "Variable": param,
                    "Coefficient": coeff,
                    "P-value": result.pvalues[param],
                    "Standard Error": result.bse[param],
                    "Log-Likelihood": result.llf,
                    "AIC": result.aic,
                    "BIC": result.bic,
                }
            )
    except Exception as e:
        print(f"Error processing group {group}: {e}")

    # Tworzenie DataFrame z wynikami
    statsmodels_results_df = pd.DataFrame(results_data)


# ================================================================================
# sklearn
# ================================================================================
from sklearn.linear_model import LogisticRegression

results_data = []

for group in groups:
    # Select Group from Group_AG
    group_data = filter_dataframe(df_encoded, Group_AG=group)
    group_data["Predicted_Class_Group_AG"] = group_data[
        "Predicted_Class_Group_AG"
    ].astype("category")

    # Select columns
    X = group_data[["Predicted_Class_Group_AG"]]  # Zmienna niezależna
    y = group_data["Fatal"]  # Zmienna zależna

    X = pd.get_dummies(group_data["Predicted_Class_Group_AG"], drop_first=True)
    # Zakładamy, że 'Fatal' ma wartości 'True'/'False', przekształcamy na int
    y = y.astype(int)
    X = X.astype(int)

    # Używamy modelu regresji logistycznej
    logreg_model = LogisticRegression(
        max_iter=1000
    )  # Zwiększamy max_iter w razie potrzeby
    logreg_model.fit(X, y)

    log_likelihood = np.sum(np.log(logreg_model.predict_proba(X)[np.arange(len(y)), y]))
    # Przechowywanie wyników
    for col, coef in zip(X.columns, logreg_model.coef_[0]):
        results_data.append(
            {
                "Group": group,
                "Variable": col,
                "Coefficient": coef,
                "Intercept": logreg_model.intercept_[0],
                "Log-Likelihood": log_likelihood,
                "AIC": 2 * (X.shape[1] + 1) - 2 * log_likelihood,  # Aproksymacja AIC
                "BIC": np.log(len(y)) * (X.shape[1] + 1) - 2 * log_likelihood,
            }
        )

# Tworzenie DataFrame z wynikami
sklearn_results_df = pd.DataFrame(results_data)


# Saveing
file_path = RESULTS_DIR / "logreg"
create_directory(file_path)
file_name = "logreg_results_ref.xlsx"
# write_to_excel(dataframe=statsmodels_results_df, file_path= file_path / file_name, sheet_name='statsmodel', mode='w', index=False )
# write_to_excel(dataframe=sklearn_results_df, file_path= file_path / file_name, sheet_name='sklearn', mode='a', index=False )
