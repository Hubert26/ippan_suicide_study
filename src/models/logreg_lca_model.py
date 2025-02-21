# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:57:12 2024

@author: huber
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from config.config import DATA_DIR, RESULTS_DIR
from src.utils.utils import read_csv, read_excel, write_excel

# ================================================================================
# Data reading
# ================================================================================
excel_file_path = DATA_DIR / "processed" / "lca_group_results.xlsx"
lca_classes = read_excel(excel_file_path, sheet_name="Group_AG", index_col=None)

csv_file_path = DATA_DIR / "processed" / "group_set.csv"
group_set = read_csv(csv_file_path)

# select columns
group_column = "Group_AG"
target_column = "Fatal"
classification_features = ["LCA_Group_AG_class"]

group_set = group_set[["ID", group_column, target_column]]
lca_classes = lca_classes[["ID"] + classification_features]

# merging
df_data = group_set.merge(lca_classes, on="ID", how="left")
df_data = df_data.drop(columns=["ID"], inplace=False)

groups = sorted(list(set(group_set[group_column])))


# ================================================================================
# statsmodels
# ================================================================================
# Lists for storing results
results_data = []
validation_data = []

for group in groups:
    try:
        # Filter data for the current group
        df_group = df_data[df_data[group_column] == group].copy()
        df_group = df_group.drop(columns=[group_column])  # Remove the group column
        df_group[classification_features] = df_group[classification_features].astype(
            "category"
        )

        # Prepare the target variable (y) and features (X)
        y = df_group[target_column].astype(int)  # Convert the target to integers
        X = df_group.drop(
            columns=[target_column], errors="ignore"
        )  # Drop the target column from features
        X = pd.get_dummies(X, drop_first=True)  # Apply one-hot encoding
        X = X.loc[:, X.nunique() > 1]  # Remove constant columns (zero variance)

        # Check for minimum requirements for the model
        if X.empty or len(y.unique()) < 2:
            print(
                f"Skipping group {group}: insufficient data or no variance in target."
            )
            continue

        # Ensure all features are numeric
        X = X.astype(int)

        # Compute class weights for balanced handling
        class_weights = compute_class_weight(
            class_weight="balanced", classes=y.unique(), y=y
        )
        weight_mapping = dict(zip(y.unique(), class_weights))
        df_group["weights"] = y.map(weight_mapping)  # Map weights to each observation

        # Add a constant column for the intercept
        X = sm.add_constant(X)

        # Train the model using sm.Logit
        model = sm.Logit(y, X)
        result = model.fit(disp=0)

        # Store parameter results (coefficients, p-values, and standard errors)
        for param in result.params.index:
            results_data.append(
                {
                    "group": group,
                    "param": param,
                    "coeff": result.params[param],
                    "pvalues": result.pvalues[param],
                    "bse": result.bse[param],  # Standard error of the coefficient
                }
            )

        # Calculate evaluation metrics (precision, recall, etc.)
        y_pred = (result.predict(X) >= 0.5).astype(int)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        # Store general model metrics for validation
        validation_data.append(
            {
                "group": group,
                "llf": result.llf,  # Log-Likelihood
                "AIC": result.aic,  # Akaike Information Criterion
                "BIC": result.bic,  # Bayesian Information Criterion
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

    except Exception as e:
        print(f"Error processing group {group}: {e}")

# Convert results to DataFrame
results_df = pd.DataFrame(results_data)

# Pivot parameter results into a wide format
if not results_df.empty:
    wide_results_df = results_df.pivot(
        index="group", columns="param", values=["coeff", "pvalues", "bse"]
    )
    # Flatten the column index created by pivot
    wide_results_df.columns = [
        f"{metric}_{param}" for metric, param in wide_results_df.columns
    ]

    # Remove the "LCA_Group_AG_class_" prefix from column names
    wide_results_df.columns = [
        col.replace("LCA_Group_AG_class_", "") for col in wide_results_df.columns
    ]

    wide_results_df.reset_index(inplace=True)
else:
    wide_results_df = pd.DataFrame()  # Empty DataFrame if no results are available

# Convert validation metrics to DataFrame
validation_df = pd.DataFrame(validation_data)

write_excel(
    file_path=RESULTS_DIR / "logreg_model_results_lca.xlsx",
    data=wide_results_df,
    sheet_name="coefficients",
    mode="w",
    index=False,
)

write_excel(
    file_path=RESULTS_DIR / "logreg_model_results_lca.xlsx",
    data=validation_df,
    sheet_name="validation_metrics",
    mode="a",
    index=False,
)
