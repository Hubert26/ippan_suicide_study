import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from config.config import (
    DATA_DIR,
    MOMENT_OF_SUICIDE_FEATURES,
    RESULTS_DIR,
    SOCIO_DEMOGRAPHIC_FEATURES,
)
from src.models.ml_utils import (
    compute_mean_decrease_accuracy,
    compute_permutation_importance,
    run_stratified_kfold,
)
from src.utils.utils import (
    read_csv,
)

# Read encoded data
csv_file_path = DATA_DIR / "processed" / "encoded_data.csv"
if not csv_file_path.exists():
    raise FileNotFoundError(f"File not found: {csv_file_path}")

df_raw = read_csv(csv_file_path)

# Read group set
csv_file_path = DATA_DIR / "processed" / "group_set.csv"
if not csv_file_path.exists():
    raise FileNotFoundError(f"File not found: {csv_file_path}")
df_groups = read_csv(csv_file_path)


# Groups
group_columns = ["Group_A", "Group_AG"]

for group_column in group_columns:
    df_group_column = df_raw.merge(df_groups[["ID", group_column]], on="ID", how="left")
    # Features
    features_dict = {
        "SOCIO_DEMOGRAPHIC_FEATURES": SOCIO_DEMOGRAPHIC_FEATURES,
        "MOMENT_OF_SUICIDE_FEATURES": MOMENT_OF_SUICIDE_FEATURES,
    }

    for feature_group, features in features_dict.items():
        # Select columns based on features
        columns_to_aggregate = [
            column
            for column in df_group_column.columns
            if any(column.startswith(feature) for feature in features)
        ]
        if not columns_to_aggregate:
            raise ValueError(f"No matching columns found for features: {features}")

        df_group_features = df_group_column[columns_to_aggregate + [group_column]]

        group_values = sorted(list(set(df_group_features[group_column])))

        for group_value in group_values:
            df_group = df_group_features[
                df_group_features[group_column] == group_value
            ].copy()

            target_column = "Fatal"
            classification_features = [
                col for col in columns_to_aggregate if col != target_column
            ]

            # Prepare the target variable 'Y' and features 'X'
            Y = df_group[target_column]
            X = df_group[classification_features]

            # Prepare the list of feature names
            feature_names = X.columns.tolist()

            # Compute class weights based on the entire dataset to handle inbalance in classes
            class_weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(Y), y=Y
            )
            class_weights_dict = {
                cls: weight for cls, weight in zip(np.unique(Y), class_weights)
            }

            # Initialize model
            model = RandomForestClassifier(
                max_depth=None, min_samples_split=10, min_samples_leaf=10
            )
            param_grid = {
                "n_estimators": 100,
                "max_features": "sqrt",
                "max_depth": None,
                "min_samples_split": 10,
                "min_samples_leaf": 10,
            }

            # Perform stratified K-Fold validation
            skf_final, skf_results = run_stratified_kfold(model, X, Y, n_splits=5)

            # Compute permutation importance
            perm_importance = compute_permutation_importance(model, X, Y)

            # Compute mean decrease accuracy
            mean_decrease_accuracy = compute_mean_decrease_accuracy(model, X, Y)

            # Specify the output Excel file name
            output_file = f"rfc_{group_column}_{feature_group}_{group_value}.xlsx"

            # Write the DataFrames to separate sheets in an Excel file
            with pd.ExcelWriter(
                RESULTS_DIR / "rfc_model_results" / output_file, engine="xlsxwriter"
            ) as writer:
                skf_final.to_excel(writer, sheet_name="r_final", index=True)
                skf_results.to_excel(writer, sheet_name="r_skf", index=True)
                perm_importance.to_excel(writer, sheet_name="PI", index=False)
                mean_decrease_accuracy.to_excel(writer, sheet_name="MDA", index=False)
