import pandas as pd
import numpy as np
from pathlib import Path
import sys
from dotenv import load_dotenv
import os

# %%
# Load environment variables from the .env file
load_dotenv()

# Convert DATA_DIR to a Path object
DATA_DIR = Path(os.getenv("DATA_DIR"))

# Read CSV File
csv_file_path = DATA_DIR / "encoded" / "encoded_final_set.csv"
try:
    df_raw = pd.read_csv(csv_file_path, delimiter=",", low_memory=False)
except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found.")
    sys.exit(1)

# %%
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Definicja siatki parametrów do RandomForestClassifier (opcjonalnie)
param_grid = {
    "n_estimators": 100,
    "max_features": "sqrt",
    "max_depth": None,
    "min_samples_split": 10,
    "min_samples_leaf": 10,
}

# Stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=2)
all_models = []
class_weights_all = []
all_results = pd.DataFrame()

for fold, (train_index, test_index) in enumerate(skf.split(X, Y), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Oblicz wagi klas
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    class_weights_all.append(class_weights_dict)

    # Inicjalizacja modelu RandomForestClassifier
    rf = RandomForestClassifier(
        random_state=42, **param_grid, class_weight=class_weights_dict
    )

    # Trenowanie modelu na zestawie treningowym
    rf.fit(X_train, y_train)

    # Dodanie do listy wytrenowanych modeli
    all_models.append(rf)

    # Walidacja modelu na danych testowych
    model_results = model_validation(rf, X_test, y_test, risk_thresholds)

    model_results["features"] = X_train.shape[1]
    model_results["train_size"] = len(y_train)
    model_results["class_0_train_size"] = (y_train == 0).sum()
    model_results["class_1_train_size"] = (y_train == 1).sum()
    model_results["test_size"] = len(y_test)
    model_results["class_weight_0"] = class_weight_dict[0]
    model_results["class_weight_1"] = class_weight_dict[1]
    model_results["fold"] = str(fold)

    # Dodanie wyników do ramki danych
    all_results = pd.concat([all_results, model_results], ignore_index=True)

    # Obliczenie średnich dla każdej kolumny
mean_values = all_results.select_dtypes(include=np.number).mean()
# Dodanie średnich jako nowego wiersza
mean_values["fold"] = "mean"  # Oznaczenie wiersza ze średnimi

all_results = pd.concat([all_results, pd.DataFrame(mean_values).T], ignore_index=True)

# Uśrednienie wag klas
avg_class_weights = {}
for class_id in range(len(np.unique(Y))):
    avg_weight = np.mean([weights[class_id] for weights in class_weights_all])
    avg_class_weights[class_id] = avg_weight

# Połączenie wszystkich modeli w jeden model
final_rf = RandomForestClassifier(
    random_state=42, **param_grid, class_weight=avg_class_weights
)

# Łączenie drzew z poszczególnych modeli
combined_estimators = []
for model in all_models:
    for tree in model.estimators_:
        combined_estimators.append(tree)

final_rf.estimators_ = combined_estimators

# Ostateczne dopasowanie modelu do całego zestawu danych
final_rf.fit(X, Y)

rf_results = model_validation(rf, X, Y, risk_thresholds)
