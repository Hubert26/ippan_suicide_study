"""
Machine Learning Utility Functions for Model Evaluation and Metrics Calculation.

This module includes functions for calculating various performance metrics,
feature importances, and model validation techniques for machine learning models.

This file contains utility functions for machine learning model evaluation and metrics calculation.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


# %%
def run_stratified_kfold(
    model, X: pd.DataFrame, Y: pd.Series, n_splits: int = 5, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform Stratified K-Fold Cross-Validation for any model and return validation metrics.

    Parameters:
    -----------
    model : object
        A machine learning model with `fit` and `predict` methods.
    X : pd.DataFrame
        Feature dataset of shape (n_samples, n_features).
    Y : pd.Series
        Target labels of shape (n_samples,).
    n_splits : int, default=5
        Number of splits for Stratified K-Fold cross-validation.
    random_state : int, default=42
        Random state for reproducibility.

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        - A DataFrame containing mean validation metrics across all folds.
        - A DataFrame containing metrics for each fold with additional fold-specific information.
    """

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # List to store fold metrics
    skf_validation_results = []

    # Iterate through each fold
    for fold, (train_indices, test_indices) in enumerate(skf.split(X, Y), 1):
        # Split the dataset into training and testing sets for the current fold
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = Y.iloc[train_indices], Y.iloc[test_indices]

        # Train the model
        model.fit(X_train, y_train)

        # Validate the model on the test set
        validation_results = validate_model(model, X_test, y_test)

        # Add fold-specific information
        validation_results["fold"] = fold
        validation_results["train_size"] = len(y_train)
        validation_results["test_size"] = len(y_test)

        # Append results for this fold
        skf_validation_results.append(validation_results)

    # Combine all fold validation results into a single DataFrame
    skf_validation_results_df = pd.concat(skf_validation_results, ignore_index=True)

    # Calculate mean metrics across all folds
    mean_metrics = (
        skf_validation_results_df.drop(columns=["fold", "train_size", "test_size"])
        .mean()
        .to_frame()
        .T
    )

    return mean_metrics, skf_validation_results_df


def validate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Validate a machine learning model on test data and return evaluation metrics.

    Parameters:
    -----------
    model : object
        A trained machine learning model with `predict` and `predict_proba` methods.
        If the model uses weights, ensure the weights are correctly applied during training.
    X_test : array-like of shape (n_samples, n_features)
        Test dataset features.
    y_test : array-like of shape (n_samples,)
        True labels for the test dataset.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the calculated metrics: recall, accuracy, precision, and F1-score.

    Raises:
    -------
    ValueError
        If `X_test` or `y_test` are empty or if their lengths do not match.
    """

    # Validate input: Ensure X_test and y_test are not empty
    if X_test is None or y_test is None or len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("X_test and y_test must not be empty.")

    # Validate input: Ensure X_test and y_test have matching lengths
    if len(X_test) != len(y_test):
        raise ValueError(
            f"Inconsistent sample sizes: X_test has {len(X_test)} samples, "
            f"but y_test has {len(y_test)} samples."
        )

    # Generate predictions and predicted probabilities
    try:
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # Assumes binary classification
    except AttributeError as e:
        raise AttributeError(
            "Model must implement both `predict` and `predict_proba` methods. "
            f"Error: {str(e)}"
        )

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Construct a DataFrame for model evaluation results
    metrics_df = pd.DataFrame(
        {
            "recall": [recall],
            "accuracy": [accuracy],
            "precision": [precision],
            "f1_score": [f1],
        }
    )

    return metrics_df


def compute_feature_importances(model, X: pd.DataFrame, sort=True) -> pd.DataFrame:
    """
    Compute feature importances for a given model and feature set X.

    Parameters:
    - model: Trained model object (e.g., DecisionTreeClassifier, RandomForestClassifier) that supports feature importance extraction.
    - X: pd.DataFrame, Feature matrix (must match the model's training features).
    - sort: bool, Whether to sort the features by importance in descending order (default: True).

    Returns:
    - importances_df: pd.DataFrame, DataFrame containing feature names and their respective importance scores.
                      Sorted by importance if sort=True.
    """

    # Check if the model has a feature_importances_ attribute
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"The model {type(model).__name__} does not have feature importances."
        )

    # Extract feature importances
    importances = model.feature_importances_

    # Sort feature importances and corresponding feature names if required
    if sort:
        sorted_indices = importances.argsort()[::-1]
        data = {
            "feature": [X.columns[index] for index in sorted_indices],
            "MDI": [importances[index] for index in sorted_indices],
        }
    else:
        data = {"feature": X.columns, "MDI": importances}

    # Create a DataFrame of the importances
    importances_df = pd.DataFrame(data)

    return importances_df


def compute_mean_decrease_accuracy(
    model, X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """
    Computes Mean Decrease Accuracy (MDA) for feature importance using permutation.

    Parameters:
    - model: The machine learning model to evaluate (must have a `fit` method).
    - X: DataFrame with feature data.
    - y: Series or array with target data.

    Returns:
    - Dictionary with features as keys and their Mean Decrease Accuracy as values.
    """

    # Calculate the baseline accuracy of the model
    baseline_accuracy = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()

    feature_importances = {}

    # Iterate through each feature to compute its Mean Decrease Accuracy
    for feature in X.columns:
        # Create a permuted copy of the dataset for the current feature
        X_permuted = X.copy()
        X_permuted[feature] = np.random.permutation(X[feature].values)

        # Calculate the accuracy of the model on the permuted dataset
        permuted_accuracy = cross_val_score(
            model, X_permuted, y, cv=5, scoring="accuracy"
        ).mean()

        # Compute the Mean Decrease Accuracy for the current feature
        feature_importances[feature] = baseline_accuracy - permuted_accuracy

    return pd.DataFrame(list(feature_importances.items()), columns=["feature", "MDA"])


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "accuracy",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computes Permutation Importance for feature importance evaluation.

    Parameters:
    ----------
    model : object
        A trained machine learning model (must have a `predict` method).
    X : pd.DataFrame
        Feature dataset of shape (n_samples, n_features).
    y : pd.Series
        Target labels of shape (n_samples,).
    scoring : str, default="accuracy"
        Metric to evaluate the model's performance. Compatible with sklearn's scoring metrics.
    n_repeats : int, default=10
        Number of repetitions for computing permutation importance.
    random_state : int, default=42
        Random state for reproducibility.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with features and their corresponding permutation importance scores.
    """

    # Compute permutation importance
    perm_importance = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state
    )

    # Convert results into a DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm_importance.importances_mean,
            "importance_std": perm_importance.importances_std,
        }
    )

    # Sort features by importance
    importance_df = importance_df.sort_values(
        by="importance_mean", ascending=False
    ).reset_index(drop=True)

    return importance_df


if __name__ == "__main__":
    pass
