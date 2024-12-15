# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:23:24 2024

@author: huber
"""
# ================================================
# Imports
# ================================================


import pandas as pd
import numpy as np
from pathlib import Path
import sys

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import _tree, export_graphviz
import graphviz

src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from utils.dataframe_utils import read_csv_file, write_to_excel
from utils.file_utils import create_directory
from utils.ml_utils import model_validation, compute_feature_importances, mean_decrease_accuracy
from utils.string_utils import group_string_by_prefix

np.random.seed(42)


#%%
# ================================================
# Constants and Configuration
# ================================================


AGE_GROUP = ['19_34']  # Age groups for filtering data
GENDER = [False, True]  # Gender options for filtering data
YEAR = [2014]  # Year(s) for filtering data
RISK_THRESHOLDS = [0.5, 0.9, 0.95, 0.99]  # Thresholds for suicide risk classification

# Define the current working directory and relevant paths
current_working_directory = Path.cwd()
grandparent_directory = current_working_directory.parent.parent

# Define the path to the input file
input_file_path = grandparent_directory / 'data' / 'encoded' / 'encoded_data.csv'

# Generate the folder name based on AGE_GROUP and GENDER constants
folder_name = "AGE_[" + ''.join(filter(str.isdigit, AGE_GROUP[0][:3])) + "_" + ''.join(filter(str.isdigit, AGE_GROUP[-1][-3:])) + "]" + '_' + "GENDER_" + str(GENDER)

# Define the path to the output file
output_file_path = grandparent_directory / 'results' / 'dtc'


#%%
# ================================================
# Function Definitions
# ================================================


def compute_node_depths(tree):
    """
    Compute the depth of each node in the decision tree.

    Args:
    tree (DecisionTree): The tree object from which node depths will be calculated.

    Returns:
    np.array: Array of node depths for each node in the decision tree.
    """
    def get_depth(node_id, current_depth):
        depths[node_id] = current_depth
        if tree.children_left[node_id] != -1:
            get_depth(tree.children_left[node_id], current_depth + 1)
        if tree.children_right[node_id] != -1:
            get_depth(tree.children_right[node_id], current_depth + 1)

    depths = np.zeros(tree.node_count, dtype=np.int32)
    get_depth(0, 0)
    return depths

def get_feature_name(tree, node_id, feature_names):
    """
    Retrieve the feature name used to split at a specific node.

    Args:
    tree (DecisionTree): The decision tree object.
    node_id (int): The ID of the node.
    feature_names (list): List of feature names in the dataset.

    Returns:
    str: Name of the feature used to split at the node. Returns 'Leaf' if the node is a leaf.
    """
    if tree.feature[node_id] != _tree.TREE_UNDEFINED:
        feature_index = tree.feature[node_id]
        return feature_names[feature_index]
    else:
        return 'Leaf'

def extract_tree_node_info(tree, feature_names, class_weights):
    """
    Extracts information from each node of the decision tree and stores it in a list of dictionaries.

    Args:
    tree (DecisionTree): The tree object to extract node information from.
    feature_names (list): The list of feature names in the dataset.
    class_weights (array): The class weights for unweighted class counts.

    Returns:
    pd.DataFrame: DataFrame containing detailed information about each node in the tree.
    """
    # Compute node depths
    node_depths = compute_node_depths(tree)

    # List to store node information
    nodes_info = []

    # Iterate over all nodes in the tree
    for node_id in range(tree.node_count):
        # Retrieve Gini impurity, number of samples, and class distribution
        gini = tree.impurity[node_id]
        samples = tree.n_node_samples[node_id]
        values = tree.value[node_id][0]  # Weighted class counts

        # Determine if the node is a leaf node
        is_leaf = (tree.children_left[node_id] == -1) and (tree.children_right[node_id] == -1)

        # Retrieve the feature and threshold if not a leaf
        feature = get_feature_name(tree, node_id, feature_names)
        threshold = tree.threshold[node_id] if not is_leaf else -1

        # Create a dictionary to hold node information
        node_info = {
            'Node ID': node_id,
            'Gini': gini,
            'Samples': samples,
            'Values (Weighted)': values,
            'Sum of Values': sum(values),
            'Class Counts (Unweighted)': np.round(values / class_weights),
            'Depth': node_depths[node_id],
            'Is Leaf': is_leaf,
            'Feature': feature,
            'Threshold': threshold,
            'Children Left': tree.children_left[node_id],
            'Children Right': tree.children_right[node_id]
        }

        # Add the node's information to the list
        nodes_info.append(node_info)

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(nodes_info)

def get_leaf_paths(tree, feature_names=None):
    """
    Retrieve the decision paths leading to each leaf in the decision tree.

    Args:
    tree (DecisionTreeClassifier): The trained decision tree model.
    feature_names (list, optional): List of feature names. If None, feature indices will be used.

    Returns:
    pd.DataFrame: A DataFrame containing information about each leaf node, 
                  including node ID, Gini impurity, number of samples, class, 
                  and the conditions leading to that leaf.

    This function works by recursively traversing the decision tree and collecting 
    information about the leaf nodes. For each leaf, it stores the conditions 
    (i.e., feature splits) leading to that leaf and additional details such as 
    Gini impurity and the number of samples at that node.
    """
    
    tree_ = tree.tree_  # Access the underlying tree structure
    feature = tree_.feature  # Feature used to split at each node
    threshold = tree_.threshold  # Threshold used for the split
    children_left = tree_.children_left  # Left child of each node
    children_right = tree_.children_right  # Right child of each node
    impurity = tree_.impurity  # Gini impurity at each node
    n_node_samples = tree_.n_node_samples  # Number of samples at each node
    value = tree_.value  # Class distribution at each node

    leaf_info = []  # List to store leaf node details

    def recurse(node, path_conditions):
        """
        Recursively traverse the tree to collect information about leaf nodes.

        Args:
        node (int): The current node being explored.
        path_conditions (dict): Dictionary holding the conditions (splits) 
                                that lead to the current node.
        """
        # If this is a leaf node (no children)
        if children_left[node] == children_right[node]:
            class_counts = value[node][0]  # Class distribution in the leaf
            leaf_class = np.argmax(class_counts)  # Class with the highest count
            leaf_details = {
                'node_id': node,
                'gini': impurity[node],
                'samples': n_node_samples[node],
                '(1-gini)*samples': (1 - impurity[node]) * n_node_samples[node],
                'leaf_class': leaf_class
            }
            # Add the conditions leading to this leaf
            leaf_details.update(path_conditions)
            leaf_info.append(leaf_details)
        else:
            # If not a leaf, recursively explore left and right children
            if feature_names is not None:
                feature_name = feature_names[feature[node]]  # Use feature name if available
            else:
                feature_name = feature[node]  # Use feature index if names are not provided

            # Recurse down the left child (where feature <= threshold)
            left_path_conditions = path_conditions.copy()
            left_path_conditions[feature_name] = 0  # Add condition for left path
            recurse(children_left[node], left_path_conditions)

            # Recurse down the right child (where feature > threshold)
            right_path_conditions = path_conditions.copy()
            right_path_conditions[feature_name] = 1  # Add condition for right path
            recurse(children_right[node], right_path_conditions)

    # Initialize the conditions for the root node (all features set to NaN initially)
    initial_conditions = {feature: np.nan for feature in feature_names}

    # Start the recursion from the root node (node 0)
    recurse(0, initial_conditions)

    # Return the collected leaf information as a DataFrame
    return pd.DataFrame(leaf_info)


#%%
# ================================================
# Data Initialization
# ================================================


# Load the encoded data from the CSV file
df_encoded = read_csv_file(input_file_path, delimiter=',', low_memory=False, index_col=None)



# ================================================
# Data Processing
# ================================================


# Filter data based on age and gender criteria
# Construct column names for age groups based on the AGE_GROUP list
age_columns = [f'GroupAge_{group}' for group in AGE_GROUP]

# Apply filters for age and gender
age_filter = df_encoded[age_columns].any(axis=1)  # True if any age column is True
gender_filter = df_encoded['Gender'].isin(GENDER)  # Filter for specified gender(s)
df_data = df_encoded[age_filter & gender_filter]  # Combine filters to get the desired data

# Drop unnecessary columns from the filtered data
df_data.drop(['GroupAge_19_34', 'GroupAge_35_64', 'GroupAge_65'], inplace=True,	 axis=1, errors='ignore')
df_data.drop(['Gender'], inplace=True, axis=1, errors='ignore')

# Prepare the target variable 'Y' and features 'X'
Y = df_data['Fatal']  # Target variable (fatal or not)
X = df_data.drop('Fatal', axis=1, errors='ignore')  # Features excluding the target variable

# Prepare the list of feature names
feature_names = X.columns.tolist()



# ================================================
# Cross Validation
# ================================================


# Compute class weights to handle imbalanced classes
class_weights = compute_class_weight('balanced', classes=np.unique(Y), y=Y)
class_weight_dict = dict(zip(np.unique(Y), class_weights))  # Convert to a dictionary

# Initialize Decision Tree Classifier
dtc = DecisionTreeClassifier(max_depth=None, min_samples_split=10, min_samples_leaf=10)

# Set up Stratified K-Fold cross-validation (to ensure balanced splits)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize DataFrame to store cross-validation results
cv_results = pd.DataFrame()
    
# Loop through each fold of the cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(X, Y), 1):
    # Split the data into training and test sets based on the indices
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Calculate sample weights for training, based on class distribution
    sample_weights = np.array([class_weight_dict[label] for label in y_train])

    # Fit the decision tree model on the training data with class sample weights
    dtc.fit(X_train, y_train, sample_weight=sample_weights)

    # Validate the model on the test data and calculate performance metrics
    model_results = model_validation(dtc, X_test, y_test, RISK_THRESHOLDS)
    
    # Add additional information about the fold and model
    model_results['features'] = X_train.shape[1]  # Number of features used in training
    model_results['train_size'] = len(y_train)  # Number of samples in the training set
    model_results['class_0_train_size'] = (y_train == 0).sum()  # Class 0 samples in training set
    model_results['class_1_train_size'] = (y_train == 1).sum()  # Class 1 samples in training set
    model_results['test_size'] = len(y_test)  # Number of samples in the test set
    model_results['class_weight_0'] = class_weight_dict[0]  # Class weight for class 0
    model_results['class_weight_1'] = class_weight_dict[1]  # Class weight for class 1
    model_results['fold'] = str(fold)  # Fold number for tracking

    # Append the fold's results to the cross-validation results DataFrame
    cv_results = pd.concat([cv_results, model_results], ignore_index=True)

# Calculate the mean of numeric columns across all folds to evaluate the final model
cv_mean_result = cv_results.select_dtypes(include=np.number).mean()

# Add a label 'mean' to indicate that this row contains the average values across all folds
cv_mean_result['fold'] = 'mean'

cv_mean_result_df = cv_mean_result.to_frame().T  # Convert Series to DataFrame


# ================================================
# Final Model
# ================================================


# Initialize the DecisionTreeClassifier with specific parameters
dtc = DecisionTreeClassifier(
    class_weight=class_weight_dict,  # Set the class weights for imbalanced data
    random_state=42,                 # Ensure reproducibility of the results
    max_depth=None,                  # No limit on the depth of the tree
    min_samples_split=10,            # Minimum samples required to split a node
    min_samples_leaf=10              # Minimum samples required at a leaf node
)

# Fit the final model using the entire dataset
dtc.fit(X, Y)

# Validate the final model using the entire dataset (X and Y)
final_result = model_validation(dtc, X, Y, RISK_THRESHOLDS)

# Add additional model metadata to the result for tracking and analysis
final_result['fold'] = 'final'                          # Indicate this result is from the final model
final_result['features'] = X.shape[1]                   # Number of features used in the model
final_result['train_size'] = len(Y)                     # Total number of training samples
final_result['class_0_train_size'] = (Y == 0).sum()     # Number of samples in class 0
final_result['class_1_train_size'] = (Y == 1).sum()     # Number of samples in class 1
final_result['test_size'] = len(Y)                      # Size of the test set (same as the training set here)
final_result['class_weight_0'] = class_weight_dict[0]   # Class weight for class 0
final_result['class_weight_1'] = class_weight_dict[1]   # Class weight for class 1

model_validation_df = pd.concat([cv_mean_result_df, final_result], ignore_index=True)



# ================================================
# Feature Validation
# ================================================


feature_importances_df = compute_feature_importances(dtc, X)

feature_mda_df = mean_decrease_accuracy(dtc, X, Y)

# Merge Mean Decrease Impurity with Mean Decrease Accuracy on 'feature'
feature_validation_df = feature_importances_df.merge(feature_mda_df, on='feature', how='left')

# Group the columns by their prefixes (e.g., "Context" or "Health")
grouped_columns = group_string_by_prefix(feature_names)

# Dictionaries to store the summed importances for each categorical variable
category_importances = {}  # For Mean Decrease Impurity (MDI)
category_mda = {}  # For Mean Decrease Accuracy (MDA)

# Loop through each category and its associated columns
for category, columns in grouped_columns.items():
    if columns:
        # Calculate the sum of 'mean_decrease_impurity' for all columns within the category
        category_importances[category] = feature_importances_df[
            feature_importances_df['feature'].isin(columns)
        ]['mean_decrease_impurity'].sum()
        
        # Calculate the sum of 'mean_decrease_accuracy' for all columns within the category
        category_mda[category] = feature_mda_df[
            feature_mda_df['feature'].isin(columns)
        ]['mean_decrease_accuracy'].sum()

# Create a DataFrame for the categories and their summed 'mean_decrease_impurity'
category_importances_df = pd.DataFrame(
    list(category_importances.items()), 
    columns=['category', 'mean_decrease_impurity']
)

# Create a DataFrame for the categories and their summed 'mean_decrease_accuracy'
category_mda_df = pd.DataFrame(
    list(category_mda.items()), 
    columns=['category', 'mean_decrease_accuracy']
)

# Merge the two DataFrames (mean_decrease_impurity and mean_decrease_accuracy) on 'category'
category_validation_df = category_importances_df.merge(
    category_mda_df, 
    on='category', 
    how='left'
)



#%%
# ================================================
# Tree results
# ================================================

# Nodes Info
tree = dtc.tree_
nodes_info_df = extract_tree_node_info(tree, feature_names, class_weights)

# Path Info
path_info_df = get_leaf_paths(dtc, feature_names)

# Export the decision tree to DOT format
try:
    dot_data = export_graphviz(
        dtc,  # Decision Tree Classifier object
        out_file=None,  # Return DOT data as a string
        feature_names=feature_names,  # List of feature names
        filled=True,  # Fill nodes with colors representing the class distribution
        rounded=True,  # Round the corners of the nodes
        class_names={False: 'NotFatal', True: 'Fatal'},  # Map class indices to class names
        special_characters=True,  # Handle special characters in node labels
        impurity=True,  # Show the impurity (Gini index) at each node
        node_ids=True,  # Display node IDs
        rotate=True  # Rotate the graph to fit better
    )
except ValueError as ve:
    raise ValueError(f"Failed to export decision tree to DOT format: {ve}")

#%%
# ================================================
# Save results
# ================================================

# Create directory if doesn't exist
create_directory(output_file_path / folder_name)

# Write model results to excel file
write_to_excel(dataframe=nodes_info_df, file_path=output_file_path / folder_name / '.xlsx', sheet_name='Nodes_Info', mode='w', index=False )
write_to_excel(dataframe=path_info_df, file_path=output_file_path / folder_name / '.xlsx', sheet_name='Path_Info', mode='a', index=False )
write_to_excel(dataframe=feature_validation_df, file_path=output_file_path / folder_name / '.xlsx', sheet_name='Feature_Validation', mode='a', index=False )
write_to_excel(dataframe=category_validation_df, file_path=output_file_path / folder_name / '.xlsx', sheet_name='Category_Validation', mode='a', index=False )


# Write the DOT data to a DOT file
with open(output_file_path / folder_name / "tree.dot", 'w') as file:
    file.write(dot_data)

# Create a graph from the DOT data and render it to an SVG file
try:
    graph = graphviz.Source(dot_data)
    graph.render(filename=output_file_path / folder_name / "tree", format="svg")
except Exception as e:
    raise RuntimeError(f"Failed to create or save SVG from DOT data: {e}")

