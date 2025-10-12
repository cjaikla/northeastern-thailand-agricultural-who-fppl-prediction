from sklearn.tree import _tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score

# Function to traverse the tree and find paths with Gini = 0 and all samples in class 0
def find_paths_with_gini_zero(tree, feature_names):
    tree_ = tree.tree_
    paths = []

    def traverse(node, path, sample_count):
        # Check if it's a leaf node
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            # Check if Gini is 0 and all samples are classed as 0
            if tree_.impurity[node] == 0 and tree_.value[node][0][1] == 0:  # All samples in class 0
                paths.append((path, sample_count))
            return

        # Traverse left child
        left_child = tree_.children_left[node]
        if left_child != _tree.TREE_UNDEFINED:
            traverse(
                left_child,
                path + [f"{feature_names[tree_.feature[node]]} <= {tree_.threshold[node]:.5f}"],
                tree_.n_node_samples[left_child]
            )

        # Traverse right child
        right_child = tree_.children_right[node]
        if right_child != _tree.TREE_UNDEFINED:
            traverse(
                right_child,
                path + [f"{feature_names[tree_.feature[node]]} > {tree_.threshold[node]:.5f}"],
                tree_.n_node_samples[right_child]
            )

    # Start traversal from the root node
    traverse(0, [], tree_.n_node_samples[0])
    return paths

# Function to evaluate metrics for a given path
def evaluate_path(path, data, target_column):
    # Start with the full dataset
    filtered_data = data.copy()

    # Apply each condition in the path to filter the dataset
    for condition in path:
        # feature, operator, threshold = condition.split(' ')
        parts = condition.rsplit(' ', 2)  # Split from the right into 3 parts
        feature = parts[0]
        operator = parts[1]
        threshold = float(parts[2])
        
        threshold = float(threshold)
        if operator == '<=':
            filtered_data = filtered_data[filtered_data[feature] <= threshold]
        elif operator == '>':
            filtered_data = filtered_data[filtered_data[feature] > threshold]

    # Get the indices of samples satisfying the path
    path_indices = filtered_data.index

    # Create predictions for the entire dataset
    y_true = data[target_column]
    y_pred = [0 if i in path_indices else 1 for i in data.index]  # Predict 0 for path samples, 1 otherwise

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    return accuracy, precision, recall

