import pandas as pd
import struct
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import pairwise_distances


def evaluate_clustering(labels):
    """
    Evaluate the clustering result by calculating the ratio of intra-cluster distance
    to inter-cluster distance.

    This function asserts that the number of unique clusters found is exactly 4.
    If not, an AssertionError is raised.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data.
    labels : array-like, shape (n_samples,)
        The cluster labels for each sample.

    Returns:
    float
        The ratio of intra-cluster distance to inter-cluster distance.
        Returns float('inf') if inter-cluster distance is zero.

    Raises:
    AssertionError
        If the number of unique clusters is not 4.
    """

    possible_files = ["Clustering_Monday.xlsx", "Clustering_Thursday.xlsx", "Clustering_Friday.xlsx","training_set.xlsx"]
    found_file = None
    
    # 查找存在的文件
    for file_name in possible_files:
        try:
            # 尝试读取文件，如果存在就用这个
            data = pd.read_excel(file_name)
            found_file = file_name
            break
        except FileNotFoundError:
            continue
    
    if not found_file:
        raise FileNotFoundError("Can not find the dataset，please make sure you have downloaded one of the following datasets：Clustering_Monday.xlsx, Clustering_Thursday.xlsx, Clustering_Friday.xlsx")
    
    # 加载数据
    X = data[['Grade', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
    X = X.values  # 转换为numpy数组


    unique_labels = np.unique(labels)
    num_unique_labels = len(unique_labels)

    # Assertion: Check if the number of clusters is exactly 4
    assert num_unique_labels == 4, \
        f"Evaluation Error: Expected 4 clusters, but found {num_unique_labels} unique clusters. " \
        f"Please ensure your clustering algorithm is configured to produce exactly 4 clusters for a fair evaluation."

    # Calculate intra-cluster distances
    intra_distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            intra_distance = np.mean(pairwise_distances(cluster_points))
            intra_distances.append(intra_distance)
        elif len(cluster_points) == 1:
            intra_distances.append(0.0)
            
    if not intra_distances:
        # This case should ideally not be reached if num_unique_labels is 4
        # and labels correctly map to points in X.
        # However, as a safeguard:
        print("Evaluation Warning: No valid intra-cluster distances could be calculated, "
              "though 4 unique labels were found. Check data and label consistency.")
        return float('inf')

    avg_intra_distance = np.mean(intra_distances)

    # Calculate inter-cluster distances
    inter_distances = []
    for i in range(num_unique_labels): # This will be 4 iterations
        for j in range(i + 1, num_unique_labels):
            cluster_i_points = X[labels == unique_labels[i]]
            cluster_j_points = X[labels == unique_labels[j]]
            
            if len(cluster_i_points) > 0 and len(cluster_j_points) > 0:
                inter_distance = np.mean(pairwise_distances(cluster_i_points, cluster_j_points))
                inter_distances.append(inter_distance)

    if not inter_distances:
        # This implies that even with 4 unique labels, valid pairs of populated clusters
        # could not be formed for inter-distance calculation.
        print("Evaluation Warning: No valid inter-cluster distances could be calculated, "
              "though 4 unique labels were found. Check data and label consistency.")
        return float('inf')

    avg_inter_distance = np.mean(inter_distances)

    if avg_inter_distance == 0:
        ratio = float('inf')
        print("Evaluation Warning: Average inter-cluster distance is zero.")
    else:
        ratio = avg_intra_distance / avg_inter_distance

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Print key metric
    print("--------------Show this to TA------------------")
    print(f"Execution Time: {current_time} - Ratio: {ratio:.4f}")
    print("--------------Show this to TA-----------------")

    return ratio


def evaluate_classification(predicted_labels, true_bin_path):
    """
    Calculate classification accuracy by comparing predicted labels with true labels

    Args:
        predicted_labels (list/np.array): Model's output predictions
        true_bin_path (str): Path to binary file containing true labels

    Returns:
        float: Accuracy percentage (0-100)
    """
    # Read true labels from binary file
    true_labels = []
    with open(true_bin_path, 'rb') as bin_file:
        while True:
            byte_data = bin_file.read(4)  # 4 bytes per integer
            if not byte_data:
                break
            true_labels.append(struct.unpack('i', byte_data)[0])

    # Convert to numpy arrays for vectorized operations
    true_array = np.array(true_labels)
    pred_array = np.array(predicted_labels)

    # Validate lengths match
    if len(true_array) != len(pred_array):
        raise ValueError(f"Length mismatch: {len(true_array)} true vs {len(pred_array)} predicted")

    # Calculate accuracy
    matches = np.sum(true_array == pred_array)
    accuracy = (matches / len(true_array)) * 100

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Execution time: {current_time} - Accuracy: {accuracy:.2f}%")

    return accuracy
