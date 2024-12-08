import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Step 1: Load and Prepare the Iris Dataset
def load_and_preprocess_data():
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # True labels (not used in clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Step 2: Perform Fuzzy C-Means Clustering
def perform_fcm(X, n_clusters):
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X.T, c=n_clusters, m=2.0, error=0.005, maxiter=1000, init=None
    )
    labels = np.argmax(u, axis=0)  # Convert fuzzy memberships to hard labels
    return labels, cntr

# Step 3: Evaluate the Clustering
def evaluate_clustering(X, labels, centers):
    # Quantization error: Sum of squared distances to the nearest cluster center
    quantization_error = sum(np.sum((X[labels == i] - center) ** 2)
                             for i, center in enumerate(centers))
    # Intra-cluster distances: Sum of squared distances within each cluster
    intra_cluster_distances = [np.sum((X[labels == i] - center) ** 2)
                               for i, center in enumerate(centers)]
    # Inter-cluster distance: Minimum distance between cluster centers
    inter_cluster_distances = np.min(
        [np.linalg.norm(center1 - center2) 
         for i, center1 in enumerate(centers) 
         for j, center2 in enumerate(centers) if i != j])

    # Print evaluation results
    print(f"Quantization Error: {quantization_error:.4f}")
    print(f"Intra-cluster Distances: {intra_cluster_distances}")
    print(f"Inter-cluster Distance: {inter_cluster_distances:.4f}")
    return quantization_error, intra_cluster_distances, inter_cluster_distances

# Step 4: Visualize the Clustering Result
def visualize_clustering(X, labels, centers):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centers')
    plt.title("Clustering Result with Fuzzy C-Means")
    plt.legend()
    plt.show()

# Step 5: Main Function
def main():
    # Load and preprocess the dataset
    X, y = load_and_preprocess_data()
    
    # Number of clusters
    n_clusters = 3

    # Perform Fuzzy C-Means clustering
    labels, centers = perform_fcm(X, n_clusters)
    
    # Evaluate the clustering
    evaluate_clustering(X, labels, centers)
    
    # Visualize the clustering result
    visualize_clustering(X, labels, centers)

if __name__ == "__main__":
    main()
