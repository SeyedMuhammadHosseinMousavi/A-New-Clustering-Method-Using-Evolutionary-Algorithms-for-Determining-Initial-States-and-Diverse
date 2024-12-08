import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt

# Step 1: Load and Prepare the Iris Dataset
def load_and_preprocess_data():
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Labels (not used in clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Step 2: Define PSO
class PSO:
    def __init__(self, n_clusters, n_particles, n_iterations, X):
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize particle positions (random cluster centers) and velocities
        self.positions = np.random.rand(n_particles, n_clusters, self.n_features)
        self.velocities = np.random.rand(n_particles, n_clusters, self.n_features) * 0.1
        self.personal_best_positions = np.copy(self.positions)
        self.global_best_position = None
        self.personal_best_scores = np.full(n_particles, np.inf)
        self.global_best_score = np.inf
        self.cost_history = []

    def fitness(self, cluster_centers):
        # Assign points to nearest cluster center
        labels = pairwise_distances_argmin(self.X, cluster_centers)
        # Compute intra-cluster distance (sum of squared distances)
        score = sum(np.sum((self.X[labels == i] - center) ** 2)
                    for i, center in enumerate(cluster_centers))
        return score

    def optimize(self):
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                score = self.fitness(self.positions[i])
                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            # Update velocities and positions
            inertia = 0.5
            cognitive = 1.5
            social = 1.5
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = cognitive * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = social * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = inertia * self.velocities[i] + cognitive_component + social_component
                self.positions[i] += self.velocities[i]
            
            self.cost_history.append(self.global_best_score)
            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best Score: {self.global_best_score}")
        
        return self.global_best_position, self.cost_history

# Step 3: Clustering with PSO-generated Centers
def clustering_with_pso(X, n_clusters, n_particles, n_iterations):
    pso = PSO(n_clusters, n_particles, n_iterations, X)
    best_centers, cost_history = pso.optimize()
    labels = pairwise_distances_argmin(X, best_centers)
    return labels, best_centers, cost_history

# Step 4: Evaluate the Clustering
def evaluate_clustering(X, labels, centers):
    quantization_error = sum(np.sum((X[labels == i] - center) ** 2)
                             for i, center in enumerate(centers))
    intra_cluster_distances = [np.sum((X[labels == i] - center) ** 2) 
                               for i, center in enumerate(centers)]
    inter_cluster_distances = np.min(
        [np.linalg.norm(center1 - center2) 
         for i, center1 in enumerate(centers) 
         for j, center2 in enumerate(centers) if i != j])
    print(f"Quantization Error: {quantization_error:.4f}")
    print(f"Intra-cluster Distances: {intra_cluster_distances}")
    print(f"Inter-cluster Distance: {inter_cluster_distances:.4f}")
    return quantization_error, intra_cluster_distances, inter_cluster_distances

# Step 5: Visualize the Clustering Result
def visualize_results(X, labels, centers, cost_history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Clustering result
    axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.7)
    axes[0].scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centers')
    axes[0].set_title("Clustering Result with PSO")
    axes[0].legend()

    # PSO iteration cost
    axes[1].plot(range(1, len(cost_history) + 1), cost_history, marker='o')
    axes[1].set_title("PSO Iteration Cost")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Cost (Fitness)")

    plt.tight_layout()
    plt.show()

# Step 6: Main Function
def main():
    X, y = load_and_preprocess_data()
    n_clusters = 3
    n_particles = 100
    n_iterations = 300

    labels, centers, cost_history = clustering_with_pso(X, n_clusters, n_particles, n_iterations)
    evaluate_clustering(X, labels, centers)
    visualize_results(X, labels, centers, cost_history)

if __name__ == "__main__":
    main()
