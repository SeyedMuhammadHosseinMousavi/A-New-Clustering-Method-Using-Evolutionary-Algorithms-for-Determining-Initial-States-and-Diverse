# A-New-Clustering-Method-Using-Evolutionary-Algorithms-for-Determining-Initial-States-and-Diverse
A New Clustering Method Using Evolutionary Algorithms for Determining Initial States, and Diverse Pairwise Distances for Clustering
"""
# A New Clustering Method Using Evolutionary Algorithms for Determining Initial States, and Diverse Pairwise Distances for Clustering

This repository implements the methodology proposed in the paper:
**"A New Clustering Method Using Evolutionary Algorithms for Determining Initial States, and Diverse Pairwise Distances for Clustering"**
by **Seyed Muhammad Hossain Mousavi**.

---

## Abstract

This repository explores a novel clustering method that leverages **Differential Evolution (DE)** and **Particle Swarm Optimization (PSO)** to effectively initialize cluster centers. The method addresses the common problem of selecting initial cluster centers in traditional clustering methods. Additionally, the clustering results are refined using diverse pairwise distance metrics, including:

- Euclidean Distance
- City-block Distance
- Chebyshev Distance

The method has been validated on benchmark datasets and compared against traditional clustering techniques such as K-Means, Fuzzy C-Means, Gaussian Mixture Models, and Self-Organizing Maps, demonstrating promising results.

---

## Features

1. **Differential Evolution (DE) Initialization**: Utilizes the DE algorithm to determine optimal initial cluster centers.
2. **Particle Swarm Optimization (PSO) Initialization**: Employs the PSO algorithm for alternative initialization.
3. **Diverse Distance Metrics**: Refines cluster allocations using multiple pairwise distance measures.
4. **Benchmark Dataset Validation**: Includes results for datasets such as:
   - Fisher-Iris
   - Ionosphere
   - User Knowledge Modeling
   - Breast Cancer
   - Blood Transfusion
5. **Comparison with Standard Methods**: Compares performance with K-Means, Fuzzy C-Means, GMM, and SOM.

---

## Installation

Clone the repository:
```bash
git clone https://github.com/username/repo-name.git
