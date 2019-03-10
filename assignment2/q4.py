"""
Machine Learning (CSP 584 Assignment 2) question 4
Authored by Saransh Kumar (A20424637)
Environment: Python 3.7.1
"""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

spiral_df = pd.read_csv("~/Desktop/machine learning/assignments/Spiral.csv", index_col='id')


plt.figure()
axs = plt.gca()
axs.scatter(spiral_df['x'], spiral_df['y'])
axs.set_xlabel("X")
axs.set_ylabel("Y")
axs.set_title("Y vs X Scatterplot")

kmeans = KMeans(n_clusters=2, n_jobs=10, random_state=60616).fit(
    spiral_df.to_numpy()
)  # n_jobs = 10 to compute 10 iterations in parallel.

plt.figure()
axs = plt.gca()


color_map = ['yellow', 'blue']
for label in range(2):
    members = spiral_df[kmeans.labels_ == label]
    axs.scatter(members['x'], members['y'], c=color_map[label], label=f"Cluster id: {label}")

axs.legend()
axs.set_xlabel("X")
axs.set_ylabel("Y")
axs.set_title("Y vs X Scatterplot")
plt.show()

knn_ks = range(1,11)  # try with values 1 - 10

num_obs = spiral_df['x'].count()


'''
-----Ignore the commented part------
Trying with Knn Regressor to find nearest neighbors
knn_scores = np.zeros(20)
train = spiral_df[0:int(0.9*num_obs)]
test = spiral_df[90:100]
for k in knn_ks:
    kNNSpec = KNeighborsRegressor(n_neighbors=k, algorithm='brute', n_jobs=5)
    knn_fit = kNNSpec.fit(
        train['x'].to_numpy().reshape((len(train), 1)),
        train['y'].to_numpy().reshape((len(train), 1))
    )
    knn_scores[k-1] = max(knn_fit.score(
        test['x'].to_numpy().reshape((len(test), 1)),
        test['y'].to_numpy().reshape((len(test), 1))
    ), 0)

print(knn_scores)
'''

for k in [3]:  # can be tried with other values of nearest neighbors.
    adjacency_mat = np.zeros((num_obs, num_obs))
    degree_mat = np.zeros((num_obs, num_obs))
    kNNSpec = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    fit = kNNSpec.fit(spiral_df.to_numpy())
    dist, nbrs = fit.kneighbors(spiral_df.to_numpy())

    for index in range(num_obs):
        for j in range(k):
            if index <= nbrs[index][j]:
                adjacency_mat[index][nbrs[index][j]] = math.exp(-1 * dist[index][j])
                adjacency_mat[nbrs[index][j]][index] = adjacency_mat[index][nbrs[index][j]]  # ensuring symmetry of
                # adjacency matrix

                degree_mat[index][index] += math.exp(-1 * dist[index][j])
                if index != nbrs[index][j]:  # to avoid double counting in degree matrix
                    degree_mat[nbrs[index][j]][nbrs[index][j]] += math.exp(-1 * dist[index][j])

    print("Adjacency Matrix:")
    print(adjacency_mat)
    print("\n\nDegree Matrix:")
    print(degree_mat)

    lmatrix = degree_mat - adjacency_mat  # computing Laplacian Matrix

    print("\n\nLaplacian Matrix:")
    print(lmatrix)
    evals, evecs = np.linalg.eigh(lmatrix)

    plt.figure()
    axs = plt.gca()
    axs.scatter(list(range(0, 9)), evals[0:9, ])
    axs.set_title(f"Sequence plot with {k} neighbors")
    axs.set_xlabel("Sequence")
    axs.set_ylabel("Eigenvalues")

plt.show()
Z = evecs[:, [0, 1]]  # first two eigenvectors
plt.figure()
axs = plt.gca()

axs.scatter(Z[[0]], Z[[1]])  # plotting first two eigenvectors (biplot)
axs.set_title(f"Z[1] vs Z[0]")
axs.set_xlabel("Z[0]")
axs.set_ylabel("Z[1]")

kmeans_spectral = KMeans(n_clusters=2, random_state=60616).fit(Z)  # spectral clustering: K-means on first two
# eigenvectors

plt.figure()
axs = plt.gca()
for label in range(2):
    members = spiral_df[kmeans_spectral.labels_ == label]
    axs.scatter(members['x'], members['y'], c=color_map[label], label=f"Cluster id: {label}")
axs.set_xlabel('x')
axs.set_ylabel('y')
axs.set_title("Y vs X Scatterplot post spectral clustering")
axs.legend()
plt.show()
