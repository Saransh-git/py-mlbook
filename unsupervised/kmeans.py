"""
Machine Learning (CSP 584 Assignment 2) question 3
Authored by Saransh Kumar (A20424637)
Environment: Python 3.7.1
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

cars_df: DataFrame = pd.read_csv(
    "~/Desktop/machine learning/assignments/cars.csv", index_col='Model'
)[['Horsepower', 'Weight']]  # using Horsepower and Weight features only.

# print(cars_df.columns)

MAX_CLUSTERS = 15
elbow_val = np.zeros(MAX_CLUSTERS)  # initializing array to hold respective elbow values.
silhoulette = np.zeros(MAX_CLUSTERS)  # initializing array to hold respective silhouette values.
k_clusters = range(1,MAX_CLUSTERS + 1)
cars_df_arr = cars_df.to_numpy(dtype='float32')

for cluster_size in k_clusters:
    kmeans = KMeans(
        n_clusters=cluster_size, random_state=60616,
        n_jobs=10  # runs 10 iterations in parallel to faster compute the best cluster allocation.
    ).fit(cars_df_arr)

    wcss = np.zeros(cluster_size)  # initializing array to hold wcss values for respective clusters.
    num_cluster_members = np.zeros(cluster_size)  # initializing array to hold number of members for respective clusters
    for index, row in enumerate(cars_df_arr):
        cluster_label = kmeans.labels_[index]
        dist = row - kmeans.cluster_centers_[cluster_label]
        wcss[cluster_label] += dist.dot(dist)
        num_cluster_members[cluster_label] += 1
    '''
    print(f"WCSS of clusters sized {cluster_size}: {wcss}")
    print(f"Number of members in clusters sized {cluster_size}: {num_cluster_members}")
    '''
    for w, n in zip(wcss, num_cluster_members):
        elbow_val[cluster_size - 1] += w/n

    if cluster_size in [1]:
        silhoulette[cluster_size - 1] = np.NaN  # silhouette value undefined for cluster sized 1
    else:
        silhoulette[cluster_size - 1] = silhouette_score(cars_df_arr, kmeans.labels_)

cluster_metric_df = pd.DataFrame(
    data={
        'num_clusters': list(k_clusters),
        'elbow_val': elbow_val,
        'silhouette': silhoulette
    }
)

elbow_val_slope = np.zeros(MAX_CLUSTERS)  # initializing array to hold elbow value slopes.
elbow_val_acceleration = np.zeros(MAX_CLUSTERS)  # initializing array to hold slope acceleration.
elbow_val_slope[0] = np.NaN
elbow_val_acceleration[0] = np.NaN
elbow_val_acceleration[1] = np.NaN
for i in range(1, MAX_CLUSTERS):
    elbow_val_slope[i] = elbow_val[i] - elbow_val[i - 1]

for i in range(2, MAX_CLUSTERS):
    elbow_val_acceleration[i] = elbow_val_slope[i] - elbow_val_slope[i - 1]

cluster_metric_df['slope'] = elbow_val_slope
cluster_metric_df['accelaration'] = elbow_val_acceleration

print(cluster_metric_df)

plt.figure()
axs = plt.gca()
axs.plot(list(k_clusters), elbow_val, 'o--', linestyle='solid')
axs.set_title(f"Elbow value vs Cluster-size")
axs.set_xlabel("Cluster size")
axs.set_ylabel("Elbow value")
axs.set_xticks(list(k_clusters))

plt.figure()
axs = plt.gca()
axs.plot(list(k_clusters), silhoulette, 'o--', linestyle='solid')
axs.set_title(f"Silhouette vs Cluster-size")
axs.set_xlabel("Cluster size")
axs.set_ylabel("Silhouette score")
axs.set_xticks(list(k_clusters) + [0])
plt.show()

