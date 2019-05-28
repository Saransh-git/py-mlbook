import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


diabetes_df = pd.read_csv(
    "/Users/saransh/Desktop/machine learning/assignments/ChicagoDiabetes.csv"
)  # ingest the data


diabetes_df.dropna(
    subset=[
        'Crude Rate 2000', 'Crude Rate 2001', 'Crude Rate 2002', 'Crude Rate 2003', 'Crude Rate 2004',
        'Crude Rate 2005', 'Crude Rate 2006', 'Crude Rate 2007', 'Crude Rate 2008', 'Crude Rate 2009',
        'Crude Rate 2010', 'Crude Rate 2011'
    ], inplace=True
)  # drop the missing values, override the current dataframe

population = pd.DataFrame()  # stores population
hosp = pd.DataFrame()  # stores hospitalization numbers
for year in range(2000, 2012):
    population[f"{year}"] = diabetes_df[f"Hospitalizations {year}"] / diabetes_df[f"Crude Rate {year}"] * 10000

for year in range(2000, 2012):
    hosp[f"{year}"] = diabetes_df[f"Hospitalizations {year}"]

population_by_year = population.cumsum().iloc[population.shape[0] -1]  # Annual population


hosp_by_year = hosp.cumsum().iloc[hosp.shape[0] - 1, ]  # Annual hospitalization numbers

crud_rates = diabetes_df[[
        'Crude Rate 2000', 'Crude Rate 2001', 'Crude Rate 2002', 'Crude Rate 2003', 'Crude Rate 2004',
        'Crude Rate 2005', 'Crude Rate 2006', 'Crude Rate 2007', 'Crude Rate 2008', 'Crude Rate 2009',
        'Crude Rate 2010', 'Crude Rate 2011'
    ]]  # required variables for PCA

nvar = crud_rates.shape[1]  # number of features = 12

pca = PCA(svd_solver='full')  # initialize PCA
pca.fit(crud_rates)  # fit into PCA

explained_variance_ratio_ = pca.explained_variance_ratio_
index = list(range(nvar))
axs: Axes = plt.gca()

axs: Axes = plt.gca()
axs.plot(index, explained_variance_ratio_, 'bo-')  # plot the explained variance ratio
axs.axhline(y=1/nvar, linestyle='dashed', color='red')  # plot the reference line
axs.set_xticks(index)
axs.set_xlabel("Index")
axs.set_ylabel("Explained Variance Ratio")
axs.set_title("Explained Variance ratio to Principal components")
plt.grid(True)
plt.show()

# Plot cumulative explained variance ratio
cum_sum_var_ratio = np.cumsum(explained_variance_ratio_)  # Cumulative sum of explained variance ratio
axs: Axes = plt.gca()
axs.plot(index, cum_sum_var_ratio, 'bo-')
axs.set_xticks(index)
axs.set_xlabel("Index")
axs.axhline(y=0.95, color='red', linestyle="dashed")  # reference of 95%
axs.set_ylabel("Cumulative percentage variance explained")
axs.set_title("Total variance percentage explained by principal components")
plt.grid(True)
plt.show()

print(f"Cumulative variance ratio explained by the major principal components: {cum_sum_var_ratio[1]}")

transformed_features: DataFrame = pd.DataFrame(PCA(n_components=2, svd_solver="full").fit_transform(crud_rates))
transformed_features.set_index(diabetes_df['Community'], inplace=True)
MAX_CLUSTERS = 10
elbow_vals = np.zeros(MAX_CLUSTERS-1)  # need index from 0 to 8 dentoing respective values for cluster size 2 to 10
silhouette = np.zeros(MAX_CLUSTERS-1)  # need index from 0 to 8 dentoing respective values for cluster size 2 to 10
cluster_index = range(2, MAX_CLUSTERS + 1)

for cluster_size in cluster_index:  # goes from 2 to 10
    wcss = np.zeros(cluster_size)
    num_members = np.zeros(cluster_size)
    kmeans_ = KMeans(n_clusters=cluster_size, random_state=20190405)
    kmeans_.fit(transformed_features)

    for index in range(transformed_features.shape[0]):
        label = kmeans_.labels_[index]
        diff = transformed_features.iloc[index, :] - kmeans_.cluster_centers_[label]
        wcss[label] += diff.dot(diff)
        num_members[label] += 1

    for w, n in zip(wcss, num_members):
        elbow_vals[cluster_size - 2] += w/n  # flll the elbow vals

    silhouette[cluster_size - 2] = silhouette_score(transformed_features, kmeans_.labels_)  # fill silhouette

# Plot elbow values vs Cluster size
axs: Axes = plt.gca()
axs.plot(cluster_index, elbow_vals, 'bo--')
axs.set_xticks(cluster_index)
axs.set_xlabel("Cluster size")
axs.set_ylabel("Elbow value")
axs.set_title("Elbow value vs Cluster size")
axs.grid(True)
plt.show()


# Plot Silhouette vs Cluster size
axs: Axes = plt.gca()
axs.plot(cluster_index, silhouette, 'bo--')
axs.set_xticks(cluster_index)
axs.set_xlabel("Cluster size")
axs.set_ylabel("Silhouette score")
axs.set_title("Silhouette score vs Cluster size")
axs.grid(True)
plt.show()

kmeans_ = KMeans(n_clusters=4, random_state=20190405)  # optimum cluster size of 4
kmeans_.fit(transformed_features)

transformed_features['cluster_labels'] = kmeans_.labels_

groups_ = transformed_features.groupby('cluster_labels')  # group by cluster labels


def plot_crud_hospitalization_rate_for_cluster_members(members, cluster_name):
    """
    Plots annual crude rate for different clusters
    """
    axs: Axes = plt.gca()
    clust_diabetes = diabetes_df.loc[diabetes_df['Community'].isin(members.reset_index()['Community'])] # filter
    # for communities in this cluster
    clust_pop = pd.DataFrame()
    clus_hosp = pd.DataFrame()
    for year in range(2000, 2012):
        clust_pop[f"{year}"] = \
            clust_diabetes[f"Hospitalizations {year}"] / clust_diabetes[f"Crude Rate {year}"] * 10000
        clus_hosp[f"{year}"] = clust_diabetes[f"Hospitalizations {year}"]

    crud_rate = clus_hosp.cumsum().iloc[clust_pop.shape[0] - 1] / clust_pop.cumsum().iloc[clust_pop.shape[0] - 1] \
                * 10000
    axs.plot(range(2000, 2012), crud_rate, label=f"Cluster {cluster_name}", marker='o')


annual_crud_rate = hosp_by_year/population_by_year * 10000  # Annual crude rate for Chicago
print("Annual crud hospitalization rates:")
print(annual_crud_rate)

for grp_name, grp_members in groups_:
    print(f"Communities in group {grp_name}: {grp_members.reset_index()['Community'].tolist()}")
    plot_crud_hospitalization_rate_for_cluster_members(grp_members, grp_name)  # plot annual crud rates for
    # different clusters

axs: Axes = plt.gca()
axs.plot(range(2000, 2012),
             hosp_by_year/population_by_year * 10000,
             label="annual", marker='o')  # plot the city annual crude rate
axs.legend()
axs.grid(True)
axs.set_title("Crud Hospitalizations rate vs year")
axs.set_ylabel("Crud Hospitalizations rate")
axs.set_xlabel("Year")
axs.set_xticks(range(2000, 2012))
plt.show()
