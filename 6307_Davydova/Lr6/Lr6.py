# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
from matplotlib import gridspec
from sklearn.cluster import KMeans, DBSCAN, OPTICS, cluster_optics_dbscan
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('CC GENERAL.csv').iloc[:, 1:].dropna()

# Проведем кластеризацию методов k-средних
# Преимуществом алгоритма являются скорость и простота реализации. К
# недостаткам можно отнести неопределенность выбора начальных центров кластеров, а также то, что число кластеров
# должно быть задано изначально, что может потребовать некоторой априорной информации об исходных данных.

k_means = KMeans(init='k-means++', n_clusters=3, n_init=15)
k_means.fit(data)
# стандартизируем данные из разны шкал
data = np.array(data, dtype='float')
min_max_scaler = preprocessing.StandardScaler()
scaled_data = min_max_scaler.fit_transform(data)

clustering = DBSCAN().fit(scaled_data)
print('Метки: ', set(clustering.labels_))
print('Количество: ', len(set(clustering.labels_)) - 1)
print('Выпавшие, %:', list(clustering.labels_).count(-1) / len(list(clustering.labels_)))

min_sample_range = np.arange(1, 4, 0.5)
eps_range = np.arange(1, 4, 0.2)

clust_number = []
clust_percent_dropped = []

for eps in min_sample_range:
    for min_sample in min_sample_range:
        clustering = DBSCAN(eps = eps, min_samples=min_sample).fit(scaled_data)
        clust_number.append(len(set(clustering.labels_)) - 1)
        clust_percent_dropped.append(100 * list(clustering.labels_).count(-1) / len(list(clustering.labels_)))
        if 5 <= clust_number[len(clust_number)-1] <= 7 and clust_percent_dropped[len(clust_percent_dropped)-1] <= 12:
            print("eps= ", eps," min_sample= ", min_sample," clust_number= ", clust_number[len(clust_number)-1]," clust_percent_dropped= ", clust_percent_dropped[len(clust_percent_dropped)-1])
        else: print("Неудача", eps, min_sample, clust_number[len(clust_number)-1], clust_percent_dropped[len(clust_percent_dropped)-1])


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(min_sample_range, clust_number)
ax[0].set_ylabel('Количество кластеров')
ax[0].set_xlabel('eps')
ax[1].plot(min_sample_range, clust_percent_dropped)
ax[1].set_ylabel('% Количество без кластера')
ax[1].set_xlabel('eps')
plt.show()

pca_data = PCA(n_components=2).fit_transform(data)

db = DBSCAN(eps=2, min_samples=3).fit(scaled_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)



# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = pca_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = pca_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


for eps in eps_range:
    for min_sample in min_sample_range:
        clustering = DBSCAN(eps = eps, min_samples=min_sample).fit(scaled_data)
        clust_number.append(len(set(clustering.labels_)) - 1)
        clust_percent_dropped.append(100 * list(clustering.labels_).count(-1) / len(list(clustering.labels_)))
        if 5 <= clust_number[len(clust_number)-1] <= 7 and clust_percent_dropped[len(clust_percent_dropped)-1] <= 12:
            print("eps= ", eps," min_sample= ", min_sample," clust_number= ", clust_number[len(clust_number)-1]," clust_percent_dropped= ", clust_percent_dropped[len(clust_percent_dropped)-1])
        else: print("Неудача", eps, min_sample, clust_number[len(clust_number)-1], clust_percent_dropped[len(clust_percent_dropped)-1])





pca_data = PCA(n_components=2).fit_transform(data)
np.random.seed(0)
n_points_per_cluster = 250


clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)

# Run the fit
clust.fit(scaled_data)

labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=0.5)
labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2)

space = np.arange(len(scaled_data))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Reachability plot
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = pca_data[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(pca_data[clust.labels_ == -1, 0], pca_data[clust.labels_ == -1, 1], 'k+', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')

# DBSCAN at 0.5
colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
for klass, color in zip(range(0, 6), colors):
    Xk = pca_data[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
ax3.plot(pca_data[labels_050 == -1, 0], pca_data[labels_050 == -1, 1], 'k+', alpha=0.1)
ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

# DBSCAN at 2.
colors = ['g.', 'm.', 'y.', 'c.']
for klass, color in zip(range(0, 4), colors):
    Xk = pca_data[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(pca_data[labels_200 == -1, 0], pca_data[labels_200 == -1, 1], 'k+', alpha=0.1)
ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

plt.tight_layout()
plt.show()

metrics = ["cosine", "euclidean", "cityblock", "manhattan", "l1"]
for metric in metrics:
    clustering = OPTICS(max_eps=2, min_samples=3, cluster_method='dbscan', metric=metric).fit(scaled_data)
    clust_number = len(set(clustering.labels_)) - 1
    clust_percent_dropped = list(clustering.labels_).count(-1) / len(list(clustering.labels_)) * 100
    print("Метрика: ", metric, " Кластеров: ", clust_number," Безкластерных: ", clust_percent_dropped)

