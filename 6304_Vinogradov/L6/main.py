import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn import preprocessing


def reachable_plot(_clustering, _space, _reachability, _labels):
    _fig = plt.figure()
    _colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(set(_clustering.labels_)))]
    for _klass, _color in zip(range(0, len(set(_clustering.labels_))), _colors):
        _Xk = _space[_labels == _klass]
        _Rk = _reachability[_labels == _klass]
        plt.plot(_Xk, _Rk, _color, alpha=0.3)
    plt.plot(_space[_labels == -1], _reachability[_labels == -1], 'k.', alpha=0.3)
    plt.plot(_space, np.full_like(_space, 2., dtype=float), 'k-', alpha=0.5)
    plt.plot(_space, np.full_like(_space, 0.5, dtype=float), 'k-.', alpha=0.5)
    plt.ylabel('Reachability (epsilon distance)')
    plt.title('Reachability Plot')
    plt.show()
    plt.close(_fig)


def cluster_plot(_data, _clustering):
    _core_samples_mask = np.zeros_like(_clustering.labels_, dtype=bool)
    _core_samples_mask[_clustering.core_sample_indices_] = True
    _labels = _clustering.labels_

    _n_clusters_ = len(set(_labels)) - (1 if -1 in _labels else 0)
    _n_noise_ = list(_labels).count(-1)

    _fig = plt.figure()
    _unique_labels = set(_labels)
    _colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(_unique_labels))]
    for k, col in zip(_unique_labels, _colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (_labels == k)

        xy = _data[class_member_mask & _core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = _data[class_member_mask & ~_core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % _n_clusters_)
    plt.show()
    plt.close(_fig)


data = pd.read_csv('CC GENERAL.csv').iloc[:, 1:].dropna()

data = np.array(data, dtype='float')
min_max_scaler = preprocessing.StandardScaler()
scaled_data = min_max_scaler.fit_transform(data)

"""clustering = DBSCAN().fit(scaled_data)
print(set(clustering.labels_))
print(len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0))
print(list(clustering.labels_).count(-1) / len(list(clustering.labels_)))

epss, clusters, noise_percent = np.linspace(0.01, 2, 10), [], []
for eps in epss:
    clustering = DBSCAN(eps=eps).fit(scaled_data)
    clusters.append(len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0))
    noise_percent.append(list(clustering.labels_).count(-1) / len(list(clustering.labels_)))
fig = plt.figure()
plt.subplot(211)
plt.plot(epss, clusters)
plt.ylabel('Clusters')
plt.subplot(212)
plt.plot(epss, noise_percent)
plt.ylabel('Noise percent')
plt.xlabel('Epsilon')
plt.show()
plt.close(fig)

points, clusters, noise_percent = np.linspace(2, 11, 10), [], []
for point in points:
    clustering = DBSCAN(min_samples=point).fit(scaled_data)
    clusters.append(len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0))
    noise_percent.append(list(clustering.labels_).count(-1) / len(list(clustering.labels_)))
fig = plt.figure()
plt.subplot(211)
plt.plot(points, clusters)
plt.ylabel('Clusters')
plt.subplot(212)
plt.plot(points, noise_percent)
plt.ylabel('Noise percent')
plt.xlabel('Min samples')
plt.show()
plt.close(fig)

clustering = DBSCAN(eps=1.7, min_samples=4).fit(scaled_data)
print(set(clustering.labels_))
print(len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0))
print(list(clustering.labels_).count(-1) / len(list(clustering.labels_)))

pca_data = PCA(n_components=2).fit_transform(scaled_data)

cluster_plot(pca_data, clustering)"""

clustering = OPTICS(cluster_method='dbscan', max_eps=0.5, min_samples=5).fit(scaled_data)
print(set(clustering.labels_))
print(len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0))
print(list(clustering.labels_).count(-1) / len(list(clustering.labels_)))

space = np.arange(len(scaled_data))
reachability = clustering.reachability_[clustering.ordering_]
labels = clustering.labels_[clustering.ordering_]

reachable_plot(clustering, space, reachability, labels)

metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']

for metric in metrics:
    clustering = OPTICS(min_samples=10, xi=0.07, metric=metric).fit(scaled_data)
    print(set(clustering.labels_))
    print(len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0))
    print(list(clustering.labels_).count(-1) / len(list(clustering.labels_)))

    space = np.arange(len(scaled_data))
    reachability = clustering.reachability_[clustering.ordering_]
    labels = clustering.labels_[clustering.ordering_]

    reachable_plot(clustering, space, reachability, labels)
