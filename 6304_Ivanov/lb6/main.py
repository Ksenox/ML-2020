
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, OPTICS


data = pd.read_csv('CC_GENERAL.csv').iloc[:,1:].dropna()

# DBSCAN


k_means = KMeans(init='k-means++', n_clusters=3, n_init=15)
k_means.fit(data)

data = np.array(data, dtype='float')
min_max_scaler = preprocessing.StandardScaler()
scaled_data = min_max_scaler.fit_transform(data)
clustering = DBSCAN().fit(scaled_data)



labels_set = set(clustering.labels_)

print(set(clustering.labels_), list(clustering.labels_).count(-1) / len(list(clustering.labels_)))


# 4
eps_ = np.arange(0.5, 4, 0.5)
info = []
for eps in eps_:
    clustering = DBSCAN(eps=eps).fit(scaled_data)
    labels_set = set(clustering.labels_)
    info.append([len(labels_set) - 1, list(clustering.labels_).count(-1) / len(list(clustering.labels_))])

info = np.array(info)
fig, ax = plt.subplots(1, 2, figsize=(13,6))
ax[0].plot(eps_, info[:,0])
ax[0].set_xlabel('eps')
ax[0].set_ylabel('Количество кластеров')
ax[1].plot(eps_, info[:,1])
ax[1].set_xlabel('eps')
ax[1].set_ylabel('Процент выбросов')


# 5
samples = np.arange(2, 15, 1)
info = []
for sample in samples:
    clustering = DBSCAN(min_samples=sample).fit(scaled_data)
    labels_set = set(clustering.labels_)
    info.append([len(labels_set) - 1, list(clustering.labels_).count(-1) / len(list(clustering.labels_))])



info = np.array(info)
fig, ax = plt.subplots(1, 2, figsize=(13,6))
ax[0].plot(samples, info[:,0])
ax[0].set_xlabel('min_samples')
ax[0].set_ylabel('Количество кластеров')
ax[1].plot(samples, info[:,1])
ax[1].set_xlabel('min_samples')
ax[1].set_ylabel('Процент выбросов')


# 6
samples = np.arange(1, 4, 1)
eps_ = np.arange(1.5, 2.5, 0.1)
info = {}
for sample in samples:
    for eps in eps_:
        clustering = DBSCAN(eps=eps ,min_samples=sample, n_jobs=-1).fit(scaled_data)
        labels_set = set(clustering.labels_)
        info[(sample, eps)]= [len(labels_set) - 1, list(clustering.labels_).count(-1) / len(list(clustering.labels_))]

print('(samples, eps) -> [count of clusters, percent of ]')
for key, value in info.items():
    if value[0]>=5 and value[0]<=7 and value[1]<=0.12:
        print(key, value)

clustering = DBSCAN(eps=2, min_samples=3, n_jobs=-1).fit(scaled_data)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True
labels = clustering.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(12, 8))
unique_labels = sorted(list(unique_labels))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = reduced_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=9)

    xy = reduced_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)


# OPTICS

# 6
samples = np.arange(10, 20, 2)
eps_ = np.arange(1, 5, 1)
info = {}
for sample in samples:
    for eps in eps_:
        clustering = OPTICS(max_eps=eps, min_samples=sample, n_jobs=-1).fit(scaled_data)
        labels_set = set(clustering.labels_)
        info[(sample, eps)]= [len(labels_set) - 1, list(clustering.labels_).count(-1) / len(list(clustering.labels_))]
print(info)

clustering = OPTICS(max_eps=2., min_samples=3, cluster_method='dbscan').fit(scaled_data)
labels_set = set(clustering.labels_)
print(len(labels_set) - 1, list(clustering.labels_).count(-1) / len(list(clustering.labels_)))

unique_labels = sorted(list(labels_set))

print(unique_labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(12, 8))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = reduced_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=9)

    xy = reduced_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

print(unique_labels)

space = np.arange(len(scaled_data))
reachability = clustering.reachability_[clustering.ordering_]
labels = clustering.labels_[clustering.ordering_]

plt.figure(figsize=(10, 7))
ax1 = plt.subplot()

# Reachability plot
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')


metrics = ['manhattan', 'euclidean', 'canberra', 'braycurtis', 'chebyshev']
for metric in metrics:
    for max_eps in (1, 2, 10):
        print(max_eps)
        clustering = OPTICS(max_eps=max_eps, min_samples=10, n_jobs=-1, metric=metric, cluster_method='dbscan').fit(scaled_data)
        print(set(clustering.labels_))
        num_of_clusters = len(set(clustering.labels_)) - 1
        not_classified = list(clustering.labels_).count(-1) / len(list(clustering.labels_))
        print('{:15} clusters: {}, not classified: {:.4f}'.format(metric, num_of_clusters, not_classified))
        print('------------')
