import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances_argmin
import random
import math


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    hierarchy.dendrogram(linkage_matrix, **kwargs)


data = pd.read_csv('iris.data', header=None)
no_labeled_data = data[[0, 1, 2, 3]].values

k_means = KMeans(init='k-means++', n_clusters=3, n_init=15)
k_means.fit(no_labeled_data)

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(no_labeled_data, k_means_cluster_centers)

f, ax = plt.subplots(1, 3)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

for i in range(3):
    my_members = k_means_labels == i
    cluster_center = k_means_cluster_centers[i]
    for j in range(3):
        ax[j].plot(no_labeled_data[my_members, j], no_labeled_data[my_members, j + 1], 'w',
                   markerfacecolor=colors[i], marker='o', markersize=4)
        ax[j].plot(cluster_center[j], cluster_center[j + 1], 'o',
                   markerfacecolor=colors[i],
                   markeredgecolor='k', markersize=8)
plt.show()
plt.close(f)

pca_data = PCA(n_components=2).fit_transform(no_labeled_data)
k_means.fit(pca_data)

x_min, x_max = pca_data[:, 0].min() - 1, pca_data[:, 0].max() + 1
y_min, y_max = pca_data[:, 1].min() - 1, pca_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

fig = plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(pca_data[:, 0], pca_data[:, 1], 'k.', markersize=2)
centroids = k_means.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.show()
plt.close(fig)

k_means = KMeans(init='random', n_clusters=3, n_init=15)
for i in range(3):
    k_means.fit(pca_data)

    x_min, x_max = pca_data[:, 0].min() - 1, pca_data[:, 0].max() + 1
    y_min, y_max = pca_data[:, 1].min() - 1, pca_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

    Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    fig = plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(pca_data[:, 0], pca_data[:, 1], 'k.', markersize=2)
    centroids = k_means.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.show()
    plt.close(fig)

k_means = KMeans(init=np.array([pca_data[5], pca_data[2], pca_data[14]]), n_clusters=3, n_init=1)
k_means.fit(pca_data)

x_min, x_max = pca_data[:, 0].min() - 1, pca_data[:, 0].max() + 1
y_min, y_max = pca_data[:, 1].min() - 1, pca_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

fig = plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(pca_data[:, 0], pca_data[:, 1], 'k.', markersize=2)
centroids = k_means.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.show()
plt.close(fig)

dist = []
clusters = np.arange(10, 0, -1)
for i in clusters:
    k_means = KMeans(init='k-means++', n_clusters=i, n_init=15)
    k_means.fit_predict(pca_data)
    dist.append(k_means.inertia_)

fig = plt.figure(1)
plt.clf()
plt.plot(clusters, dist)
plt.show()
plt.close(fig)

k_means = KMeans(init='k-means++', n_clusters=3, n_init=15)
k_means.fit(pca_data)
nobatch = k_means.labels_
batch_k_means = MiniBatchKMeans(init='k-means++', n_clusters=3, n_init=15)
batch_k_means.fit(pca_data)
batch = batch_k_means.labels_

equal_clusters_labels = np.equal(batch, nobatch)
equal_points = np.empty((0, 2))
not_equal_points = np.empty((0, 2))

for i, elem in enumerate(equal_clusters_labels):
    if elem:
        equal_points = np.append(equal_points, np.reshape(pca_data[i], (1, 2)), axis=0)
    else:
        not_equal_points = np.append(not_equal_points, np.reshape(pca_data[i], (1, 2)), axis=0)

fig = plt.figure(1)
plt.clf()
plt.scatter(equal_points[:, 0], equal_points[:, 1])
plt.scatter(not_equal_points[:, 0], not_equal_points[:, 1])
plt.show()
plt.close(fig)

for i in range(2, 6):
    hier = AgglomerativeClustering(n_clusters=i, linkage='average')
    hier = hier.fit(no_labeled_data)
    hier_labels = hier.labels_

    f, ax = plt.subplots(1, 3)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    for i in range(3):
        my_members = hier_labels == i
        for j in range(3):
            ax[j].plot(no_labeled_data[my_members, j], no_labeled_data[my_members, j + 1], 'w',
                       markerfacecolor=colors[i], marker='o', markersize=4)
    plt.show()
    plt.close(f)


hier = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average')
hier = hier.fit(no_labeled_data)
hier_labels = hier.labels_

fig = plt.figure(1)
plt.clf()
plot_dendrogram(hier, p=6, truncate_mode='level')
plt.show()
plt.close(fig)

data1 = np.zeros([250, 2])
for i in range(250):
    r = random.uniform(1, 3)
    a = random.uniform(0, 2 * math.pi)
    data1[i, 0] = r * math.sin(a)
    data1[i, 1] = r * math.cos(a)

data2 = np.zeros([500, 2])
for i in range(500):
    r = random.uniform(5, 9)
    a = random.uniform(0, 2 * math.pi)
    data2[i, 0] = r * math.sin(a)
    data2[i, 1] = r * math.cos(a)

data = np.vstack((data1, data2))

linkages = ['ward', 'complete', 'single', 'average']

for link in linkages:
    hier = AgglomerativeClustering(n_clusters=2, linkage=link)
    hier = hier.fit(data)
    hier_labels = hier.labels_

    fig = plt.figure(1)
    plt.clf()
    my_members = hier_labels == 0
    plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o',
             markersize=4, color='red', linestyle='None')
    my_members = hier_labels == 1
    plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o',
             markersize=4, color='blue', linestyle='None')
    plt.show()
    plt.close(fig)
