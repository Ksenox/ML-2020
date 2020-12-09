import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.cluster.hierarchy import dendrogram


# 2
data = pd.read_csv('iris.data',header=None)

no_labeled_data = data.iloc[:,:4].to_numpy()
colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#4E0006', '#0FFFFF']


def k_means_function(clust_data, figsize=(18,6), **kwargs):
    k_means = KMeans(**kwargs)
    k_means.fit(clust_data)
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels = pairwise_distances_argmin(clust_data, k_means_cluster_centers)
    plots_count = clust_data.shape[1]-1
    f, ax = plt.subplots(1, plots_count, figsize=figsize)
    for i in range(kwargs.get('n_clusters')):
        my_members = k_means_labels == i
        cluster_center = k_means_cluster_centers[i]
        for j in range(plots_count):
            axis = ax if plots_count == 1 else ax[j]
            axis.plot(clust_data[my_members, j],
                clust_data[my_members, j+1], 'p',
                markerfacecolor=colors[i], marker='o',markeredgecolor=colors[i], markersize=4)
            axis.plot(cluster_center[j], cluster_center[j+1], 'o',
                markerfacecolor=colors[i],
                markeredgecolor='k', markersize=8)
    return k_means_labels, ax

# 3
labels, ax = k_means_function(no_labeled_data, figsize=(18,6), init='k-means++', n_clusters=3, n_init=1)
reduced_data = PCA(n_components=2).fit_transform(no_labeled_data)
labels, ax = k_means_function(reduced_data, figsize=(8,6), init='k-means++', n_clusters=3, n_init=10)

# 4
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
kmeans.fit(reduced_data)
h = .01
x_min, x_max = reduced_data[:, 0].min() - .3, reduced_data[:, 0].max() + .3
y_min, y_max = reduced_data[:, 1].min() - .3, reduced_data[:, 1].max() + .3
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(8,6))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.rainbow,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=4)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

# 5
labels, ax = k_means_function(no_labeled_data, figsize=(18,6), init='random', n_clusters=3, max_iter=500)
labels, ax = k_means_function(no_labeled_data, figsize=(18,6), init=np.array([[5.0,3.4,1.5,0.2],[5.8,2.2,4.4,1.5],[6.8,3.1,5.9,2.2]]), n_clusters=3, max_iter=5)

# 6
from scipy.spatial.distance import cdist
wcss=[]
for i in range(1,15):
    kmean = KMeans(n_clusters=i,init="k-means++")
    kmean.fit_predict(no_labeled_data)
    wcss.append(kmean.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1,15), wcss)
plt.title('The Elbow Method')
plt.xlabel("Num of Clusters")
plt.ylabel("WCSS")
plt.show()


# 7
data = no_labeled_data
n_clusters = 3
np.random.seed(1)
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
kmeans.fit(data)
mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=100, n_init=10)
mbk.fit(data)
fig = plt.figure(figsize=(25, 8))
k_means_cluster_centers = kmeans.cluster_centers_
order = pairwise_distances_argmin(kmeans.cluster_centers_,
                                  mbk.cluster_centers_)
mbk_means_cluster_centers = mbk.cluster_centers_[order]
k_means_labels = pairwise_distances_argmin(data, k_means_cluster_centers)
mbk_means_labels = pairwise_distances_argmin(data, mbk_means_cluster_centers)
print(mbk_means_labels)

# KMeans
# == 3
def ax_fill(ax, data, labels, centers, title, plot_centers=True, n_clusters=3):
    for k, col in zip(range(n_clusters), colors):
        my_members = labels == k
        ax.plot(data[my_members, 0], data[my_members, 1], 'o',
                markerfacecolor=col, markersize=6, markeredgecolor=col)
        if plot_centers:
            cluster_center = centers[k]
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=9)
    ax.set_title(title)

ax = fig.add_subplot(1, 3, 1)
ax_fill(ax,data, k_means_labels, k_means_cluster_centers, 'KMeans')
ax = fig.add_subplot(1, 3, 2)
ax_fill(ax, data, mbk_means_labels, mbk_means_cluster_centers, 'MiniBatchKMeans')
# diff
different = (mbk_means_labels == 4)
ax = fig.add_subplot(1, 3, 3)

for k in range(n_clusters):
    different += ((k_means_labels == k) != (mbk_means_labels == k))

identic = np.logical_not(different)
ax.plot(data[identic, 0], data[identic, 1], 'o',
        markerfacecolor='#bbbbbb', markersize=6, markeredgecolor='#bbbbbb')
ax.plot(data[different, 0], data[different, 1], 'o',
        markerfacecolor='m', markersize=6, markeredgecolor='m')
ax.set_title('Difference')
plt.show()


# Иерархическая кластеризация
np.random.seed(1)
hier = AgglomerativeClustering(n_clusters=5, linkage='average')
hier = hier.fit(no_labeled_data)
hier_labels = hier.labels_

max_cl_num = 4
f, ax = plt.subplots(3, max_cl_num, figsize=(16,8))

for j in range(max_cl_num):
    hier = AgglomerativeClustering(n_clusters=j+2, linkage='ward')
    hier = hier.fit(no_labeled_data)
    hier_labels = hier.labels_
    for i in range(3):
        ax_fill(ax[i][j], no_labeled_data[:,[i,i+1]], hier_labels, None, '', False, j+2)


# 4
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

X = no_labeled_data

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.figure(figsize=(24,15))
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(model, truncate_mode='level', p=2)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()



# 5
import random
import math
data1 = np.zeros([250,2])
for i in range(250):
    r = random.uniform(1, 3)
    a = random.uniform(0, 2 * math.pi)
    data1[i,0] = r * math.sin(a)
    data1[i,1] = r * math.cos(a)

data2 = np.zeros([500,2])
for i in range(500):
    r = random.uniform(5, 9)
    a = random.uniform(0, 2 * math.pi)
    data2[i,0] = r * math.sin(a)
    data2[i,1] = r * math.cos(a)
data = np.vstack((data1, data2))
plt.figure(figsize=(6,6))
plt.scatter(data[:,0],data[:,1])



# 6
hier = AgglomerativeClustering(n_clusters=2, linkage='ward')
hier = hier.fit(data)
hier_labels = hier.labels_
my_members = hier_labels == 0
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='red',linestyle='None')
my_members = hier_labels == 1
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='blue',linestyle='None')
plt.show()


# 8
type_linkage = ['ward', 'complete', 'average', 'single']
f, ax = plt.subplots(1, len(type_linkage), figsize=(24,6))

for idx, type_ in enumerate(type_linkage):
    hier = AgglomerativeClustering(n_clusters=2, linkage=type_)
    hier = hier.fit(data)
    hier_labels = hier.labels_
    my_members = hier_labels == 0
    ax[idx].plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='red',linestyle='None')
    my_members = hier_labels == 1
    ax[idx].plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='blue',linestyle='None')
    ax[idx].set_title(type_)

plt.show()
