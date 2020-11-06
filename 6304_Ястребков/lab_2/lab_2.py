import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FactorAnalysis


df = pd.read_csv('../../../glass.csv')
var_names = list(df.columns)  # получение имен признаков

labels = df.to_numpy('int')[:,-1]  # метки классов
data = df.to_numpy('float')[:,:-1]  # описательные признаки

print(df.head())

# Data scaling
data = preprocessing.minmax_scale(data)

# Plot scatter
_, axs = plt.subplots(2, 4, figsize=(15, 7))
for i in range(data.shape[1] - 1):
    axs[i // 4, i % 4].scatter(data[:, i], data[:, (i + 1)], c=labels, cmap='hsv')

    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i + 1])
plt.show()

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='hsv')
plt.show()

# Inverse transform
inversed = pca.inverse_transform(pca_data)

for i in range(data.shape[1]):
    std_init = np.std(data[:, i])
    std_inv = np.std(inversed[:, i])
    print(f"*Mean* init: {np.mean(data[:, i]):.3f} inversed: {np.mean(inversed[:, i]):.3f}" +
          f"\t*STD* init: {std_init:.3f} inversed: {std_inv:.3f}" +
          f" lost: {(std_init - std_inv) / std_init * 100:.3f}")

# Explained variance comparison
components = np.arange(1, 9)
explained_var = np.zeros(len(components))

for c in components:
    pca = PCA(n_components=c)
    pca.fit_transform(data)
    explained_var[c - 1] = np.sum(pca.explained_variance_ratio_)

plt.bar(components, explained_var, alpha=0.5, align='center')
plt.hlines(0.85, 0, 9, colors='r')
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.show()


# KernelPCA
pca_full = PCA(n_components=2, svd_solver='full').fit_transform(data)
pca_arpack = PCA(n_components=2, svd_solver='arpack').fit_transform(data)
pca_randomized = PCA(n_components=2, svd_solver='randomized').fit_transform(data)

_, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].scatter(pca_full[:, 0], pca_full[:, 1], c=labels, cmap='hsv')
ax[0].set_title('full')

ax[1].scatter(pca_arpack[:, 0], pca_arpack[:, 1], c=labels, cmap='hsv')
ax[1].set_title('arpack')

ax[2].scatter(pca_randomized[:, 0], pca_randomized[:, 1], c=labels, cmap='hsv')
ax[2].set_title('randomized')

# Kernels comparison
kernel_lin = KernelPCA(n_components=4, kernel='linear')
kernel_poly = KernelPCA(n_components=4, kernel='poly')
kernel_rbf = KernelPCA(n_components=4, kernel='rbf')
kernel_sig = KernelPCA(n_components=4, kernel='sigmoid')
kernel_cos = KernelPCA(n_components=4, kernel='cosine')

fig = plt.figure(constrained_layout=True, figsize=(15, 8))
gs = GridSpec(2, 6, figure=fig)

data_lin = kernel_lin.fit_transform(data)
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.scatter(data_lin[:, 0], data_lin[:, 1], c=labels, cmap='hsv')
ax1.set_title('linear')

data_poly = kernel_poly.fit_transform(data)
ax2 = fig.add_subplot(gs[0, 2:4])
ax2.scatter(data_poly[:, 0], data_poly[:, 1], c=labels, cmap='hsv')
ax2.set_title('polynomial')

data_rbf = kernel_rbf.fit_transform(data)
ax3 = fig.add_subplot(gs[0, 4:])
ax3.scatter(data_rbf[:, 0], data_rbf[:, 1], c=labels, cmap='hsv')
ax3.set_title('rbf')

data_sig = kernel_sig.fit_transform(data)
ax4 = fig.add_subplot(gs[1, 0:3])
ax4.scatter(data_sig[:, 0], data_sig[:, 1], c=labels, cmap='hsv')
ax4.set_title('sigmoid')

data_cos = kernel_cos.fit_transform(data)
ax5 = fig.add_subplot(gs[1, 3:])
ax5.scatter(data_cos[:, 0], data_cos[:, 1], c=labels, cmap='hsv')
ax5.set_title('cosine')

# Kernels variance comparison
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']

for k in kernels:
    krnl = KernelPCA(n_components=4, kernel=k).fit(data)
    krnl_full = KernelPCA(n_components=data.shape[1], kernel=k).fit(data)
    lam = krnl.lambdas_
    lam_full = np.sum(krnl_full.lambdas_)
    print(f"{lam[0]:.3f} {lam[1]:.3f} {lam[2]:.3f} {lam[3]:.3f} \tvar: {np.sum(lam) / lam_full}")


# SparsePCA
sparse_lar = SparsePCA(2, method='lars')
lar_data = sparse_lar.fit_transform(data)

sparse_cd = SparsePCA(2, method='cd')
cd_data = sparse_cd.fit_transform(data)

_, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].scatter(lar_data[:, 0], lar_data[:, 1], c=labels, cmap='hsv')
ax[0].set_title('lar')
ax[1].scatter(cd_data[:, 0], cd_data[:, 1], c=labels, cmap='hsv')
ax[1].set_title('cd')

# Alpha comparison
sparse_1 = SparsePCA(2, method='lars', alpha=0.1)
data_1 = sparse_1.fit_transform(data)

sparse_2 = SparsePCA(2, method='lars', alpha=0.5)
data_2 = sparse_2.fit_transform(data)

sparse_3 = SparsePCA(2, method='lars', alpha=1)
data_3 = sparse_3.fit_transform(data)

_, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].scatter(data_1[:, 0], data_1[:, 1], c=labels, cmap='hsv')
ax[0].set_title('alpha=0.1')
ax[1].scatter(data_2[:, 0], data_2[:, 1], c=labels, cmap='hsv')
ax[1].set_title('alpha=0.5')
ax[2].scatter(data_3[:, 0], data_3[:, 1], c=labels, cmap='hsv')
ax[2].set_title('alpha=1')

print(sparse_1.components_, sparse_2.components_, sparse_3.components_)


# Factor analysis
factor = FactorAnalysis(n_components=2)
data_factor = factor.fit_transform(data)

_, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='hsv')
ax[0].set_title('PCA')
ax[1].scatter(data_factor[:, 0], data_factor[:, 1], c=labels, cmap='hsv')
ax[1].set_title('Factor analysis')

