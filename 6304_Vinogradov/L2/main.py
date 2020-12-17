import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FactorAnalysis

df = pd.read_csv('glass.csv')
var_names = list(df.columns)  # получение имен признаков
labels = df.to_numpy('int')[:, -1]  # метки классов
data = df.to_numpy('float')[:, :-1]  # описательные признаки
data = preprocessing.minmax_scale(data)

fig, axs = plt.subplots(2, 4)
for i in range(data.shape[1] - 1):
    scatter = axs[i // 4, i % 4].scatter(data[:, i], data[:, (i + 1)], c=labels, cmap='hsv')
    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i + 1])
    axs[i // 4, i % 4].legend(*scatter.legend_elements())
plt.show()
plt.close(fig)

fig, axs = plt.subplots()
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)
axs.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='hsv')
plt.show()
plt.close(fig)

gammas = [-10, -2, -1, -0.1, 0.1, 1, 10, 100, 1000, 10000]
degrees = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
coef0 = [-1000, -100, -10, -1, -0.001, 0, 1, 10, 100, 1000]

fig, axs = plt.subplots()
kernel = KernelPCA(n_components=10, kernel='linear')
kernel_data = kernel.fit_transform(data)
axs.plot(np.arange(1, 11, 1), kernel.lambdas_.cumsum() / sum(kernel.lambdas_))
plt.xticks(np.arange(1, 11, 1))
axs.legend('linear')
plt.grid()
plt.show()
plt.close(fig)

fig, axs = plt.subplots()
for gamma in gammas:
    kernel = KernelPCA(n_components=10, kernel='rbf', gamma=gamma)
    kernel_data = kernel.fit_transform(data)
    axs.plot(np.arange(1, 11, 1), kernel.lambdas_.cumsum() / sum(kernel.lambdas_))
    plt.xticks(np.arange(1, 11, 1))
    axs.legend(gammas)
plt.grid()
plt.show()
plt.close(fig)

fig, axs = plt.subplots()
for gamma in gammas:
    kernel = KernelPCA(n_components=10, kernel='poly', gamma=gamma)
    kernel_data = kernel.fit_transform(data)
    axs.plot(np.arange(1, 11, 1), kernel.lambdas_.cumsum() / sum(kernel.lambdas_))
    plt.xticks(np.arange(1, 11, 1))
    axs.legend(gammas)
plt.grid()
plt.show()
plt.close(fig)

fig, axs = plt.subplots()
for degree in degrees:
    kernel = KernelPCA(n_components=10, kernel='poly', degree=degree)
    kernel_data = kernel.fit_transform(data)
    axs.plot(np.arange(1, 11, 1), kernel.lambdas_.cumsum() / sum(kernel.lambdas_))
    plt.xticks(np.arange(1, 11, 1))
    axs.legend(degrees)
plt.grid()
plt.show()
plt.close(fig)

fig, axs = plt.subplots()
for coef in coef0:
    kernel = KernelPCA(n_components=10, kernel='poly', coef0=coef)
    kernel_data = kernel.fit_transform(data)
    axs.plot(np.arange(1, 11, 1), kernel.lambdas_.cumsum() / sum(kernel.lambdas_))
    plt.xticks(np.arange(1, 11, 1))
    axs.legend(coef0)
plt.grid()
plt.show()
plt.close(fig)

gammas = [-10, -2, -1, -0.1, 0.1, 1, 10, 100]
coef0 = [-10, -1, -0.001, 0, 1, 10]

fig, axs = plt.subplots()
for gamma in gammas:
    kernel = KernelPCA(n_components=10, kernel='sigmoid', gamma=gamma)
    kernel_data = kernel.fit_transform(data)
    axs.plot(np.arange(1, 11, 1), kernel.lambdas_.cumsum() / sum(kernel.lambdas_))
    plt.xticks(np.arange(1, 11, 1))
    axs.legend(gammas)
plt.grid()
plt.show()
plt.close(fig)

fig, axs = plt.subplots()
for coef in coef0:
    kernel = KernelPCA(n_components=10, kernel='sigmoid', coef0=coef)
    kernel_data = kernel.fit_transform(data)
    axs.plot(np.arange(1, 11, 1), kernel.lambdas_.cumsum() / sum(kernel.lambdas_))
    plt.xticks(np.arange(1, 11, 1))
    axs.legend(coef0)
plt.grid()
plt.show()
plt.close(fig)

fig, axs = plt.subplots()
kernel = KernelPCA(n_components=10, kernel='cosine')
kernel_data = kernel.fit_transform(data)
axs.plot(np.arange(1, 11, 1), kernel.lambdas_.cumsum() / sum(kernel.lambdas_))
plt.xticks(np.arange(1, 11, 1))
axs.legend('cosine')
plt.grid()
plt.show()
plt.close(fig)

types = ['linear', 'rbf', 'poly', 'cosine', 'sigmoid']

fig, axs = plt.subplots()
for typee in types:
    kernel = KernelPCA(n_components=10, kernel=typee)
    kernel_data = kernel.fit_transform(data)
    axs.plot(np.arange(1, 11, 1), kernel.lambdas_.cumsum() / sum(kernel.lambdas_))
    plt.xticks(np.arange(1, 11, 1))
    axs.legend(types)
plt.grid()
plt.show()
plt.close(fig)

fig, axs = plt.subplots()
sparce_pca = SparsePCA(n_components=2)
sparce_pca_data = sparce_pca.fit_transform(data)
axs.scatter(sparce_pca_data[:, 0], sparce_pca_data[:, 1], c=labels, cmap='hsv')
plt.show()
plt.close(fig)

elems = np.empty((1, 0))
for i in range(20):
    sparce_pca = SparsePCA(n_components=9, alpha=-1 + i * 0.2)
    sparce_pca_data = sparce_pca.fit_transform(data)
    spca_comp = sparce_pca.components_
    elems = np.append(elems, np.count_nonzero(spca_comp) / spca_comp.size * 100)
fig, ax = plt.subplots()
ax.plot(np.arange(-1, 3, 0.2), elems)
plt.grid()
plt.show()
plt.close(fig)

fig, axs = plt.subplots()
factor_analysis = FactorAnalysis(n_components=2)
factor_analysis_data = factor_analysis.fit_transform(data)
axs.scatter(factor_analysis_data[:, 0], factor_analysis_data[:, 1], c=labels, cmap='hsv')
plt.show()
plt.close(fig)
