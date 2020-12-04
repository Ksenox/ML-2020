import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import cm, colors
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FactorAnalysis

df = pd.read_csv('glass.csv')
var_names = list(df.columns)  # получение имен признаков
labels = df.to_numpy('int')[:, -1]  # метки классов
data = df.to_numpy('float')[:, :-1]  # описательные признаки
data = preprocessing.minmax_scale(data)

df_scaled = df.copy()
df_scaled.iloc[:, :-1] = data

fig, axs = plt.subplots(2, 4)
for i in range(data.shape[1] - 1):
    axs[i // 4, i % 4].scatter(data[:, i], data[:, (i + 1)], c=labels, cmap='hsv')
    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i + 1])
# plt.show()

norm = colors.Normalize(vmin=min(labels), vmax=max(labels))
cmap = cm.get_cmap('hsv')
unique_labels = list(set(labels))
label_colors = [cmap(norm(label)) for label in unique_labels]
fig, ax = plt.subplots(figsize=(6, 2))
ax.bar(range(len(unique_labels)), 1, color=label_colors)
ax.set_xticklabels([0, *unique_labels])
# plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit(data).transform(data)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='hsv')
# plt.show()


fig = plt.figure(figsize=(10, 10))
axs = plt.axes()
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='hsv')
# plt.show()


variance = []
for i in range(1, data.shape[1] + 1):
    compare_pca = PCA(n_components=i)
    compare_pca.fit_transform(data)
    variance.append(sum(compare_pca.explained_variance_ratio_))

for component, var in enumerate(variance):
    print(component + 1, var, 'is bigger than 0.85 - ', var > 0.85)

pca_85 = PCA(n_components=4, whiten=True)
pca_data_85 = pca_85.fit_transform(data)

pca_inversed_85 = pca_85.inverse_transform(pca_data_85)
df_inversed_pca_85 = pd.DataFrame(pca_inversed_85)
df_inversed_pca_85.columns = df.columns[:-1]

print('PCA: ', df_scaled.describe())
print('PCA inversed', df_inversed_pca_85.describe())

print((np.square(df_scaled - df_inversed_pca_85)).mean(axis=None))


def compare_pca(type):
    pca = PCA(n_components=4, svd_solver=type)
    pca_data = pca.fit_transform(data)
    print(f'pca_{type}_variance = ', sum(pca.explained_variance_ratio_))
    return pca_data


pca_full_data = compare_pca('full')
pca_arpack_data = compare_pca('arpack')
pca_randomized_data = compare_pca('randomized')

print(f"pca_full mean = {np.mean(pca_full_data, axis=0)}")
print(f"pca_arpack mean = {np.mean(pca_arpack_data, axis=0)}")
print(f"pca_randomized mean = {np.mean(pca_randomized_data, axis=0)}")

fig, axs = plt.subplots(1, 3)
fig.set_figwidth(16)
fig.set_figheight(5)
axs[0].scatter(pca_full_data[:, 0], pca_full_data[:, 1], c=labels, cmap='hsv')
axs[0].set_title('full')
axs[1].scatter(pca_arpack_data[:, 0], pca_arpack_data[:, 1], c=labels, cmap='plasma')
axs[1].set_title('arpack')
axs[2].scatter(pca_randomized_data[:, 0], pca_randomized_data[:, 1], c=labels, cmap='plasma')
axs[2].set_title('randomized')


# plt.show()


def compare_KernelPCA(kernel='linear'):
    kernel_pca = KernelPCA(n_components=4, kernel=kernel)
    kernel_pca_full = KernelPCA(data.shape[1] if kernel == 'sigmoid' else None)
    kernel_pca_data = kernel_pca.fit(data)
    kernel_pca_full_data = kernel_pca_full.fit(data)
    return kernel_pca.lambdas_, sum(kernel_pca.lambdas_), (sum(kernel_pca.lambdas_) / sum(kernel_pca_full.lambdas_))


print(len(compare_KernelPCA()))
print("Linear: {}".format(compare_KernelPCA()))
print("Poly: {}".format(compare_KernelPCA(kernel='poly')))
print("RBF: {}".format(compare_KernelPCA(kernel='rbf')))
print("Sigmoid: {}".format(compare_KernelPCA(kernel='sigmoid')))
print("Cosine: {}".format(compare_KernelPCA(kernel='cosine')))

pca_2 = PCA(n_components=2, whiten=True)
pca_data_2 = pca_2.fit_transform(data)
print("PCA.\ncomponents = {}".format(pca_2.components_))

sparse_pca_lars = SparsePCA(2, method='lars')
sparse_pca_lars_data = sparse_pca_lars.fit_transform(data)
print("Sparse - lars.\ncomponents = {}".format(sparse_pca_lars.components_))

sparse_pca_cd = SparsePCA(2, method='cd')
sparse_pca_cd_data = sparse_pca_cd.fit_transform(data)
print("Sparse - cd.\ncomponents = {}".format(sparse_pca_cd.components_))

fig, axs = plt.subplots(1, 3)
fig.set_figwidth(11)
fig.set_figheight(5)
axs[0].scatter(sparse_pca_lars_data[:, 0], sparse_pca_lars_data[:, 1], c=labels, cmap='hsv')
axs[0].set_title('sparcepca lars')
axs[1].scatter(sparse_pca_cd_data[:, 0], sparse_pca_cd_data[:, 1], c=labels, cmap='hsv')
axs[1].set_title('sparcepca cd')
axs[2].scatter(sparse_pca_cd_data[:, 0], pca_data_2[:, 1], c=labels, cmap='hsv')
axs[2].set_title('pca')
# plt.show()


sparse_pca_lars_alpha_75 = SparsePCA(2, method='lars', alpha=0.75)
sparse_pca_lars_alpha_50 = SparsePCA(2, method='lars', alpha=0.50)
sparse_pca_lars_alpha_25 = SparsePCA(2, method='lars', alpha=0.25)

sparse_pca_lars_alpha_75_data = sparse_pca_lars_alpha_75.fit_transform(data)
sparse_pca_lars_alpha_50_data = sparse_pca_lars_alpha_50.fit_transform(data)
sparse_pca_lars_alpha_25_data = sparse_pca_lars_alpha_25.fit_transform(data)

print(sparse_pca_lars_alpha_75.components_.round(3))
print(sparse_pca_lars_alpha_50.components_.round(3))
print(sparse_pca_lars_alpha_25.components_.round(3))

fig, axs = plt.subplots(1, 3)
fig.set_figwidth(11)
fig.set_figheight(5)
axs[0].scatter(sparse_pca_lars_alpha_75_data[:, 0], sparse_pca_lars_alpha_75_data[:, 1], c=labels, cmap='hsv')
axs[0].set_title('alpha=0.75')
axs[1].scatter(sparse_pca_lars_alpha_50_data[:, 0], sparse_pca_lars_alpha_50_data[:, 1], c=labels, cmap='hsv')
axs[1].set_title('alpha=0.50')
axs[2].scatter(sparse_pca_lars_alpha_25_data[:, 0], sparse_pca_lars_alpha_25_data[:, 1], c=labels, cmap='hsv')
axs[2].set_title('alpha=0.25')
# plt.show()


transformer = FactorAnalysis(n_components=2)
fa_2 = transformer.fit_transform(data)
transformer.components_.round(3)


def scatter_plot(data1, data2, title1=None, title2=None, xlabel=None, ylabel=None):
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    axs[0].scatter(data1[:, 0], data1[:, 1], c=labels, cmap='plasma')
    axs[0].set_title(title1)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[1].scatter(data2[:, 0], data2[:, 1], c=labels, cmap='plasma')
    axs[1].set_title(title2)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    return fig, axs


scatter_plot(pca_data[:, [0, 1]], fa_2[:, [0, 1]], 'PCA', 'FA')
plt.show()
