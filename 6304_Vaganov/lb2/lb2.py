import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FactorAnalysis

df = pd.read_csv('glass.csv')


var_names = list(df)
labels = df.to_numpy('int')[:,-1]
data = df.to_numpy('float')[:,:-1]

data = preprocessing.minmax_scale(data)

df_scaled = df.copy()
df_scaled.iloc[:,:-1] = data


fig, axs = plt.subplots(4, 2)
fig.set_figwidth(12)
fig.set_figheight(20)

for i in range(data.shape[1]-1):
    df_scaled.plot.scatter(x=var_names[i], y=var_names[i+1], c='Type', cmap='viridis', ax=axs[i // 2, i % 2])


eig_vals, _ = np.linalg.eig(df.iloc[:,:-1].corr())
data_variance_ratio = np.sort(eig_vals/sum(eig_vals))[::-1]


def plot_cumulative_variance(variance_ratio):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))

        plt.bar(range(1, len(variance_ratio)+1), variance_ratio, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(1, len(variance_ratio)+1), np.cumsum(variance_ratio), where='mid',
                label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()


plot_cumulative_variance(data_variance_ratio)



pca = PCA(n_components = 2)
pca_data = pca.fit_transform(data)
pca_variance_ratio = pca.explained_variance_ratio_


print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.singular_values_)


fig = plt.figure(figsize=(8,6))
axs = plt.axes()
df_pca_fitted = pd.DataFrame(pca_data).rename(columns={0: 'First component', 1: 'Second component'})
df_pca_fitted['Type'] = df.Type
df_pca_fitted.plot.scatter(x='First component', y='Second component', c='Type', cmap='viridis', ax=axs)


print('Explained variance ratio(2 components):', sum(pca.explained_variance_ratio_))


plot_cumulative_variance(pca.explained_variance_ratio_)


explained_variance = []

for i in range(1, data.shape[1]+1):
    compare_pca = PCA(n_components = i)
    compare_pca.fit_transform(data)
    explained_variance.append(sum(compare_pca.explained_variance_ratio_))


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(9, 6))
    plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.5, align='center')
    plt.hlines(0.85, 0.6, 9.4, colors='red')
    plt.ylabel('Sum of explained variance ratio')
    plt.xlabel('Principal components count')


pca_85 = PCA(n_components = 4, whiten=True)
pca_data_85 = pca_85.fit_transform(data)

print('Explained variance ratio(4 components):', sum(pca_85.explained_variance_ratio_))


plot_cumulative_variance(pca_85.explained_variance_ratio_)


def compare_scatter_plot(first_data, second_data, left_title=None, right_title=None, xlabel=None, ylabel=None):
    fig, axs = plt.subplots(1,2)
    fig.set_figwidth(11)
    fig.set_figheight(5)
    axs[0].scatter(first_data[:,0],first_data[:,1],c=labels,cmap='viridis')
    axs[0].set_title(left_title)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[1].scatter(second_data[:,0],second_data[:,1],c=labels,cmap='viridis')
    axs[1].set_title(right_title)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    return fig, axs


pca_inversed = pca.inverse_transform(pca_data)
pca_inversed_85 = pca_85.inverse_transform(pca_data_85)


df_iversed_pca = pd.DataFrame(pca_inversed)
df_iversed_pca.columns = df.columns[:-1]
df_iversed_pca_85 = pd.DataFrame(pca_inversed_85)
df_iversed_pca_85.columns = df.columns[:-1]


print('original scaled data:')
print(df_scaled.describe())

print('restored scaled data:')
print(df_iversed_pca_85.describe())


compare_scatter_plot(data[:, [1,5]], pca_inversed_85[:, [1,5]], 'original', 'restored', 'Na', 'K')


compare_scatter_plot(data[:, [2,8]], pca_inversed_85[:, [2,8]], 'original', 'restored', 'Mg', 'Fe')


# ---
# ### Вариация svd_solver в PCA

pca_compare_n_comp = 4


pca_full = PCA(n_components = pca_compare_n_comp, svd_solver='full')
pca_full_data = pca_full.fit_transform(data)
pca_full_variance_ratio = pca_full.explained_variance_ratio_


plot_cumulative_variance(pca_full_variance_ratio)
pca_full_variance_ratio


pca_arpack = PCA(n_components = pca_compare_n_comp, svd_solver='arpack')
pca_arpack_data = pca_arpack.fit_transform(data)
pca_arpack_variance_ratio = pca_arpack.explained_variance_ratio_


plot_cumulative_variance(pca_arpack_variance_ratio)
pca_arpack_variance_ratio


pca_randomized = PCA(n_components = pca_compare_n_comp, svd_solver='randomized')
pca_randomized_data = pca_randomized.fit_transform(data)
pca_randomized_variance_ratio = pca_randomized.explained_variance_ratio_


plot_cumulative_variance(pca_randomized_variance_ratio)
pca_randomized_variance_ratio


np.mean(pca_full_data, axis=0), np.mean(pca_arpack_data, axis=0), np.mean(pca_randomized_data, axis=0)


std_full = np.std(pca_full_data, axis = 0)
std_arpack = np.std(pca_arpack_data, axis = 0)
std_randomized = np.std(pca_randomized_data, axis = 0)


fig, axs = plt.subplots(1,3)
fig.set_figwidth(16)
fig.set_figheight(5)
axs[0].scatter(pca_full_data[:,0],pca_full_data[:,1],c=labels,cmap='viridis')
axs[0].set_title('full')
axs[1].scatter(pca_arpack_data[:,0],pca_arpack_data[:,1],c=labels,cmap='viridis')
axs[1].set_title('arpack')
axs[2].scatter(pca_randomized_data[:,0], pca_randomized_data[:,1],c=labels,cmap='viridis')
axs[2].set_title('randomized')


# ---
# ## Модификации метода главных компонент
# ### KernelPCA

kernel_pca_n_comp = 4


def compare_KernelPCA(n_comp=kernel_pca_n_comp, n_max_comp=100, **kwargs):
    kernel_pca = KernelPCA(n_components=n_comp, **kwargs)
    kernel_pca_full = KernelPCA(data.shape[1] if kwargs.get('kernel') == 'sigmoid' else None, **kwargs)
    kernel_pca.fit(data)
    kernel_pca_full.fit(data)
    return kernel_pca.lambdas_, \
        kernel_pca_full.lambdas_, \
        (sum(kernel_pca.lambdas_)/sum(kernel_pca_full.lambdas_)).round(3)


kernel_pca_linear_lambdas_, kernel_pca_full_linear_lambdas_, explained_variance = compare_KernelPCA()


print('Is equal singular values KernelPCA with linear kernel and PCA')
print(np.sqrt(kernel_pca_linear_lambdas_).round(3) == pca_85.singular_values_.round(3))


kernel_pca_linear_lambdas_, kernel_pca_full_linear_lambdas_, explained_variance


res_poly = compare_KernelPCA(kernel='poly')
res_rbf = compare_KernelPCA(kernel='rbf')
res_sigmoid = compare_KernelPCA(kernel='sigmoid')
res_cosine = compare_KernelPCA(kernel='cosine')


kernel_pca_precomputed = KernelPCA(n_components=kernel_pca_n_comp, kernel='precomputed')
kernel_pca_precomputed_data = kernel_pca_precomputed.fit_transform(data.dot(data.T))
kernel_pca_precomputed.lambdas_.round(3)

# ---
# ## Модификации метода главных компонент
# ### SparcePCA

sparse_pca_lars = SparsePCA(2, method='lars')
sparse_pca_lars_data = sparse_pca_lars.fit_transform(data)

print("Sparse PCA with lars method components")
print(sparse_pca_lars.components_)


sparse_pca_cd = SparsePCA(2, method='cd')
sparse_pca_cd_data = sparse_pca_cd.fit_transform(data)

print("Sparse PCA with cd method components")
print(sparse_pca_cd.components_)


fig, axs = plt.subplots(1,2)
fig.set_figwidth(11)
fig.set_figheight(5)
axs[0].scatter(sparse_pca_lars_data[:,0],sparse_pca_lars_data[:,1],c=labels,cmap='viridis')
axs[0].set_title('lars')
axs[1].scatter(sparse_pca_cd_data[:,0],sparse_pca_cd_data[:,1],c=labels,cmap='viridis')
axs[1].set_title('cd')


sparse_pca_lars_alpha_1  = SparsePCA(2, method='lars', alpha=1)
sparse_pca_lars_alpha_03 = SparsePCA(2, method='lars', alpha=0.3)
sparse_pca_lars_alpha_1_data  = sparse_pca_lars_alpha_1.fit_transform(data)
sparse_pca_lars_alpha_03_data = sparse_pca_lars_alpha_03.fit_transform(data)

print("Sparse pca with method lars and alpha = 1.0")
print(sparse_pca_lars_alpha_1.components_.round(3))
print("Sparse pca with method lars and alpha = 0.3")
print(sparse_pca_lars_alpha_03.components_.round(3))


fig, axs = plt.subplots(1,2)
fig.set_figwidth(11)
fig.set_figheight(5)
axs[0].scatter(sparse_pca_lars_alpha_1_data[:,0],sparse_pca_lars_alpha_1_data[:,1],c=labels,cmap='viridis')
axs[0].set_title('aplha=1')
axs[1].scatter(sparse_pca_lars_alpha_03_data[:,0],sparse_pca_lars_alpha_03_data[:,1],c=labels,cmap='viridis')
axs[1].set_title('aplha=0.3')

# ---
# ## Факторный анализ

transformer = FactorAnalysis(n_components=2)

fa_2 = transformer.fit_transform(data)

print('Components of FA')
print(transformer.components_.round(3))

compare_scatter_plot(pca_data[:,[0,1]], fa_2[:,[0,1]], 'PCA', 'FA')

plt.show()