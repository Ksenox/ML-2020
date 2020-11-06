import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FactorAnalysis


def scatter_plot(data1, data2, title1=None, title2=None, xlabel=None, ylabel=None):
    fig, axs = plt.subplots(1,2, figsize=(11,5))
    axs[0].scatter(data1[:,0],data1[:,1],c=labels,cmap='plasma')
    axs[0].set_title(title1)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[1].scatter(data2[:,0],data2[:,1],c=labels,cmap='plasma')
    axs[1].set_title(title2)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    return fig, axs


df = pd.read_csv('glass.csv')

var_names = list(df.columns)

labels = df.to_numpy('int')[:,-1]
data = df.to_numpy('float')[:,:-1]

# Data Scaling
data = preprocessing.minmax_scale(data)

df_scale = df.copy()
df_scale.iloc[:,:-1] = data

# Plot scatter
fig, axs = plt.subplots(2, 4)
for i in range(data.shape[1]-1):
    axs[i // 4, i % 4].scatter(data[:,i],data[:,(i+1)],c=labels,cmap='hsv')
    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i+1])

fig.tight_layout()
plt.savefig('image1.png')
plt.show()

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit(data).transform(data)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,cmap='hsv')
plt.show()

explained_variance = []

for i in range(1, data.shape[1]+1):
    compare_pca = PCA(n_components = i)
    compare_pca.fit_transform(data)
    explained_variance.append(sum(compare_pca.explained_variance_ratio_))


plt.figure(figsize=(9, 6))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.5, align='center')
plt.hlines(0.85, 0.6, 9.4, colors='red')
plt.ylabel('Sum of explained variance ratio')
plt.xlabel('Components count')
plt.show()

# PCA 85%
pca_85 = PCA(n_components = 4, whiten=True)
pca_data_85 = pca_85.fit_transform(data)
print(pca_85)
print(pca_85.explained_variance_ratio_)

# PCA inverse
pca_inverse = pca.inverse_transform(pca_data)
pca_inverse_85 = pca_85.inverse_transform(pca_data_85)
df_inverse_pca = pd.DataFrame(pca_inverse)
df_inverse_pca.columns = df.columns[:-1]
df_inverse_pca_85 = pd.DataFrame(pca_inverse_85)
df_inverse_pca_85.columns = df.columns[:-1]

print(df_scale.describe())
print(df_inverse_pca_85.describe())

# Compare full/arpack/random
pca_full = PCA(n_components=4, svd_solver='full')
pca_full_data = pca_full.fit_transform(data)
pca_full_variance_ratio = pca_full.explained_variance_ratio_
print(pca_full_variance_ratio)

pca_arpack = PCA(n_components=4, svd_solver='arpack')
pca_arpack_data = pca_arpack.fit_transform(data)
pca_arpack_variance_ratio = pca_arpack.explained_variance_ratio_
print(pca_arpack_variance_ratio)

pca_randomized = PCA(n_components=4, svd_solver='randomized')
pca_randomized_data = pca_randomized.fit_transform(data)
pca_randomized_variance_ratio = pca_randomized.explained_variance_ratio_
print(pca_randomized_variance_ratio)

print("Mean:\n\tpca_full_data = {}\n\tpca_arpack_data = {}\n\tpca_randomized_data = {}". format(np.mean(pca_full_data, axis=0), np.mean(pca_arpack_data, axis=0), np.mean(pca_randomized_data, axis=0)))

fig, axs = plt.subplots(1,3)
fig.set_figwidth(16)
fig.set_figheight(5)
axs[0].scatter(pca_full_data[:,0],pca_full_data[:,1],c=labels,cmap='hsv')
axs[0].set_title('full')
axs[1].scatter(pca_arpack_data[:,0],pca_arpack_data[:,1],c=labels,cmap='plasma')
axs[1].set_title('arpack')
axs[2].scatter(pca_randomized_data[:,0], pca_randomized_data[:,1],c=labels,cmap='plasma')
axs[2].set_title('randomized')
plt.savefig('pca_svd.png')


# KernelPCA
def Compare_KernelPCA(n_comp=4, n_max_comp=100, **kwargs):
    kernel_pca = KernelPCA(n_components=n_comp, **kwargs)
    kernel_pca_full = KernelPCA(data.shape[1] if kwargs.get('kernel') == 'sigmoid' else None, **kwargs)
    kernel_pca_data = kernel_pca.fit(data)
    kernel_pca_full_data = kernel_pca_full.fit(data)
    return kernel_pca.lambdas_, kernel_pca_full.lambdas_, (sum(kernel_pca.lambdas_)/sum(kernel_pca_full.lambdas_)).round(3)


print(len(Compare_KernelPCA()))
print("Linear: ".format(Compare_KernelPCA()))
print("Poly: ".format(Compare_KernelPCA(kernel='poly')))
print("RBF: ".format(Compare_KernelPCA(kernel='rbf')))
print("Sigmoid: ".format(Compare_KernelPCA(kernel='sigmoid')))
print("Cosine: ".format(Compare_KernelPCA(kernel='cosine')))


kernel_pca_precomputed = KernelPCA(n_components=4, kernel='precomputed')
kernel_pca_precomputed_data = kernel_pca_precomputed.fit_transform(data.dot(data.T))
print("Precomputed:\n\tlambdas =", kernel_pca_precomputed.lambdas_.round(3))


# SparcePCA
sparse_pca_lars = SparsePCA(2, method='lars')
sparse_pca_lars_data = sparse_pca_lars.fit_transform(data)
print("Sparse - lars.\n\tcomponents = {}".format(sparse_pca_lars.components_))

sparse_pca_cd = SparsePCA(2, method='cd')
sparse_pca_cd_data = sparse_pca_cd.fit_transform(data)
print("Sparse - cd.\n\tcomponents = {}".format(sparse_pca_cd.components_))

fig, axs = plt.subplots(1,2)
fig.set_figwidth(11)
fig.set_figheight(5)
axs[0].scatter(sparse_pca_lars_data[:,0],sparse_pca_lars_data[:,1],c=labels,cmap='hsv')
axs[0].set_title('lars')
axs[1].scatter(sparse_pca_cd_data[:,0],sparse_pca_cd_data[:,1],c=labels,cmap='hsv')
axs[1].set_title('cd')
plt.savefig('sparse_pca.png')


sparse_pca_lars_alpha_75  = SparsePCA(2, method='lars', alpha=0.75)
sparse_pca_lars_alpha_25 = SparsePCA(2, method='lars', alpha=0.25)
sparse_pca_lars_alpha_75_data = sparse_pca_lars_alpha_75.fit_transform(data)
sparse_pca_lars_alpha_25_data = sparse_pca_lars_alpha_25.fit_transform(data)

print(sparse_pca_lars_alpha_75.components_.round(3))
print(sparse_pca_lars_alpha_25.components_.round(3))

fig, axs = plt.subplots(1,2)
fig.set_figwidth(11)
fig.set_figheight(5)
axs[0].scatter(sparse_pca_lars_alpha_75_data[:,0],sparse_pca_lars_alpha_75_data[:,1],c=labels,cmap='hsv')
axs[0].set_title('aplha=0.75')
axs[1].scatter(sparse_pca_lars_alpha_25_data[:,0],sparse_pca_lars_alpha_25_data[:,1],c=labels,cmap='hsv')
axs[1].set_title('aplha=0.25')
plt.savefig('sparse_pca_alpha.png')


# Factor analisis
transformer = FactorAnalysis(n_components=2)
fa = transformer.fit_transform(data)
transformer.components_.round(3)


scatter_plot(pca_data[:,[0,1]], fa[:,[0,1]], 'PCA', 'FA')
plt.savefig('fa_pca.png')




