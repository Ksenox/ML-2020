import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FactorAnalysis

df = pd.read_csv('glass.csv')
var_names = list(df.columns) #получение имен признаков

labels = df.to_numpy('int')[:,-1] #метки классов
data = df.to_numpy('float')[:,:-1] #описательные признаки

data = preprocessing.minmax_scale(data)

fig, axs = plt.subplots(2,4)

for i in range(data.shape[1]-1):
    axs[i // 4, i % 4].scatter(data[:,i],data[:,(i+1)],c=labels,cmap='hsv')

    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i+1])

plt.show()

plt.scatter(labels, [0.0]*len(labels), c=labels, cmap='hsv')
plt.show()

pca = PCA(n_components = 4, svd_solver = 'randomized')
pca_data = pca.fit(data).transform(data)

print(np.sum(pca.explained_variance_ratio_))
print(pca.singular_values_)

plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,cmap='hsv')
plt.show()

inverse_data = pca.inverse_transform(pca_data)

fig, axs = plt.subplots(2,4)

for i in range(inverse_data.shape[1]-1):
    axs[i // 4, i % 4].scatter(inverse_data[:,i],data[:,(i+1)],c=labels,cmap='hsv')

    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i+1])

plt.show()


#KernelPCA
kernelPCA = KernelPCA(n_components = 4, kernel = 'cosine')
kernelPCA_data = kernelPCA.fit(data).transform(data)

fig, axs = plt.subplots(1,1)

plt.scatter(kernelPCA_data[:,0],kernelPCA_data[:,1],c=labels,cmap='hsv')
plt.show()

plt.show()

#SparsePCA
sparsePCA = SparsePCA(n_components = 4, alpha=0.0)
sparsePCA_data = sparsePCA.fit(data).transform(data)

fig, axs = plt.subplots(1,1)

plt.scatter(sparsePCA_data[:,0],sparsePCA_data[:,1],c=labels,cmap='hsv')
plt.show()

plt.show()

#FactorAnalysis
factorAnalysis = FactorAnalysis(n_components = 2)
factorAnalysis_data = factorAnalysis.fit(data).transform(data)

fig, axs = plt.subplots(1,1)

plt.scatter(factorAnalysis_data[:,0],factorAnalysis_data[:,1],c=labels,cmap='hsv')
plt.show()

plt.show()