
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA, FactorAnalysis

df = pd.read_csv('glass.csv')

var_names = list(df.columns) #получение имен признаков
labels = df.to_numpy('int')[:,-1] #метки классов
data = df.to_numpy('float')[:,:-1] #описательные признаки

data = preprocessing.minmax_scale(data)
print(df.describe())

print(data.mean())

fig, axs = plt.subplots(2, 4)
for i in range(data.shape[1] - 1):
    axs[i // 4, i % 4].scatter(data[:, i], data[:, (i + 1)], c=labels, cmap='hsv')
    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i + 1])
#    plt.suptitle("1")
#plt.show()


print(len(labels))

#plt.hist(labels,  bins=np.arange(len(labels)))
fig, axs = plt.subplots(1, 1)
plt.scatter(labels, [1]*214, c = labels, cmap = 'hsv')   
plt.suptitle("Цвета/признаки")   
#plt.show()


plt.scatter(data[:,0], data[:,1], c=labels, cmap='hsv')
plt.suptitle("Factor")

#plt.show()


#метод главных компонент
fig, axs = plt.subplots(2, 4)
pca = PCA(n_components = 6)
pca_data = pca.fit(data).transform(data)

#значение объясненной дисперсии в процентах и собственные числа соответствующие компонентам
print(pca.explained_variance_ratio_)
print(pca.singular_values_)


#диаграмму рассеяния после метода главных компонент

for i in range(pca_data.shape[1] - 1):
    axs[i // 4, i % 4].scatter(pca_data[:, i], pca_data[:, (i + 1)], c=labels, cmap='hsv')
    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i + 1])
plt.suptitle("pca 4 компоненты")
#plt.show()


inversed_data = pca.inverse_transform(pca_data)

fig, axs = plt.subplots(2, 4)
for i in range(inversed_data.shape[1] - 1):
    axs[i // 4, i % 4].scatter(inversed_data[:, i], inversed_data[:, (i + 1)], c=labels, cmap='hsv')
    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i + 1])
plt.suptitle("inverse_transform")



#значение объясненной дисперсии в процентах и собственные числа соответствующие компонентам

print(inversed_data.var())
print(inversed_data.mean())
#plt.show()


kernels = ["linear", "poly", "rbf", "sigmoid", "cosine"]

plt.tight_layout()

for i in range(len(kernels)):
    kernels_pca = KernelPCA(n_components=4, kernel=kernels[i])
    kernels_pca_data = kernels_pca.fit(data).transform(data)
    print(kernels_pca.lambdas_)
#    plt.scatter(kernels_pca_data[:,0], kernels_pca_data[:,1], c=labels, cmap='hsv')
    plt.suptitle(kernels[i])
#    plt.show()




sparse = ["lars", "cd"]
for i in range(len(sparse)):
    sparse_pca = SparsePCA(n_components=4, method=sparse[i], alpha = 0.25)
    sparse_pca_data = sparse_pca.fit(data).transform(data)
    
    plt.scatter(sparse_pca_data[:,0], sparse_pca_data[:,1], c=labels, cmap='hsv')
    plt.suptitle(sparse[i]+ " alpha = 0.25")
   # plt.show()




factor_pca = FactorAnalysis(n_components= 4, svd_method='randomized')
factor_pca_data = factor_pca.fit(data).transform(data)
    
plt.scatter(factor_pca_data[:,0], factor_pca_data[:,1], c=labels, cmap='hsv')
plt.suptitle("factorAnalysis")
plt.show()
 