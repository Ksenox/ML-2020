import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FactorAnalysis

def showPlot(var_names, labels, data):
  fig, axs = plt.subplots(2,4)
  for i in range(data.shape[1]-1):
    axs[i // 4, i % 4].scatter(data[:,i],data[:,(i+1)],c=labels,cmap='nipy_spectral')
    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i+1])
  plt.show()

def testSvdSolver(n_components, svd_solver):
  print(n_components, svd_solver)
  pca = PCA(n_components = n_components, svd_solver=svd_solver)
  pca_data = pca.fit(data).transform(data)
  print(pca.explained_variance_ratio_)
  print(pca.singular_values_)
  print(np.sum(pca.explained_variance_ratio_))
  plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,cmap='nipy_spectral')
  plt.show()

def testKernel(n_components, kernel, degree):
  print(n_components, kernel)
  kpca = KernelPCA(n_components, kernel=kernel, degree=degree)
  kpca_data = kpca.fit(data).transform(data)
  plt.scatter(kpca_data[:,0],kpca_data[:,1],c=labels,cmap='nipy_spectral')
  plt.show()

def testSparse(n_components, alpha):
  spca = SparsePCA(n_components = n_components, alpha=alpha)
  spca_data = spca.fit(data).transform(data)
  plt.scatter(spca_data[:,0],spca_data[:,1],c=labels,cmap='nipy_spectral')
  plt.show()

df = pd.read_csv('glass.csv')
var_names = list(df.columns)
labels = df.to_numpy('int')[:,-1]
data = df.to_numpy('float')[:,:-1]
data = preprocessing.minmax_scale(data)
showPlot(var_names, labels, data)

testSvdSolver(2, 'auto')

pca = PCA(n_components = 4)
pca_data = pca.fit(data).transform(data)
print(np.sum(pca.explained_variance_ratio_))
inversed_data = pca.inverse_transform(pca_data)
showPlot(var_names, labels, inversed_data)

testSvdSolver(2, 'full')
testSvdSolver(2, 'arpack')
testSvdSolver(2, 'randomized')

testKernel(2, 'linear', 0)
testKernel(2, 'poly', 2)
testKernel(2, 'poly', 3)
testKernel(2, 'poly', 4)
testKernel(2, 'rbf', 0)
testKernel(2, 'sigmoid', 0)
testKernel(2, 'cosine', 0)

testSparse(2, 1)
testSparse(2, 0.25)

fa = FactorAnalysis(2, tol=0.01)
fa_data = fa.fit(data).transform(data)
plt.scatter(fa_data[:,0],fa_data[:,1],c=labels,cmap='nipy_spectral')
plt.show()