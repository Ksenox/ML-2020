import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.model_selection import train_test_split


def params_plot(_object, _sizes, _X, _Y, **kwargs):
    _accuracy = []
    _lda = _object(**kwargs)
    for _size in _sizes:
        _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _Y, test_size=_size, random_state=630415)
        _y_pred = _lda.fit(_X_train, _y_train).predict(_X_test)
        _accuracy.append(metrics.accuracy_score(_y_test, _y_pred))
    plt.plot(_sizes, _accuracy)


data = pd.read_csv('iris.data', header=None)

X = data.iloc[:, :4].to_numpy()
labels = data.iloc[:, 4].to_numpy()

le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

lda = LinearDiscriminantAnalysis()
y_pred = lda.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())  # количество наблюдений, который были неправильно определены
print(metrics.accuracy_score(y_test, y_pred))

sizes, accuracy = np.arange(0.05, 0.95, 0.05), []

fig = plt.figure()
for size in sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=630415)
    y_pred = lda.fit(X_train, y_train).predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(sizes, accuracy)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.title('LDA')
plt.show()
plt.close(fig)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

pca = PCA(n_components=2)
pca_reducted = pca.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
lda_reducted = lda.fit(X_train, y_train).transform(X)

fig = plt.figure()
for i in set(Y):
    plt.scatter(pca_reducted[Y == i, 0], pca_reducted[Y == i, 1])
plt.title('PCA reducted')
plt.show()
plt.close(fig)

fig = plt.figure()
for i in set(Y):
    plt.scatter(lda_reducted[Y == i, 0], lda_reducted[Y == i, 1])
plt.title('LDA reducted')
plt.show()
plt.close(fig)

solvers = ['svd', 'lsqr', 'eigen']
shrinkages = [None, *np.arange(0.05, 0.95, 0.05).tolist()]

fig = plt.figure()
for solver in solvers:
    params_plot(LinearDiscriminantAnalysis, sizes, X, Y, solver=solver)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.legend(solvers)
plt.title('Solvers test')
plt.show()
plt.close(fig)

for solver in solvers[1:]:
    fig = plt.figure()
    for shrink in shrinkages:
        params_plot(LinearDiscriminantAnalysis, sizes, X, Y, solver= solver, shrinkage=shrink)
    plt.ylabel('Prediction accuracy in %')
    plt.xlabel('Test samples in %')
    plt.legend(['{:.2}'.format(shrink) if shrink else str(shrink) for shrink in shrinkages])
    plt.title('Shrinkage test: ' + solver)
    plt.show()
    plt.close(fig)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

fig = plt.figure()
params_plot(LinearDiscriminantAnalysis, sizes[:-1], X, Y, priors=[0.15, 0.7, 0.15])
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.title('Priors test')
plt.show()
plt.close(fig)


clf = SVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())
print(clf.score(X, Y))
print(metrics.accuracy_score(y_test, y_pred))
print(clf.support_vectors_)
print(clf.support_)
print(clf.n_support_)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=630415)

fig = plt.figure()
params_plot(SVC, sizes, X, Y)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.title('SVC test')
plt.show()
plt.close(fig)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
degrees = range(1, 11)
max_iters = [-1, *range(1, 10)]

fig = plt.figure()
for kernel in kernels:
    params_plot(SVC, sizes, X, Y, kernel=kernel)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.legend(kernels)
plt.title('SVC kerenels test')
plt.show()
plt.close(fig)

fig = plt.figure()
for degree in degrees:
    params_plot(SVC, sizes, X, Y, kernel='poly', degree=degree)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.legend(degrees)
plt.title('SVC degrees test')
plt.show()
plt.close(fig)

fig = plt.figure()
for iter in max_iters:
    params_plot(SVC, sizes, X, Y, max_iter=iter)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.legend(max_iters)
plt.title('SVC max iters test')
plt.show()
plt.close(fig)

fig = plt.figure()
params_plot(NuSVC, sizes, X, Y, nu=.1)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.title('NuSVC test')
plt.show()
plt.close(fig)

fig = plt.figure()
params_plot(LinearSVC, sizes, X, Y)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.title('LinearSVC test')
plt.show()
plt.close(fig)