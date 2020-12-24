import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


data = pd.read_csv('iris.data', header=None)

X = data.iloc[:, :4].to_numpy()
labels = data.iloc[:, 4].to_numpy()

le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

clf = LinearDiscriminantAnalysis()
y_pred = clf.fit(X_train, y_train).predict(X_test)

print('Score: {}'.format(round(clf.score(X_test, y_test), 3)))
print('Wrong: {}'.format((y_test != y_pred).sum()))


def plot_clf(clf, title=""):
    test_sizes = np.arange(0.05, 0.95, 0.05)
    wrong_results = []
    scores = []
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=630408)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        wrong_results.append((y_test != y_pred).sum())
        scores.append(clf.score(X_test, y_test))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(test_sizes, wrong_results, label="Кол-во неправильно опр. набл.")
    axs[1].plot(test_sizes, scores, label="Точность классификации")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_xlabel('Размер выборки')
    axs[1].set_xlabel('Размер выборки')
    plt.tight_layout()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


plot_clf(LinearDiscriminantAnalysis(), 'LinearDiscriminantAnalysis')

target_names = ['1', '2', '3']
y = y_train
X_r = clf.transform(X_train)
plt.figure()
colors = ['#ff0000', '#00ff00', '#0000ff']
lw = 2
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA')
# plt.show()


def plotter(x, y1, y2):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(x, y1, label="Кол-во неправильно опр. набл.")
    axs[1].plot(x, y2, label="Точность классификации")
    axs[0].set_xlabel('Размер выборки')
    axs[1].set_xlabel('Размер выборки')
    plt.tight_layout()
    plt.show()


def lda_data(**kwargs):
    wrong_results_1 = []
    scores_1 = []
    for test_size in np.arange(0.05, 0.95, 0.05):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=630408)
        clf_1 = LinearDiscriminantAnalysis(**kwargs)
        y_pred = clf_1.fit(X_train, y_train).predict(X_test)
        wrong_results_1.append((y_test != y_pred).sum())
        scores_1.append(clf_1.score(X_test, y_test))
    return wrong_results_1, scores_1


test_sizes = np.arange(0.05, 0.95, 0.05)
w1, s1 = lda_data()
w2, s2 = lda_data(solver='lsqr', shrinkage='auto')
w3, s3 = lda_data(solver='eigen', shrinkage='auto')
w4, s4 = lda_data(solver='lsqr', shrinkage=None)

plotter(test_sizes, w1, s1)
plotter(test_sizes, w2, s2)
plotter(test_sizes, w3, s3)
plotter(test_sizes, w4, s4)


plot_clf(LinearDiscriminantAnalysis(priors=[0.15, 0.7, 0.15]))

clf = svm.SVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())
print(clf.score(X, Y))

print('clf.support_vectors_', clf.support_vectors_)
print('clf.support_', clf.support_)
print('clf.n_support_', clf.n_support_)

plot_clf(svm.SVC(), 'SVC')

for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
    clf = svm.SVC(kernel=kernel)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print('kernel = ', kernel, (y_test != y_pred).sum(), clf.score(X, Y))

for degree in range(1, 8, 1):
    clf = svm.SVC(degree=degree, kernel='poly')
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print('degree = ', degree, (y_test != y_pred).sum(), clf.score(X, Y))


for max_iter in range(1, 15, 1):
    clf = svm.SVC(max_iter=max_iter)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print('max_iter = ', max_iter, (y_test != y_pred).sum(), clf.score(X, Y))


clf = svm.NuSVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())
print(clf.score(X_train, y_train))

clf = svm.LinearSVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())
print(clf.score(X_train, y_train))

plot_clf(svm.NuSVC(), 'NuSVC')

plot_clf(svm.LinearSVC(), 'LinearSVC')