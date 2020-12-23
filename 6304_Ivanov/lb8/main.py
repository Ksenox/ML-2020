import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

data = pd.read_csv('iris.data',header=None)

X = data.iloc[:,:4].to_numpy()
labels = data.iloc[:,4].to_numpy()

le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

clf = LinearDiscriminantAnalysis()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())

print(clf.score(X_train, y_train))

def plot_clf(clf, title=""):
    test_sizes = np.arange(0.05, 0.95, 0.05)
    wrong_results = []
    scores = []
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=630407)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        wrong_results.append((y_test != y_pred).sum())
        scores.append(clf.score(X_test, y_test))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(test_sizes, wrong_results)
    axs[1].plot(test_sizes, scores)
    axs[0].set_ylabel('Wrong classified')
    axs[1].set_ylabel('Score')
    axs[0].set_xlabel('test_size')
    axs[1].set_xlabel('test_size')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_clf(LinearDiscriminantAnalysis(), 'LinearDiscriminantAnalysis')

target_names = ['1', '2', '3']
y = y_train
X_r = clf.transform(X_train)
plt.figure()
colors = ['#efcdb8', '#0a0908', '#aa0908']
lw = 2
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA')
plt.show()

wrong_results_1 = []
scores_1 = []
wrong_results_2 = []
scores_2 = []
wrong_results_3 = []
scores_3 = []
wrong_results_4 = []
scores_4 = []
for test_size in np.arange(0.05, 0.95, 0.05):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=630407)
    clf_1 = LinearDiscriminantAnalysis(solver='svd', shrinkage=None)
    y_pred = clf_1.fit(X_train, y_train).predict(X_test)
    wrong_results_1.append((y_test != y_pred).sum())
    scores_1.append(clf_1.score(X_test, y_test))
    clf_2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    y_pred = clf_2.fit(X_train, y_train).predict(X_test)
    wrong_results_2.append((y_test != y_pred).sum())
    scores_2.append(clf_2.score(X_test, y_test))
    clf_3 = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    y_pred = clf_3.fit(X_train, y_train).predict(X_test)
    wrong_results_3.append((y_test != y_pred).sum())
    scores_3.append(clf_3.score(X_test, y_test))
    clf_4 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
    y_pred = clf_4.fit(X_train, y_train).predict(X_test)
    wrong_results_4.append((y_test != y_pred).sum())
    scores_4.append(clf_4.score(X_test, y_test))

test_sizes = np.arange(0.05, 0.95, 0.05)
wrong_label = 'Wrong classified'
score_label = 'Score'
size_label = 'test_size'
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(test_sizes, wrong_results_1)
axs[1].plot(test_sizes, scores_1)
axs[0].set_ylabel(wrong_label)
axs[1].set_ylabel(score_label)
axs[0].set_xlabel(size_label)
axs[1].set_xlabel(size_label)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(test_sizes, wrong_results_2)
axs[1].plot(test_sizes, scores_2)
axs[0].set_ylabel(wrong_label)
axs[1].set_ylabel(score_label)
axs[0].set_xlabel(size_label)
axs[1].set_xlabel(size_label)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(test_sizes, wrong_results_3)
axs[1].plot(test_sizes, scores_3)
axs[0].set_ylabel(wrong_label)
axs[1].set_ylabel(score_label)
axs[0].set_xlabel(size_label)
axs[1].set_xlabel(size_label)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(test_sizes, wrong_results_4)
axs[1].plot(test_sizes, scores_4)
axs[0].set_ylabel(wrong_label)
axs[1].set_ylabel(score_label)
axs[0].set_xlabel(size_label)
axs[1].set_xlabel(size_label)
plt.tight_layout()
plt.show()

plot_clf(LinearDiscriminantAnalysis(priors=[0.15, 0.7, 0.15]))

clf = LinearDiscriminantAnalysis(priors=[0.15, 0.7, 0.15])
y_pred = clf.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum()) #количество наблюдений, который были неправильно определены

print(clf.score(X, Y))

clf = svm.SVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())
print(clf.score(X, Y))

for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
    clf = svm.SVC(kernel=kernel)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print('kernel = ' + kernel)
    print((y_test != y_pred).sum())
    print(clf.score(X, Y))

for degree in range(1, 8, 1):
    clf = svm.SVC(degree=degree, kernel='poly')
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print('degree = ' + str(degree))
    print((y_test != y_pred).sum())
    print(clf.score(X, Y))

for max_iter in range(1, 15, 1):
    clf = svm.SVC(max_iter=max_iter)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print('max_iter = ' + str(max_iter))
    print((y_test != y_pred).sum())
    print(clf.score(X, Y))


plot_clf(svm.SVC(), 'SVC')

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
