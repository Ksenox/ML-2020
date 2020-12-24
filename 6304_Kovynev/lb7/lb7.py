import pandas as pd
import numpy as np
from sklearn import preprocessing, tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('iris.data', header=None)
data.head()
X = data.iloc[:, :4].to_numpy()
labels = data.iloc[:, 4].to_numpy()
le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)


gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print('Score: {}'.format(round(gnb.score(X_test, y_test), 3)))
print('Wrong: {}'.format((y_test != y_pred).sum()))


def plot_clf(clf):
    test_sizes = np.arange(0.05, 0.95, 0.05)
    wrong_results = []
    accuracies = []
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=630408)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        wrong_results.append((y_test != y_pred).sum())
        accuracies.append(clf.score(X_test, y_test))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(test_sizes, wrong_results, label="Кол-во неправильно опр. набл.")
    axs[1].plot(test_sizes, accuracies, label="Точность классификации")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_xlabel('Размер выборки')
    axs[1].set_xlabel('Размер выборки')
    plt.tight_layout()
    plt.legend()
    plt.show()


plot_clf(GaussianNB())
plot_clf(MultinomialNB())
plot_clf(ComplementNB())
plot_clf(BernoulliNB())

clf = tree.DecisionTreeClassifier()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print('Wrong classified: {}'.format((y_test != y_pred).sum()))

print("Score: {}".format(round(clf.score(X_test, y_test), 3)))

print('Num of leaves: {}'.format(clf.get_n_leaves()))
print('Depth: {}'.format(clf.get_depth()))


plt.subplots(1,1,figsize = (6,6))
tree.plot_tree(clf, filled = True)
plt.show()

plot_clf(clf)

clf = tree.DecisionTreeClassifier(criterion="entropy")
plot_clf(clf)
clf = tree.DecisionTreeClassifier(splitter="random")
plot_clf(clf)

depths = np.arange(1, 6, 1)
for depth in depths:
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    # plot_clf(clf)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print('Wrong classified: {}'.format((y_test != y_pred).sum()))
    print("Score: {}".format(round(clf.score(X_test, y_test), 3)))
    print('------')


min_samples = np.arange(10, 100, 10)
for min_sample in min_samples:
    clf = tree.DecisionTreeClassifier(min_samples_split=min_sample)
    # plot_clf(clf)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    print(min_sample, (y_test != y_pred).sum(), round(clf.score(X_test, y_test), 3))


min_samples = np.arange(10, 100, 10)
for min_sample in min_samples:
    clf = tree.DecisionTreeClassifier(min_samples_leaf=min_sample)
    plot_clf(clf)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    print(min_sample, (y_test != y_pred).sum(), round(clf.score(X_test, y_test), 3))
