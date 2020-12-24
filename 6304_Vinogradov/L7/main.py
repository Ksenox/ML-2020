import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn import tree


def params_plot(_sizes, _X, _Y, **kwargs):
    _accuracy = []
    _clf = tree.DecisionTreeClassifier(**kwargs)
    for _size in _sizes:
        _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _Y, test_size=_size, random_state=630415)
        _y_pred = _clf.fit(_X_train, _y_train).predict(_X_test)
        _accuracy.append(metrics.accuracy_score(_y_test, _y_pred))
    plt.plot(_sizes, _accuracy)

data = pd.read_csv('iris.data', header=None)

X = data.iloc[:, :4].to_numpy()
labels = data.iloc[:, 4].to_numpy()

le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())  # количество наблюдений, который были неправильно определены
print(metrics.accuracy_score(y_test, y_pred))

fig = plt.figure()
sizes, accuracy = np.arange(0.05, 0.95, 0.05), []

for size in sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=630415)
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(sizes, accuracy)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.title('Gaussian naive bayes classifier test')
plt.show()
plt.close(fig)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

mnb = MultinomialNB()
y_pred = mnb.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())  # количество наблюдений, который были неправильно определены
print(metrics.accuracy_score(y_test, y_pred))

fig = plt.figure()
accuracy = []

for size in sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=630415)
    y_pred = mnb.fit(X_train, y_train).predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(sizes, accuracy)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.title('Multinomial naive bayes classifier test')
plt.show()
plt.close(fig)

cnb = ComplementNB()
y_pred = cnb.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())  # количество наблюдений, который были неправильно определены
print(metrics.accuracy_score(y_test, y_pred))

fig = plt.figure()
accuracy = []

for size in sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=630415)
    y_pred = cnb.fit(X_train, y_train).predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(sizes, accuracy)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.title('Complement naive bayes classifier test')
plt.show()
plt.close(fig)

bnb = BernoulliNB()
y_pred = bnb.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())  # количество наблюдений, который были неправильно определены
print(metrics.accuracy_score(y_test, y_pred))

fig = plt.figure()
accuracy = []

for size in sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=630415)
    y_pred = bnb.fit(X_train, y_train).predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(sizes, accuracy)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.title('Binomial naive bayes classifier test')
plt.show()
plt.close(fig)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=630415)

clf = tree.DecisionTreeClassifier()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())
print(metrics.accuracy_score(y_test, y_pred))
print('Leaves number:', clf.get_n_leaves())
print('Depth:', clf.get_depth())

plt.subplots(1, 1, figsize=(10, 10))
tree.plot_tree(clf, filled=True)
plt.show()

fig = plt.figure()
accuracy = []

for size in sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=630415)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
plt.plot(sizes, accuracy)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.title('Standart classifier tree test')
plt.show()
plt.close(fig)

criterions = ['gini', 'entropy']
splitters = ['best', 'random']
nums = np.arange(1, 10, 1)
splits = np.arange(0.1, 1, 0.1)

fig = plt.figure()
for criter in criterions:
    params_plot(sizes, X, Y, criterion=criter)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.legend(criterions)
plt.title('Criterion test')
plt.show()
plt.close(fig)


fig = plt.figure()
for split in splitters:
    params_plot(sizes, X, Y, splitter=split)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.legend(splitters)
plt.title('Splitter test')
plt.show()
plt.close(fig)

fig = plt.figure()
for num in nums:
    params_plot(sizes, X, Y, max_depth=num)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.legend(nums)
plt.title('Max depth test')
plt.show()
plt.close(fig)

fig = plt.figure()
for split in splits:
    params_plot(sizes, X, Y, min_samples_split=split)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.legend(['{:.2}'.format(split) for split in splits])
plt.title('Min samples split test')
plt.show()
plt.close(fig)

fig = plt.figure()
for num in nums:
    params_plot(sizes, X, Y, min_samples_leaf=num)
plt.ylabel('Prediction accuracy in %')
plt.xlabel('Test samples in %')
plt.legend(nums)
plt.title('Min samples leaf test')
plt.show()
plt.close(fig)