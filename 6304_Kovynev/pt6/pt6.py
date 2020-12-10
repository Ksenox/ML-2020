from collections import OrderedDict
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

eps = 2
minPts = 3

points = OrderedDict({
    'a': (5, 8),
    'b': (10, 8),
    'c': (11, 8),
    'd': (6, 7),
    'e': (10, 7),
    'f': (12, 7),
    'g': (13, 7),
    'h': (5, 6),
    'i': (10, 6),
    'j': (13, 6),
    'k': (6, 5),
    'l': (9, 4),
    'm': (11, 5),
    'n': (14, 6),
    'o': (15, 5),
    'p': (2, 4),
    'q': (3, 4),
    'r': (5, 4),
    's': (6, 4),
    't': (7, 4),
    'u': (15, 4),
    'v': (3, 3),
    'w': (7, 3),
    'x': (8, 2)
})

euclidean_dist = lambda p1, p2: np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def core_points_by_f(function, points_, eps_=eps, minPts_=minPts):
    core_points = {}
    for (point1, coord1) in points_.items():
        for (point2, coord2) in points_.items():
            dist = function(np.array(coord1), np.array(coord2))
            if dist <= eps_ and point1 != point2:
                if point1 in core_points:
                    core_points[point1].append(point2)
                else:
                    core_points[point1] = [point2]
    for (point, close_points) in core_points.items():
        print(point, close_points, len(close_points), 'isCore:', len(close_points) >= minPts_)
    print('---')


core_points_by_f(euclidean_dist, points)

max_dist = lambda x, y: np.max(np.abs(x - y))
sum_dist = lambda x, y: np.sum(np.abs(x - y) ** 0.5) ** 2
min_dist = lambda x, y: np.min(np.abs(x - y))
twice_dist = lambda x, y: np.sum([2 ** i * (x[i] - y[i]) ** 2 for i in range(len(x))]) ** (1 / 2)

X = np.array([[p[0], p[1]] for p in points.values()])


def clustrering_by_f(function, eps_=eps, minPts_=minPts):
    clustering = DBSCAN(eps=eps_, min_samples=minPts_, metric=function).fit(X)

    scatters = {}
    for index, state in enumerate(clustering.labels_):
        print(chr(97 + index), state)
        if state in scatters:
            scatters[state].append(X[index])
        else:
            scatters[state] = [X[index]]

    for (state, points) in scatters.items():
        x, y = zip(*points)
        if state == -1:
            plt.scatter(x, y, color='red')
        else:
            plt.scatter(x, y)

    plt.show()


# clustrering_by_f(euclidean_dist)


points2 = OrderedDict({
    'a': (5, 8),
    'b': (6, 7),
    'c': (6, 5),
    'd': (2, 4),
    'e': (3, 4),
    'f': (5, 4),
    'g': (7, 4),
    'h': (9, 4),
    'i': (3, 3),
    'j': (8, 2),
    'k': (7, 5)
})

X = np.array([[p[0], p[1]] for p in points2.values()])

# clustrering_by_f(max_dist, eps_=2, minPts_=5)
# clustrering_by_f(sum_dist, eps_=4, minPts_=3)
# clustrering_by_f(min_dist, eps_=1, minPts_=6)
# clustrering_by_f(twice_dist, eps_=4, minPts_=6)


core_points_by_f(max_dist, points2, eps_=2, minPts_=5)
core_points_by_f(sum_dist, points2, eps_=4, minPts_=3)
core_points_by_f(min_dist, points2, eps_=1, minPts_=6)
core_points_by_f(twice_dist, points2, eps_=4, minPts_=6)
