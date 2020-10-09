#!/usr/bin/env python3

import numpy
from scipy import stats
from scipy import integrate
from matplotlib import pyplot


def exercise_1():
    x = numpy.array([69, 74, 68, 70, 72, 67, 66, 70, 76, 68, 72, 79, 74, 67, 66, 71,
                     74, 75, 75, 76])
    y = numpy.array([153, 175, 155, 135, 172, 150, 115, 137, 200, 130, 140, 265, 185,
                     112, 140, 150, 165, 185, 210, 220])
    print(f'X: {x}')
    print(f'Y: {y}')

    print('Exercise A:')

    mean_x = x.mean()
    median_x = numpy.median(x)
    mode_x = stats.mode(x).mode[0]

    print(f'X mean: {mean_x}')
    print(f'X median: {median_x}')
    print(f'X mode: {mode_x}')

    print('Exercise B:')

    variance_y = y.var()

    print(f'Y variance: {variance_y}')

    print('Exercise C:')

    std_x = x.std()

    figure, axes = pyplot.subplots()

    plot_x = numpy.linspace(x.min(), x.max())
    plot_y = stats.norm.pdf(plot_x, mean_x, std_x)

    pyplot.plot(plot_x, plot_y)

    pyplot.show()
    print('See plot')

    print('Exercise D:')

    def gauss_func_x(_x):
        return stats.norm.pdf(_x, mean_x, std_x)

    probability_that_x_gt_80 = integrate.quad(gauss_func_x, 80, mean_x + std_x * 3)[0]

    print(f'Probability that x > 80: {probability_that_x_gt_80}')

    print('Exercise E:')

    mean_y = y.mean()

    cov_matrix = numpy.cov(x, y)

    print(f'Y mean: {mean_y}')
    print(f'Cov matrix of X and Y:\n {cov_matrix}')

    print('Exercise F:')

    corr_x_y = numpy.corrcoef(x, y)[0][1]

    print(f'X & Y correlation: {corr_x_y}')

    print('Exercise G:')

    figure, axes = pyplot.subplots()

    pyplot.scatter(x, y)

    pyplot.show()


def exercise_2():
    data = numpy.array([[17, 17, 12],
                        [11, 9, 13],
                        [11, 8, 19]])

    cov_mat = numpy.cov(data)

    print(f'Cov matrix: {cov_mat}')

    general_variance = numpy.linalg.det(cov_mat)

    print(f'General variance: {general_variance}')


def exercise_3():
    n_a_mean = 4
    n_a_std = 1

    n_b_mean = 8
    n_b_std = 4

    sample = [5, 6, 7]

    # 1. For each in sample: in which norm distribution is probability bigger?
    probability_sample_n_a = stats.norm.pdf(sample, n_a_mean, n_a_std)
    probability_sample_n_b = stats.norm.pdf(sample, n_b_mean, n_b_std)

    for index in range(0, len(sample)):
        print(f'Value: {sample[index]}')
        if probability_sample_n_a[index] > probability_sample_n_b[index]:
            print(f'Probability is bigger in N_a with the value {probability_sample_n_a[index]}')
        else:
            print(f'Probability is bigger in N_b with the value {probability_sample_n_b[index]}')

    # 2. Find val which has equal probability in both distributions.
    # Idea: find the intersection of two curves,
    # i.e. solve the quadratic equation:
    # a*x^2 + b*x + c = 0, where a, b and c are defined below

    n_a_var = n_a_std**2
    n_b_var = n_b_std**2

    a = 1 / (2 * n_a_var) - 1 / (2 * n_b_var)
    b = n_b_mean / n_b_var - n_a_mean / n_a_var
    c = n_a_mean ** 2 / (2 * n_a_var) - n_b_mean ** 2 / (2 * n_b_var) - numpy.log(n_b_std / n_a_std)
    intersection_points = numpy.roots([a, b, c])

    print(f'Values with equal probabilities: {intersection_points}')


def main():
    exercise_1()
    exercise_2()
    exercise_3()


if __name__ == '__main__':
    main()
