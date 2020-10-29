import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import root


X = np.array([69, 74, 68, 70, 72, 67, 66, 70, 76, 68, 72, 79, 74, 67, 66, 71, 74, 75, 75, 76])
Y = np.array([153, 175, 155, 135, 172, 150, 115, 137, 200, 130, 140, 265, 185, 112, 140, 150, 165, 185, 210, 220])

# Найти среднее, медиану и моду величины X
x_mean = X.mean()
x_median = np.median(X)
x_mode = stats.mode(X)[0]
print('Среднее х: ', x_mean, '\nМедиана х: ', x_median, '\nМода х: ', x_mode, '\n')

# Найти дисперсию Y
y_var = np.var(Y)
print('Дисперсия Y', y_var, '\n')

# Построить график нормального распеределния для X
x_axis = np.linspace(X.min(), X.max())
y_axis = stats.norm.pdf(x_axis, X.mean(), X.std())
plt.plot(x_axis, y_axis)
plt.grid()
plt.show()

# Вероятность того, что возраст > 80
count80 = np.count_nonzero(X > 80)
print('Вероятность того, что возраст больше 80: ', count80, '%\n')

# Двумерное мат. ожидание и ковариациогнную матрицу для этих двух величин
xy_mean = np.mean([X, Y], axis=1)
xy_cov = np.cov([X, Y])

print('Двумерное мат. ожиадние: ', xy_mean)
print('Ковариационная матрица: \n', xy_cov, '\n')

# Корреляцию между X и Y
xy_corr = np.corrcoef(X, Y)[0, 1]
print('Коэфициент корреляции Пирсона', xy_corr, '\n')
# Величины очень сильно коррелируют, о чем говорит близкое к 1 значение коэфициента.
# К тому же они обе возрастают, о чем говорит положительное значение коэфициента.

# Диаграмма рассеивания
plt.scatter(X, Y)
plt.title('Диаграмма рассеивания')
plt.xlabel('Возраст')
plt.ylabel('Вес')
plt.axis([X.min() - 2, X.max() + 2, Y.min() - 2, Y.max() + 2])
plt.show()

# Задание 2
matrix = np.array([
    [17, 17, 12],
    [11, 9, 13],
    [11, 8, 19]
])

# Ковариационная матрица
cov_matrix = np.cov(matrix.T)
print(cov_matrix)

# Обобщенная дисперсия - определитель км
gen_variance = np.linalg.det(cov_matrix)
print('Обобщенная дисперсия: ', gen_variance, '\n')

# Задание 3

# Какое из распределений сгенерировало значение с большей вероятностью
m_Na, m_Nb = 4, 8
std_Na, std_Nb = 1, 2
array = [5, 6, 7]

N = 100
Na = np.random.normal(m_Na, std_Na, N)
Nb = np.random.normal(m_Nb, std_Nb, N)

prob = [[np.count_nonzero(np.abs(Na - x) < 1) / N, np.count_nonzero(np.abs(Nb - x) < 1) / N]
         for x in array]
print('Вероятность генерации', prob)

# Найти значение, которой могло быть сгенерировано обеими распределениями с равной вероятностью
res = root(lambda x: stats.norm.pdf(x, m_Nb, std_Nb) - stats.norm.pdf(x, m_Na, std_Na), 5)
print('Значение с равной вероятностью', res)
