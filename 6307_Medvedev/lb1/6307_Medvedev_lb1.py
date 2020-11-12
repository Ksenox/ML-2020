import pandas as pd
import numpy as np
from math import sqrt
import scipy.stats as sps
import matplotlib.pyplot as plt
from sklearn import preprocessing


def plot(data, title):
    if disableDrawing: return
    n_bins = 'auto'
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].hist(data[:, 0], bins=n_bins, edgecolor='black', linewidth=1)
    axs[0, 0].set_title('age')
    axs[0, 1].hist(data[:, 1], bins=n_bins, edgecolor='black', linewidth=1)
    axs[0, 1].set_title('creatinine_phosphokinase')
    axs[0, 2].hist(data[:, 2], bins=n_bins, edgecolor='black', linewidth=1)
    axs[0, 2].set_title('ejection_fraction')
    axs[1, 0].hist(data[:, 3], bins=n_bins, edgecolor='black', linewidth=1)
    axs[1, 0].set_title('platelets')
    axs[1, 1].hist(data[:, 4], bins=n_bins, edgecolor='black', linewidth=1)
    axs[1, 1].set_title('serum_creatinine')
    axs[1, 2].hist(data[:, 5], bins=n_bins, edgecolor='black', linewidth=1)
    axs[1, 2].set_title('serum_sodium')
    fig.suptitle(title, fontsize=16)
    plt.show()


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT'])
print(df)

disableDrawing = False
data = df.to_numpy(dtype='float')
plot(data, 'Исходные данные')

#  StandardScaler
scaler = preprocessing.StandardScaler()

scaler.fit(data[:150, :])
data_scaled150 = scaler.transform(data)
scalerObjMean150 = scaler.mean_
scalerObjVar150 = scaler.var_

plot(data_scaled150, 'Стандартизированные данные (по 150 наблюдениям)')

scaler.fit(data)
data_scaled = scaler.transform(data)
scalerObjMean = scaler.mean_
scalerObjVar = scaler.var_

plot(data_scaled, 'Стандартизированные данные (по всем наблюдениям)')

write2file = True
if write2file: f = open("out.csv", "w")
for i, col in enumerate(df.columns):
    print('\n' + col)
    X = data[:, i]
    print('Выборочное среденее (до стандартизации) для столбца %s: %.3f' % (col, X.mean()))
    print('СКО (до стандартизации) для столбца %s: %.3f' % (col, X.std()))
    if write2file: f.write(col + ";" + str(X.mean()) + ";" + str(X.std()) + ";")

    X = data_scaled150[:, i]
    print('Выборочное среденее по 150 для столбца %s: %.3f' % (col, X.mean()))
    print('СКО по 150 для столбца %s: %.3f' % (col, X.std()))
    if write2file: f.write(str(X.mean()) + ";" + str(X.std()) + ";")
    print('Выборочное среденее по 150 (scaler) для столбца %s: %.3f' % (col, scalerObjMean150[i]))
    print('СКО по 150 (scaler) для столбца %s: %.3f' % (col, sqrt(scalerObjVar150[i])))
    if write2file: f.write(str(scalerObjMean150[i]) + ";" + str(sqrt(scalerObjVar150[i])) + ";")

    X = data_scaled[:, i]
    print('Выборочное среденее по всем для столбца %s: %.3f' % (col, X.mean()))
    print('СКО по всем для столбца %s: %.3f' % (col, X.std()))
    if write2file: f.write(str(X.mean()) + ";" + str(X.std()) + ";")
    print('Выборочное среденее по всем (scaler) для столбца %s: %.3f' % (col, scaler.mean_[i]))
    print('СКО по всем (scaler) для столбца %s: %.3f' % (col, sqrt(scaler.var_[i])))
    if write2file: f.write(str(scaler.mean_[i]) + ";" + str(sqrt(scaler.var_[i])) + "\n")

if write2file: f.close()

#  MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(data)
data_min_max_scaled = min_max_scaler.transform(data)

plot(data_min_max_scaled, 'Стандартизированные данные (MinMaxScaler)')

print('\nMinMaxScaler\nМинимум: ')
for item in min_max_scaler.data_min_:
    print('{0:.1f}'.format(item), end=" ")
print('\n\nМаксимум: ')
for item in min_max_scaler.data_max_:
    print('{0:.1f}'.format(item), end=" ")

#  MaxAbsScaler
max_abs_scaler = preprocessing.MaxAbsScaler()
max_abs_scaler.fit(data)
data_max_abs_scaled = max_abs_scaler.transform(data)

plot(data_max_abs_scaled, 'Стандартизированные данные (MaxAbsScaler)')

print('\n\nMaxAbsScaler\nМаксимальный модуль: ')
for item in max_abs_scaler.max_abs_:
    print('{0:.1f}'.format(item), end=" ")

#  RobustScaler
robust_scaler = preprocessing.RobustScaler()
robust_scaler.fit(data)
data_robust_scaled = robust_scaler.transform(data)

plot(data_robust_scaled, 'Стандартизированные данные (RobustScaler)')


#  Fit to [-5; 10]
def my_scale(data):
    scaler = preprocessing.MinMaxScaler(feature_range=(-5, 10))
    scaler.fit(data)
    return scaler.transform(data)


data_scaled_custom = my_scale(data)
plot(data_scaled_custom, 'Стандартизированные данные ([-5; 10])')

#  Nonlinear transformations
#  Uniform distribution
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0)
quantile_transformer.fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

plot(data_quantile_scaled, 'Равномерное распределение, 100 квантилей')

quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=10, random_state=0)
quantile_transformer.fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

plot(data_quantile_scaled, 'Равномерное распределение, 10 квантилей')

#  Gaussian distribution
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100,
                                                         output_distribution='normal', random_state=0)
quantile_transformer.fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

plot(data_quantile_scaled, 'Нормальное распределение')

#  Power transformer
power_transformer = preprocessing.PowerTransformer()
power_transformer.fit(data)
data_power_scaled = power_transformer.transform(data)

plot(data_power_scaled, 'Нормальное распределение (Power transformer)')


#  Discretization
est = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal')
est.fit(data)
discretized_data = est.transform(data)

plot(discretized_data, 'Дискретизация')

print('\n\nKBinsDiscretizer\nКрая диапазонов: ')
for i, col in enumerate(df.columns):
    print(col + ": [ ", end="")
    for val in est.bin_edges_[i]:
        print('{0:.1f}'.format(val), end=" ")
    print(']')
