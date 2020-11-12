import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from prettytable import PrettyTable


def plot(data, title):
    if disableDrawing:
        return
    n_bins = 'auto'
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
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


def print_table(title):
    if disableTable:
        return

    varDF = np.mean(df)
    stdDF = np.std(df)
    scaler_var = scaler.var_
    scaler_mean = scaler.mean_
    scaler_all_var = scaler_all.var_
    scaler_all_mean = scaler_all.mean_
    print(title)
    table = PrettyTable()
    table2 = PrettyTable()
    table.add_column('Признак',
                     ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine',
                      'serum_sodium'])
    table2.add_column('Признак',
                     ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine',
                      'serum_sodium'])
    table.add_column('mean original',
                     [varDF['age'], varDF['creatinine_phosphokinase'], varDF['ejection_fraction'], varDF['platelets'],
                      varDF['serum_creatinine'], varDF['serum_sodium']])
    table2.add_column('std original',
                     [stdDF['age'], stdDF['creatinine_phosphokinase'], stdDF['ejection_fraction'], stdDF['platelets'],
                      stdDF['serum_creatinine'], stdDF['serum_sodium']])
    table.add_column('mean scaled 150',
                     [data_scaled[:, 0].mean(), data_scaled[:, 1].mean(), data_scaled[:, 2].mean(),
                      data_scaled[:, 3].mean(), data_scaled[:, 4].mean(), data_scaled[:, 5].mean()])
    table.add_column('scared.mean_ 150',
                     [scaler_mean[0], scaler_mean[1], scaler_mean[2],
                      scaler_mean[3], scaler_mean[4], scaler_mean[5]])
    table2.add_column('std scaled 150',
                     [data_scaled[:, 0].std(), data_scaled[:, 1].std(), data_scaled[:, 2].std(),
                      data_scaled[:, 3].std(), data_scaled[:, 4].std(), data_scaled[:, 5].std()])
    table2.add_column('scared.var_ 150',
                     [scaler_var[0], scaler_var[1], scaler_var[2],
                      scaler_var[3], scaler_var[4], scaler_var[5]])
    table.add_column('mean scaled all',
                     [data_scaled_all[:, 0].mean(), data_scaled_all[:, 1].mean(), data_scaled_all[:, 2].mean(),
                      data_scaled_all[:, 3].mean(), data_scaled_all[:, 4].mean(), data_scaled_all[:, 5].mean()])
    table.add_column('scared.mean_ all',
                     [scaler_all_mean[0], scaler_all_mean[1], scaler_all_mean[2],
                      scaler_all_mean[3], scaler_all_mean[4], scaler_all_mean[5]])
    table2.add_column('std scaled all',
                     [data_scaled_all[:, 0].std(), data_scaled_all[:, 1].std(), data_scaled_all[:, 2].std(),
                      data_scaled_all[:, 3].std(), data_scaled_all[:, 4].std(), data_scaled_all[:, 5].std()])
    table2.add_column('scared.var_ all',
                     [scaler_all_var[0], scaler_all_var[1], scaler_all_var[2],
                      scaler_all_var[3], scaler_all_var[4], scaler_all_var[5]])

    print(table)
    print(table2)



df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT'])
print(df)

disableDrawing = False
disableTable = False
data_np = df.to_numpy(dtype='float')
plot(data_np, 'Исходные данные')

# настройка на основе первых 150 наблюдений  и стандартизация
scaler = preprocessing.StandardScaler().fit(data_np[:150, :])
data_scaled = scaler.transform(data_np)

plot(data_scaled, 'Стандартизованные данные (150 наблюдений)')

scaler_all = preprocessing.StandardScaler().fit(data_np[:, :])
data_scaled_all = scaler_all.transform(data_np)

plot(data_scaled_all, 'Стандартизованные данные (все наблюдения)')

print_table('Cравнение')

#приведение к диапазону

min_max_scaler = preprocessing.MinMaxScaler().fit(data_np)
data_min_max_scaled = min_max_scaler.transform(data_np)
plot(data_min_max_scaled, 'Приведенные к диапазону данные (MinMaxScaler)')

print('\nMinMaxScaler\nМинимум: ')
for item in min_max_scaler.data_min_:
    print('{0:.1f}'.format(item), end=" ")
print('\n\nМаксимум: ')
for item in min_max_scaler.data_max_:
    print('{0:.1f}'.format(item), end=" ")

max_abs_scaler = preprocessing.MaxAbsScaler()
max_abs_scaler.fit(data_np)
data_max_abs_scaled = max_abs_scaler.transform(data_np)

plot(data_max_abs_scaled, 'Стандартизированные данные (MaxAbsScaler)')

print('\n\nMaxAbsScaler\nМаксимальный модуль: ')
for item in max_abs_scaler.max_abs_:
    print('{0:.1f}'.format(item), end=" ")

#  RobustScaler
robust_scaler = preprocessing.RobustScaler()
robust_scaler.fit(data_np)
data_robust_scaled = robust_scaler.transform(data_np)

plot(data_robust_scaled, 'Стандартизированные данные (RobustScaler)')

def my_scale(data):
    scaler = preprocessing.MinMaxScaler(feature_range=(-5, 10))
    scaler.fit(data)
    return scaler.transform(data)

data_scaled_custom = my_scale(data_np)
plot(data_scaled_custom, 'Стандартизированные данные ([-5; 10])')

#  Нелинейные преобразования
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0)
quantile_transformer.fit(data_np)
data_quantile_scaled = quantile_transformer.transform(data_np)

plot(data_quantile_scaled, 'Равномерное распределение, 100 квантилей')

quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=50, random_state=0)
quantile_transformer.fit(data_np)
data_quantile_scaled = quantile_transformer.transform(data_np)

plot(data_quantile_scaled, 'Равномерное распределение, 50 квантилей')

#  Нормальное распределение
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100,
                                                         output_distribution='normal', random_state=0)
quantile_transformer.fit(data_np)
data_quantile_scaled = quantile_transformer.transform(data_np)

plot(data_quantile_scaled, 'Нормальное распределение')

#  Power transformer
power_transformer = preprocessing.PowerTransformer()
power_transformer.fit(data_np)
data_power_scaled = power_transformer.transform(data_np)

plot(data_power_scaled, 'Нормальное распределение (Power transformer)')

#  Discretization
est = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal')
est.fit(data_np)
discretized_data = est.transform(data_np)

plot(discretized_data, 'Дискретизация')

print('\n\nKBinsDiscretizer\nКрая диапазонов: ')
for i, col in enumerate(df.columns):
    print(col + ": [ ", end="")
    for val in est.bin_edges_[i]:
        print('{0:.1f}'.format(val), end=" ")
    print(']')