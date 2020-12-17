import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from sklearn import preprocessing

columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT']
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns=columns)
print(df)  # Вывод датафрейма с данными для лаб. работы. Должно быть 299 наблюдений и 6 признаков


def plot_hist(data):
    n_bins = 20
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].hist(data[:, 0], bins=n_bins)
    axs[0, 0].set_title('age')
    axs[0, 1].hist(data[:, 1], bins=n_bins)
    axs[0, 1].set_title('creatinine_phosphokinase')
    axs[0, 2].hist(data[:, 2], bins=n_bins)
    axs[0, 2].set_title('ejection_fraction')
    axs[1, 0].hist(data[:, 3], bins=n_bins)
    axs[1, 0].set_title('platelets')
    axs[1, 1].hist(data[:, 4], bins=n_bins)
    axs[1, 1].set_title('serum_creatinine')
    axs[1, 2].hist(data[:, 5], bins=n_bins)
    axs[1, 2].set_title('serum_sodium')

    plt.show()


data = df.to_numpy(dtype='float')
# plot_hist(data)

used_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']


def measure(innerData):
    for i, column in enumerate(used_columns):
        values = innerData[:, i]
        print(f'column: "{column}", mode: {mode(values)}, min: {min(values)}, max: {max(values)}')
    print('-' * 80)


def print_array(array):
    [print(el, end=', ') for el in array]
    print()


measure(data)

scaler = preprocessing.StandardScaler().fit(data[:150, :])
data_scaled = scaler.transform(data)
# plot_hist(data_scaled)


measure(data_scaled)

full_scaler = preprocessing.StandardScaler()
full_data_scaled = full_scaler.fit_transform(data)

print('data.mean : data.std')
print_array(data.mean(axis=0))
print_array(data.std(axis=0))

print('data_scaled.mean : data_scaled.std')
print_array(data_scaled.mean(axis=0))
print_array(data_scaled.std(axis=0))

print('full_data_scaled.mean : full_data_scaled.std')
print_array(full_data_scaled.mean(axis=0))
print_array(full_data_scaled.std(axis=0))

print('scaler.mean_ : scaler.var_')
print_array(scaler.mean_)
print_array(scaler.var_)

print('full_scaler.mean_ : full_scaler.var_')
print_array(full_scaler.mean_)
print_array(full_scaler.var_)

min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)
# plot_hist(data_min_max_scaled)
print('min_max_scaler')
print_array(min_max_scaler.data_min_)
print_array(min_max_scaler.data_max_)

max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaled = max_abs_scaler.transform(data)
# plot_hist(data_min_max_scaled)


robust_scaler = preprocessing.RobustScaler().fit(data)
data_robust_scaled = robust_scaler.transform(data)


# plot_hist(data_robust_scaled)


def range_5_10(data):
    return preprocessing.MinMaxScaler().fit(data).transform(data) * 15 - 5


# plot_hist(range_5_10(data))


quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

# plot_hist(data_quantile_scaled)


quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal").fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

# plot_hist(data_quantile_scaled)


power_transformer = preprocessing.PowerTransformer().fit(data)
data_power_scaled = power_transformer.transform(data)

# plot_hist(data_quantile_scaled)

discretizer = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal')
discretized_data = discretizer.fit_transform(data)
plot_hist(discretized_data)
print('bin_edges_')
print_array(discretizer.bin_edges_)
