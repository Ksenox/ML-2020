import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def plot(data, title='Figure'):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    fig.suptitle(title, fontsize=16)

    axs[0, 0].hist(data[:, 0], bins=n_bins, linewidth=0.5, edgecolor='black')
    axs[0, 0].set_title('age')

    axs[0, 1].hist(data[:, 1], bins=n_bins, linewidth=0.5, edgecolor='black')
    axs[0, 1].set_title('creatinine_phosphokinase')

    axs[0, 2].hist(data[:, 2], bins=n_bins, linewidth=0.5, edgecolor='black')
    axs[0, 2].set_title('ejection_fraction')

    axs[1, 0].hist(data[:, 3], bins=n_bins, linewidth=0.5, edgecolor='black')
    axs[1, 0].set_title('platelets')

    axs[1, 1].hist(data[:, 4], bins=n_bins, linewidth=0.5, edgecolor='black')
    axs[1, 1].set_title('serum_creatinine')

    axs[1, 2].hist(data[:, 5], bins=n_bins, linewidth=0.5, edgecolor='black')
    axs[1, 2].set_title('serum_sodium')

    plt.show()


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT'])

n_bins = 20

data = df.to_numpy(dtype='float')

plot(data, 'Original data')

print(df) # Вывод датафрейма с данными для лаб. работы. Должно быть 299 наблюдений и 6 признаков

print('Mode: ', df.mode())

# Standart Scaler 150
scaler1 = preprocessing.StandardScaler().fit(data[:150, :])
data_scaled = scaler1.transform(data)
df_data_scaled = pd.DataFrame(data_scaled)
plot(data_scaled, 'Standart Scaler')

# Full Standart Scaler
scaler2 = preprocessing.StandardScaler()
full_data_scaled = scaler2.fit_transform(data)
df_full_data_scaled = pd.DataFrame(full_data_scaled)

print('Original', '\n', df.describe())
print('Standart Scaler 150', '\n', df_data_scaled.describe())
print('Standart Scaler', '\n', df_full_data_scaled.describe())

# MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)
plot(data_min_max_scaled, 'MinMaxScaler')

print(min_max_scaler.data_min_)
print(min_max_scaler.data_max_)

# MaxAbsScaler
max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaler = max_abs_scaler.transform(data)
plot(data_max_abs_scaler, 'MaxAbsScaler')

# RobustScaler
robust_scaler = preprocessing.RobustScaler().fit(data)
data_robust_scaler = robust_scaler.transform(data)
plot(data_robust_scaler, 'RobustScaler')


def custom_range(data):
    custom_data = preprocessing.MinMaxScaler().fit_transform(data)*15-5
    return custom_data


custom_data = custom_range(data)
plot(custom_data, 'Data range [-5 10]')

# QuantileTransformer
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)
plot(data_quantile_scaled, 'UniformQuantileTransformer')

data_normal_quantile_transformer = preprocessing \
    .QuantileTransformer(n_quantiles = data.shape[0], output_distribution='normal').fit_transform(data)
plot(data_normal_quantile_transformer, 'NormalQuantileTransformer')

# PowerTransformer
data_power_transformer = preprocessing.PowerTransformer().fit_transform(data)
plot(data_power_transformer, 'PowerTransformer')

# KBinsDiscretizer
discretized = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal')
discretized_data = discretized.fit_transform(data)
plot(discretized_data, 'KBinsDiscretizer')
print(discretized.bin_edges_)

