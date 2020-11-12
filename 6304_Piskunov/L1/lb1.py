import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing


def show_plt(_data, _bins):
    _fig, _axs = plt.subplots(2, 3)
    _axs[0, 0].hist(_data[:, 0], bins=_bins, linewidth=0.3, edgecolor='red')
    _axs[0, 0].set_title('age')
    _axs[0, 1].hist(_data[:, 1], bins=_bins, linewidth=0.3, edgecolor='red')
    _axs[0, 1].set_title('creatinine_phosphokinase')
    _axs[0, 2].hist(_data[:, 2], bins=_bins, linewidth=0.3, edgecolor='red')
    _axs[0, 2].set_title('ejection_fraction')
    _axs[1, 0].hist(_data[:, 3], bins=_bins, linewidth=0.3, edgecolor='red')
    _axs[1, 0].set_title('platelets')
    _axs[1, 1].hist(_data[:, 4], bins=_bins, linewidth=0.3, edgecolor='red')
    _axs[1, 1].set_title('serum_creatinine')
    _axs[1, 2].hist(_data[:, 5], bins=_bins, linewidth=0.3, edgecolor='red')
    _axs[1, 2].set_title('serum_sodium')
    plt.show()


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT'])
# print(df)

n_bins = 20
used_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(df['age'].values, bins=n_bins, linewidth=0.3, edgecolor='red')
axs[0, 0].set_title('age')
axs[0, 1].hist(df['creatinine_phosphokinase'].values, bins=n_bins, linewidth=0.3, edgecolor='red')
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(df['ejection_fraction'].values, bins=n_bins, linewidth=0.3, edgecolor='red')
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(df['platelets'].values, bins=n_bins, linewidth=0.3, edgecolor='red')
axs[1, 0].set_title('platelets')
axs[1, 1].hist(df['serum_creatinine'].values, bins=n_bins, linewidth=0.3, edgecolor='red')
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(df['serum_sodium'].values, bins=n_bins, linewidth=0.3, edgecolor='red')
axs[1, 2].set_title('serum_sodium')
plt.show()

data = df.to_numpy(dtype='float')

scaler = preprocessing.StandardScaler().fit(data[:150, :])
scaler_full = preprocessing.StandardScaler()
min_max_scaler = preprocessing.MinMaxScaler()
max_abs_scaler = preprocessing.MaxAbsScaler()
robust_scaler = preprocessing.RobustScaler()
scaler_custom = preprocessing.MinMaxScaler().fit(data)
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0)
quantile_transformer_normal = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0,
                                                                output_distribution='normal')
power_transformer = preprocessing.PowerTransformer()
kbins_discretizer = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal')


data_scaled = scaler.transform(data)
data_full = scaler_full.fit_transform(data)
data_min_max_scaled = min_max_scaler.fit_transform(data)
data_abs_scaled = max_abs_scaler.fit_transform(data)
data_robust_scaled = robust_scaler.fit_transform(data)
data_custom = scaler_custom.transform(data)*15-5
data_quantile_scaled = quantile_transformer.fit_transform(data)
data_quantile_scaled_normal = quantile_transformer_normal.fit_transform(data)
data_powered = power_transformer.fit_transform(data)
data_discretized = kbins_discretizer.fit_transform(data)

show_plt(data_scaled, n_bins)
show_plt(data_min_max_scaled, n_bins)
show_plt(data_abs_scaled, n_bins)
show_plt(data_robust_scaled, n_bins)
show_plt(data_custom, n_bins)
show_plt(data_quantile_scaled, n_bins)
show_plt(data_quantile_scaled_normal, n_bins)
show_plt(data_powered, n_bins)
show_plt(data_discretized, n_bins)

mean = [np.mean(i) for i in data.T]
stddev = [np.std(j) for j in data.T]
mean_after = [np.mean(i) for i in data_scaled.T]
stddev_after = [np.std(j) for j in data_scaled.T]
mean_full = [np.mean(i) for i in data_full.T]
stddev_full = [np.std(j) for j in data_full.T]


print('------------------------------------------ \nMean\n')
for i in range(len(mean)):
    print(f'{scaler.mean_[i]}         {mean[i]}         {mean_after[i]}         {mean_full[i]}\n')
print('------------------------------------------ \nDeviation\n')
for i in range(len(stddev)):
    print(f'{scaler.var_[i]}         {stddev[i]}         {stddev_after[i]}         {stddev_full[i]}\n')

print()
for i in range(len(used_columns)):
    print(f'{used_columns[i]}         {min_max_scaler.data_max_[i]}         {min_max_scaler.data_min_[i]}')


print()
for i in range(len(used_columns)):
    print(f'{used_columns[i]}         {kbins_discretizer.bin_edges_[i]}')
