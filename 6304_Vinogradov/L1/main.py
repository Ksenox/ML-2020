import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

info = []

n_bins = 40

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT'])

names = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

data = df.to_numpy(dtype='float')

print(df)  # Вывод датафрейма с данными для лаб. работы. Должно быть 299 наблюдений и 6 признаков


def draw_orig_hist(_axs, _indicies, _names):
    i = 0
    for index in _indicies:
        values, bins, not_used = _axs[index[0], index[1]].hist(df[_names[i]].values, bins=n_bins)
        _axs[index[0], index[1]].set_title(_names[i])
        append_info(info, _names[i], bins, values)
        i += 1


def draw_hist(_axs, _data, _indicies, _names):
    i = 0
    for index in _indicies:
        _axs[index[0], index[1]].hist(_data[:, i], bins=n_bins)
        _axs[index[0], index[1]].set_title(_names[i])
        i += 1


def append_info(info, name, bins, values):
    info.append({'name': name,
                 'left edge': bins[0],
                 'right edge': bins[-1],
                 'max': {
                     'value': np.max(values),
                     'left edge': bins[np.argmax(values)],
                     'right edge': bins[np.argmax(values) + 1]
                        }
                 })


def transform_to_range(X):
    low, high = -5, 10
    data = np.empty((len(X), 0), dtype=float)
    for i in range(len(X[0])):
        min = np.min(X[:, i])
        max = np.max(X[:, i])
        append_data = np.array((X[:, i] - min) / (max - min) * (high - low) + low)
        data = np.column_stack([data, append_data])
    return data



fig, axs = plt.subplots(2, 3)

draw_orig_hist(axs, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], names)

print('Traits info')
for elem in info:
    print(elem)
plt.show()
plt.close(fig)

mean_before = [np.mean(data[:, 0]), np.mean(data[:, 1]), np.mean(data[:, 2]),
               np.mean(data[:, 3]), np.mean(data[:, 4]), np.mean(data[:, 5])]
var_before = [np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]),
              np.var(data[:, 3]), np.var(data[:, 4]), np.var(data[:, 5])]

# Standart scaler

scaler = preprocessing.StandardScaler().fit(data[:150, ])
data_scaled = scaler.transform(data)

mean_after_150 = [np.mean(data_scaled[:, 0]), np.mean(data_scaled[:, 1]), np.mean(data_scaled[:, 2]),
                  np.mean(data_scaled[:, 3]), np.mean(data_scaled[:, 4]), np.mean(data_scaled[:, 5])]
var_after_150 = [np.var(data_scaled[:, 0]), np.var(data_scaled[:, 1]), np.var(data_scaled[:, 2]),
                 np.var(data_scaled[:, 3]), np.var(data_scaled[:, 4]), np.var(data_scaled[:, 5])]
scaler_meam_after_150 = scaler.mean_
scaler_var_after_150 = scaler.var_


names_tr = []
for name in names:
    names_tr.append(name + '_tr')

fig, axs = plt.subplots(2, 3)
draw_hist(axs, data_scaled, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], names_tr)
plt.show()
plt.close(fig)

data_scaled_full = scaler.fit_transform(data)

mean_after_full = [np.mean(data_scaled_full[:, 0]), np.mean(data_scaled_full[:, 1]), np.mean(data_scaled_full[:, 2]),
                   np.mean(data_scaled_full[:, 3]), np.mean(data_scaled_full[:, 4]), np.mean(data_scaled_full[:, 5])]
var_after_full = [np.var(data_scaled_full[:, 0]), np.var(data_scaled_full[:, 1]), np.var(data_scaled_full[:, 2]),
                  np.var(data_scaled_full[:, 3]), np.var(data_scaled_full[:, 4]), np.var(data_scaled_full[:, 5])]

print('Mean before', mean_before, 'Mean after 150', mean_after_150, 'Mean after full', mean_after_full,
      'Var before', var_before, 'Var after 150', var_after_150, 'Var after full', var_after_full,
      'Scaler mean 150', scaler_meam_after_150, 'Scaler var 150', scaler_var_after_150, 'Scaler mean', scaler.mean_,
      'Scaler var', scaler.var_, '\n', sep='\n')

names_tr_full = []
for name in names:
    names_tr_full.append(name + '_tr_full')

fig, axs = plt.subplots(2, 3)
draw_hist(axs, data_scaled_full, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], names_tr_full)
plt.show()
plt.close(fig)

# Min-max scaler

fig, axs = plt.subplots(2, 3)

# x_scaled[i] = ((x[i] - min(x)) / (max(x) - min (x)) * (max_range - min_range) + min_range

min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)

names_minmax = []
for name in names:
    names_minmax.append(name + '_minmax')

draw_hist(axs, data_min_max_scaled, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], names_minmax)

print('Traits', names, '\nMax', min_max_scaler.data_max_, '\nMin', min_max_scaler.data_min_, '\n')

# plt.show()
plt.close(fig)


# Max abs scaler
fig, axs = plt.subplots(2, 3)

# x_scaled[i] = x[i] / max_abs(x)

max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaled = max_abs_scaler.transform(data)

names_maxabs = []
for name in names:
    names_maxabs.append(name + '_maxabs')

draw_hist(axs, data_max_abs_scaled, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], names_maxabs)

plt.show()
plt.close(fig)


# Robust scaler
fig, axs = plt.subplots(2, 3)

# x_scaled[i] = (x[i] - median[i]) / IQR[i]

robust_scaler = preprocessing.RobustScaler().fit(data)
data_robust_scaled = robust_scaler.transform(data)

names_robust = []
for name in names:
    names_robust.append(name + '_robust')

draw_hist(axs, data_robust_scaled, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], names_robust)

plt.show()
plt.close(fig)


# Transform function to range (-5, 10)
fig, axs = plt.subplots(2, 3)

# x_scaled[i] = (x[i] - min(x)) / (max(x) - min (x)) * (10 - (-5)) + (-5)

data_func_scaled = transform_to_range(data)

names_func_scaled = []
for name in names:
    names_func_scaled.append(name + '_func_scaled')

draw_hist(axs, data_func_scaled, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], names_func_scaled)

plt.show()
plt.close(fig)


# Nonlinear transformations

fig, axs = plt.subplots(2, 3)

sorted_data = np.sort(data, axis=0)

quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0).fit(data)
data_quantile_scaled_uniform = quantile_transformer.transform(data)

names_quant_transform_uniform = []
for name in names:
    names_quant_transform_uniform.append(name + '_q_tr_uni')

draw_hist(axs, data_quantile_scaled_uniform, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], names_quant_transform_uniform)

plt.show()
plt.close(fig)

fig, axs = plt.subplots(2, 3)

quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=0)
data_quantile_scaled_normal = quantile_transformer.fit_transform(data)

names_quant_transform_normal = []
for name in names:
    names_quant_transform_normal.append(name + '_q_tr_norm')

draw_hist(axs, data_quantile_scaled_normal, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], names_quant_transform_normal)
plt.show()
plt.close(fig)


fig, axs = plt.subplots(2, 3)

power_transformer = preprocessing.PowerTransformer()
data_power_transformed = power_transformer.fit_transform(data)

names_power_transform = []
for name in names:
    names_power_transform.append(name + '_pw_tr')

draw_hist(axs, data_power_transformed, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], names_power_transform)

plt.show()
plt.close(fig)

# Bins discretizing

fig, axs = plt.subplots(2, 3)

bins_list = [3, 4, 3, 10 ,2 ,4]

discretized_data = np.empty((len(data), 0), dtype=float)

for i in range(len(bins_list)):
    bins_discretizer = preprocessing.KBinsDiscretizer(n_bins=bins_list[i], encode='ordinal', strategy='uniform')
    feature_arr = data[:, i]
    reshaped_arr = np.reshape(feature_arr, (-1, 1))
    transformed_data = bins_discretizer.fit_transform(reshaped_arr)
    axs[i // 3, i % 3].hist(transformed_data, bins=bins_list[i])
    print(names[i], bins_discretizer.bin_edges_)

plt.show()
plt.close(fig)

