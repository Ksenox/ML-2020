import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, KBinsDiscretizer


# To make plots on Linux-based systems
matplotlib.use('TkAgg')


# Plot histograms
def plot_hist(arr, n_bins=20):
    if type(arr) is not np.ndarray and arr.shape[1] != 6:
        raise ValueError('Numpy array of (x, 6) shape is expected!')

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    axs[0, 0].hist(arr[:, 0], bins=n_bins)
    axs[0, 0].set_title('age')

    axs[0, 1].hist(arr[:, 1], bins=n_bins)
    axs[0, 1].set_title('creatinine_phosphokinase')

    axs[0, 2].hist(arr[:, 2], bins=n_bins)
    axs[0, 2].set_title('ejection_fraction')

    axs[1, 0].hist(arr[:, 3], bins=n_bins)
    axs[1, 0].set_title('platelets')

    axs[1, 1].hist(arr[:, 4], bins=n_bins)
    axs[1, 1].set_title('serum_creatinine')

    axs[1, 2].hist(arr[:, 5], bins=n_bins)
    axs[1, 2].set_title('serum_sodium')

    plt.show()


df = pd.read_csv('../../../heart_failure_clinical_records_dataset.csv')
df = df.drop(columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT'])
print(df.head(7))

data = df.to_numpy(dtype=float)

plot_hist(data)


# DATA STANDARDIZATION
# MinMaxScaler
scaler = StandardScaler().fit(data[:150, :])
print(f'{scaler.mean_}\n{scaler.var_}')
data_scaled = scaler.transform(data)

data_scaled_full = scaler.fit_transform(data)

plot_hist(data_scaled)
plot_hist(data_scaled_full)

cols = df.columns.to_numpy()
for i in range(len(cols)):
    print('Column ' + cols[i])
    print('==Mean== before: {}; after: {};'.format(data[:, i].mean(), data_scaled[:, i].mean()))
    print(' ==STD== before: {}; after: {};'.format(data[:, i].std(), data_scaled[:, i].std()))


# RANGE CASTING
# MinMaxScaler
min_max_scaler = MinMaxScaler()
data_min_max_scaled = min_max_scaler.fit_transform(data)
print(f'{min_max_scaler.data_min_}\n{min_max_scaler.data_max_}')

plot_hist(data_min_max_scaled)

# MaxAbsScaler & RobustScaler
data_max_abs_scaled = MaxAbsScaler().fit_transform(data)
data_robust_scaled = RobustScaler().fit_transform(data)
plot_hist(data_max_abs_scaled)
plot_hist(data_robust_scaled)


# Scale to [-5, 10]
def fit_range(arr, min_val=-5, max_val=10):
    if type(arr) is not np.ndarray:
        raise ValueError('Numpy array is expected!')

    scaled = np.empty(arr.shape)
    rng = max_val - min_val
    for col in range(arr.shape[1]):
        min_, max_ = np.min(arr[:, col]), np.max(arr[:, col])
        scaled[:, col] = [(x - min_) / (max_ - min_) * rng + min_val for x in arr[:, col]]

    return scaled


data_scaled_5_10 = fit_range(data)
plot_hist(data_scaled_5_10)


# NON-LINEAR TRANSFORMATION
data_scaled_quantile = QuantileTransformer(n_quantiles=100,
                                           random_state=0) \
    .fit_transform(data)
plot_hist(data_scaled_quantile)

data_scaled_quantile_normal = QuantileTransformer(n_quantiles=100,
                                                  random_state=0,
                                                  output_distribution='normal') \
    .fit_transform(data)
plot_hist(data_scaled_quantile_normal)

data_scaled_power = PowerTransformer().fit_transform(data)
plot_hist(data_scaled_power)


# FEATURE SAMPLING
sampler = KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal')
data_discrete = sampler.fit_transform(data)
print(sampler.bin_edges_)
plot_hist(data_discrete)
