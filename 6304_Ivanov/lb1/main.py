# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns=['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])
print(df)

# %%
def plot_hist(data, title):
    n_bins = 20
    fig, axs = plt.subplots(2,3, figsize=(15, 10))
    fig.suptitle(title, fontsize=20)
    axs[0, 0].hist(data[:,0], bins = n_bins)
    axs[0, 0].set_title('age')
    axs[0, 1].hist(data[:,1], bins = n_bins)
    axs[0, 1].set_title('creatinine_phosphokinase')
    axs[0, 2].hist(data[:,2], bins = n_bins)
    axs[0, 2].set_title('ejection_fraction')
    axs[1, 0].hist(data[:,3], bins = n_bins)
    axs[1, 0].set_title('platelets')
    axs[1, 1].hist(data[:,4], bins = n_bins)
    axs[1, 1].set_title('serum_creatinine')
    axs[1, 2].hist(data[:,5], bins = n_bins)
    axs[1, 2].set_title('serum_sodium')

    plt.savefig('{}.png'.format(title))


# %%
data = df.to_numpy(dtype='float')
plot_hist(data, 'Input data')


# %%
scaler = preprocessing.StandardScaler().fit(data[:150,:])
data_scaled = scaler.transform(data)
plot_hist(data_scaled, 'Standardized data')


# %%
full_scaler = preprocessing.StandardScaler()
full_data_scaled = full_scaler.fit_transform(data)

print('\n>>>data.mean/data.std')
print(data.mean(axis=0))
print(data.std(axis=0))

print('\n>>>data_scaled.mean/data_scaled.std')
print(data_scaled.mean(axis=0))
print(data_scaled.std(axis=0))

print('\n>>>full_data_scaled.mean/full_data_scaled.std')
print(full_data_scaled.mean(axis=0))
print(full_data_scaled.std(axis=0))

print('\n>>>scaler.mean_/scaler.var_')
print(scaler.mean_)
print(scaler.var_)

print('\n>>>full_scaler.mean_/full_scaler.var_')
print(full_scaler.mean_)
print(full_scaler.var_)


# %%
min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)    
plot_hist(data_min_max_scaled, 'MinMaxScaled data')
print('\n>>>min_max_scaler')
print(min_max_scaler.data_min_)
print(min_max_scaler.data_max_)


# %%
max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaled = max_abs_scaler.transform(data)
plot_hist(data_min_max_scaled, 'MaxAbsScaled data')


# %%
robust_scaler = preprocessing.RobustScaler().fit(data)
data_robust_scaled = robust_scaler.transform(data)
plot_hist(data_robust_scaled, 'Robusted data')


# %%
def range_5_10(data): return preprocessing.MinMaxScaler().fit(data).transform(data)*15-5

plot_hist(range_5_10(data), '[-5,10] data')


# %%
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100, random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

plot_hist(data_quantile_scaled, 'UniformQuantileTransformered data')


# %%
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100, random_state=0, output_distribution="normal").fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

plot_hist(data_quantile_scaled, 'NormalQuantileTransformered data')


# %%
power_transformer = preprocessing.PowerTransformer().fit(data)
data_power_scaled = power_transformer.transform(data)

plot_hist(data_quantile_scaled, 'PowerTransformered data')


# %%
discretizer = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal')
discretized_data = discretizer.fit_transform(data)
plot_hist(discretized_data, 'Discretized data')
print('\n>>>bin_edges_')
print(discretizer.bin_edges_)


# %%



