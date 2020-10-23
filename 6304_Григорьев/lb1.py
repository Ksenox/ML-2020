# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
null.tpl [markdown]
# # Лабораторная работа №1. Предобработка данных
null.tpl [markdown]
# ## Загрузка данных

# %%
import pandas as pd
import numpy as np
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns = ['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])
df


# %%
import matplotlib.pyplot as plt
n_bins = 20
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
# plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace=1, hspace=1)
axs[0, 0].hist(df['age'].values, bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(df['creatinine_phosphokinase'].values, bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(df['ejection_fraction'].values, bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(df['platelets'].values, bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(df['serum_creatinine'].values, bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(df['serum_sodium'].values, bins = n_bins)
axs[1, 2].set_title('serum_sodium')
fig.tight_layout()
plt.show()


# %%
from tabulate import tabulate
min(df['age'].values)
display(tabulate([
        [min(df['age'].values), max(df['age'].values)],
        [min(df['creatinine_phosphokinase'].values), max(df['creatinine_phosphokinase'].values)],
        [min(df['ejection_fraction'].values), max(df['ejection_fraction'].values)],
        [min(df['platelets'].values), max(df['platelets'].values)],
        [min(df['serum_creatinine'].values), max(df['serum_creatinine'].values)],
        [min(df['serum_sodium'].values), max(df['serum_sodium'].values)]
    ],
    tablefmt='html',
    headers=['min', 'max'],
    showindex=['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']))

null.tpl [markdown]
# ## Стандартизация данных

# %%
from sklearn import preprocessing
data = df.values
scaler = preprocessing.StandardScaler().fit(data[:150,:])
data_scaled = scaler.transform(data)
data_scaled_full = preprocessing.StandardScaler().fit_transform(data)


# %%
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs[0, 0].hist(data_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
fig.tight_layout()
plt.show()


# %%
def get_mean_std_var(data):
    return [np.mean(col) for col in data.T], [np.std(col) for col in data.T]

mean_data, std_data = get_mean_std(data)
mean_data_scaled, std_data_scaled = get_mean_std(data_scaled)
mean_data_scaled_full, std_data_scaled_full = get_mean_std(data_scaled_full)


# %%
mean_data, std_data


# %%
mean_data_scaled, std_data_scaled


# %%
mean_data_scaled_full, std_data_scaled_full


# %%
scaler.mean_


# %%
scaler.var_

null.tpl [markdown]
# ## Приведение к диапазону

# %%
min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)


# %%
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs[0, 0].hist(data_min_max_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_min_max_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_min_max_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_min_max_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_min_max_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_min_max_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
fig.tight_layout()
plt.show()


# %%
min_max_scaler.data_min_


# %%
min_max_scaler.data_max_


# %%
max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaled = max_abs_scaler.transform(data)


# %%
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs[0, 0].hist(data_max_abs_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_max_abs_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_max_abs_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_max_abs_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_max_abs_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_max_abs_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
fig.tight_layout()
plt.show()


# %%
robust_scaler = preprocessing.RobustScaler().fit(data)
data_robust_scaled = robust_scaler.transform(data)


# %%
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs[0, 0].hist(data_robust_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_robust_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_robust_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_robust_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_robust_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_robust_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
fig.tight_layout()
plt.show()


# %%
data_min_max_scaled_custom = np.array([[((x - np.min(col)) / (np.max(col) - np.min(col))) * 15 - 5 for x in col] for col in data.T]).T
data_min_max_scaled_custom


# %%
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs[0, 0].hist(data_min_max_scaled_custom[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_min_max_scaled_custom[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_min_max_scaled_custom[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_min_max_scaled_custom[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_min_max_scaled_custom[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_min_max_scaled_custom[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
fig.tight_layout()
plt.show()

null.tpl [markdown]
# ## Нелинейные преобразования

# %%
quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100, random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)


# %%
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs[0, 0].hist(data_quantile_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_quantile_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_quantile_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_quantile_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_quantile_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_quantile_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
fig.tight_layout()
plt.show()


# %%
quantile_transformer_normal = preprocessing.QuantileTransformer(n_quantiles = 100, random_state=0, output_distribution='normal').fit(data)
data_quantile_scaled_normal = quantile_transformer_normal.transform(data)


# %%
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs[0, 0].hist(data_quantile_scaled_normal[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_quantile_scaled_normal[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_quantile_scaled_normal[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_quantile_scaled_normal[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_quantile_scaled_normal[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_quantile_scaled_normal[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
fig.tight_layout()
plt.show()


# %%
power_transformer = preprocessing.PowerTransformer().fit(data)
data_power_scaled = power_transformer.transform(data)


# %%
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs[0, 0].hist(data_power_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_power_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_power_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_power_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_power_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_power_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
fig.tight_layout()
plt.show()

null.tpl [markdown]
# ## Дискретизация признаков

# %%
bins_discretizer = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal').fit(data)
data_bins_discretized = bins_discretizer.transform(data)


# %%
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
axs[0, 0].hist(data_bins_discretized[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_bins_discretized[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_bins_discretized[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_bins_discretized[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_bins_discretized[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_bins_discretized[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')
fig.tight_layout()
plt.show()


# %%
bins_discretizer.bin_edges_


# %%



