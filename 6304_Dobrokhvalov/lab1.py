import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

np.set_printoptions(precision=3)
pd.set_option('precision', 3)

# Загрузка данных
print('\n\tЗагрузка данных')

saveFigures = False

df = pd.read_csv('heart_failure_clinical_records_dataset.csv').drop(columns = \
['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])

print(df)

def plot_hists(data, title='Figure', scaled_platelets=1):
    fig, axs = plt.subplots(2,3)
    fig.suptitle(title, fontsize=16)
    fig.set_figheight(7)
    fig.set_figwidth(10)
    axs[0, 0].hist(data[:,0], bins = n_bins, linewidth=0.5, edgecolor='black')
    axs[0, 0].set_title('age')
    axs[0, 1].hist(data[:,1], bins = n_bins, linewidth=0.5, edgecolor='black')
    axs[0, 1].set_title('creatinine_phosphokinase')
    axs[0, 2].hist(data[:,2], bins = n_bins, linewidth=0.5, edgecolor='black')
    axs[0, 2].set_title('ejection_fraction')
    axs[1, 0].hist(data[:,3]/scaled_platelets, bins = n_bins, linewidth=0.5, edgecolor='black')
    axs[1, 0].set_title('platelets{}'.format('' if scaled_platelets==1 else ' *{}'.format(scaled_platelets)))
    axs[1, 1].hist(data[:,4], bins = n_bins, linewidth=0.5, edgecolor='black')
    axs[1, 1].set_title('serum_creatinine')
    axs[1, 2].hist(data[:,5], bins = n_bins, linewidth=0.5, edgecolor='black')
    axs[1, 2].set_title('serum_sodium')
    return fig, axs

n_bins = 20
to_plot = df.to_numpy()
plot_hists(to_plot, 'Original data', 10**3)
if saveFigures:
    plt.savefig('plots/lab1_Original data.png')

print('\nMode:')
print(df.mode())

data = df.to_numpy(dtype='float')

# Стандартизация данных
print('\n\tСтандартизация данных')

scaler = preprocessing.StandardScaler().fit(data[:150,:])
data_scaled = scaler.transform(data)
df_scalled = pd.DataFrame(data_scaled)
df_scalled.columns = df.columns
plot_hists(data_scaled, 'Standart Scaler')
if saveFigures:
    plt.savefig('plots/lab1_Standart Scaler.png')

print('\nBefore scale:')
print(df.describe())

print('\nAfter scale:')
print(df_scalled.describe())

print('\nStandardScaler (150 ). Mean & Var:')
print(scaler.mean_)
print(scaler.var_)


full_scaler = preprocessing.StandardScaler()
full_data_scaled = full_scaler.fit_transform(data)
df_full_scalled = pd.DataFrame(full_data_scaled)
df_full_scalled.columns = df.columns

print('\nAfter full scale:')
print(df_full_scalled.describe())

print('\nStandardScaler (full data). Mean & Var:')
print(full_scaler.mean_)
print(full_scaler.var_)

# Приведение к диапазону
print('\n\tПриведение к диапазону')

min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)

plot_hists(data_min_max_scaled, 'MinMaxScaler')
if saveFigures:
    plt.savefig('plots/lab1_MinMaxScaler.png')

print('\nMinMaxScaler. Min & Max')
print(min_max_scaler.data_min_)
print(min_max_scaler.data_max_)

max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaler = max_abs_scaler.transform(data)

plot_hists(data_max_abs_scaler, 'MaxAbsScaler')
if saveFigures:
    plt.savefig('plots/lab1_MaxAbsScaler.png')

print('\nMaxAbsScaler. Max abs')
print(max_abs_scaler.max_abs_)

robust_scaler = preprocessing.RobustScaler().fit(data)
data_robust_scaler = robust_scaler.transform(data)

plot_hists(data_robust_scaler, 'RobustScaler')
if saveFigures:
    plt.savefig('plots/lab1_RobustScaler.png')

print('\nRobustScaler. Center & Scale')
print(robust_scaler.center_)
print(robust_scaler.scale_)

def range_5_10(data):
    custom_scaler = preprocessing.MinMaxScaler().fit(data)
    return custom_scaler.transform(data)*15-5

data_custom_scaled = range_5_10(data)

plot_hists(data_custom_scaled, 'CustomScaler')
if saveFigures:
    plt.savefig('plots/lab1_CustomScaler.png')


# Нелинейные преобразования
print('\n\tНелинейные преобразования')

uniform_quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100, random_state=0).fit(data)
data_uniform_quantile_scaled = uniform_quantile_transformer.transform(data) # попытка привести к равномерному
plot_hists(data_uniform_quantile_scaled, 'UniformQuantileTransformer')
if saveFigures:
    plt.savefig('plots/lab1_UniformQuantileTransformer.png')

normal_quantile_transformer = preprocessing \
    .QuantileTransformer(n_quantiles = data.shape[0],output_distribution='normal').fit(data)
data_normal_quantile_scaled = normal_quantile_transformer.transform(data)
plot_hists(data_normal_quantile_scaled, 'NormalQuantileTransformer')
if saveFigures:
    plt.savefig('plots/lab1_NormalQuantileTransformer.png')

power_transformer = preprocessing.PowerTransformer().fit(data)
data_power_scaled = power_transformer.transform(data) # попытка привести к равномерному
plot_hists(data_power_scaled, 'PowerTransformer')
if saveFigures:
    plt.savefig('plots/lab1_PowerTransformer.png')

# Дискретизация признаков
print('\n\tДискретизация признаков')

def to_categorial(data_pd_series, n_bins=10):
    bins_transform = data_pd_series.to_numpy().reshape(-1,1)
    descritizer = preprocessing.KBinsDiscretizer(n_bins=n_bins).fit(bins_transform)
    tmp1 = pd.DataFrame(descritizer.transform(bins_transform).todense()).stack()
    return pd.Series(pd.Categorical(tmp1[tmp1!=0].index.get_level_values(1))), descritizer.bin_edges_

age_categorial, age_bin_edges_ = to_categorial(df.age,3)
print('\nAge. nbins = 3')
print(age_bin_edges_)

creatinine_phosphokinase_categorial, creatinine_phosphokinase_bin_edges_ = to_categorial(df.creatinine_phosphokinase, 4)
print('\nCreatinine phosphokinase. nbins = 4')
print(creatinine_phosphokinase_bin_edges_)


ejection_fraction_categorial, ejection_fraction_bin_edges_ = to_categorial(df.ejection_fraction, 3)
print('\nEjection fraction. nbins = 3')
print(ejection_fraction_bin_edges_)


platelets_categorial, platelets_bin_edges_ = to_categorial(df.platelets, 10)
print('\nPlatelets. nbins = 10')
print(platelets_bin_edges_)


serum_creatinine_categorial, serum_creatinine_bins_edges_ = to_categorial(df.serum_creatinine, 2)
print('\nSerum creatinine. nbins = 2')
print(serum_creatinine_bins_edges_)

serum_sodium_categorial, serum_sodium_bins_edges_ = to_categorial(df.serum_sodium, 4)
print('\nSerum sodium. nbins = 4')
print(serum_sodium_bins_edges_)


merged_categorial = np.vstack((age_categorial, \
    creatinine_phosphokinase_categorial, ejection_fraction_categorial, \
    platelets_categorial, serum_creatinine_categorial, serum_sodium_categorial)).T

plot_hists(merged_categorial, "MergedCategorial")
if saveFigures:
    plt.savefig('plots/lab1_MergedCategorial.png')

plt.show()