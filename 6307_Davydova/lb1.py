import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df = df.drop(columns =['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])
print(df) #Вывод датафрейма с данными для лаб. работы. Должно быть 299 наблюдений и 6 признаков

#Гистограммы

n_bins = 20

fig, axs = plt.subplots(2,3)

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



data = df.to_numpy(dtype='float')

#до изм
print(np.mean(data, axis=0))
print(np.var(data, axis=0))


#setting
scaler = preprocessing.StandardScaler().fit(data)

#до изм scaler
print(scaler.mean_)
print(scaler.var_)

#standartisation
data_scaled = scaler.transform(data)

#гистограммы новых данных
fig, axs = plt.subplots(2,3)
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
#plt.show()

#после изм
print(np.mean(data_scaled, axis=0))
print(np.var(data_scaled, axis=0))



#после изм 150
scaler = preprocessing.StandardScaler().fit(data[:150,:])
data_scaled = scaler.transform(data)


print(np.mean(data_scaled[:150,:], axis=0))
print(np.var(data_scaled[:150,:], axis=0))

min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)

plot_hists(data_min_max_scaled)


#print(min_max_scaler.data_min_)
#print(min_max_scaler.data_max_)


max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaler = max_abs_scaler.transform(data)



robust_scaler = preprocessing.RobustScaler().fit(data)
data_robust_scaler = robust_scaler.transform(data)


plot_hists(data_max_abs_scaler)
plot_hists(data_robust_scaler)


range_scaler = preprocessing.MinMaxScaler().fit(data)
data_range_scaler = range_scaler.transform(data)*15-5

plot_hists(data_range_scaler)



quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100,
random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

plot_hists(data_quantile_scaled)


quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100,
random_state=0, output_distribution='normal').fit(data)
data_quantile_scaled2 = quantile_transformer.transform(data)

plot_hists(data_quantile_scaled2)



power_scaled = preprocessing.PowerTransformer().fit(data)
data_power_scaled = power_scaled.transform(data)

plot_hists(data_power_scaled)



discret = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal').fit(data)
data_discret = discret.transform(data)

plot_hists(data_discret)
print(discret.bin_edges_)



plt.show()



