import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

df = df.drop(columns =
             ['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])

print(df) #Вывод датафрейма с данными для лаб. работы. Должно быть 299 наблюдений и 6 признаков

n_bins = 20

#fig, axs = plt.subplots(2,3)

#axs[0, 0].hist(df['age'].values, bins = n_bins)
#axs[0, 0].set_title('age')
#print('min {}'.format(df['age'].values.min()),
#      'max {}'.format(df['age'].values.max()))

#axs[0, 1].hist(df['creatinine_phosphokinase'].values, bins = n_bins)
#axs[0, 1].set_title('creatinine_phosphokinase')
#print('min {}'.format(df['creatinine_phosphokinase'].values.min()),
#      'max {}'.format(df['creatinine_phosphokinase'].values.max()))

#axs[0, 2].hist(df['ejection_fraction'].values, bins = n_bins)
#axs[0, 2].set_title('ejection_fraction')
#print('min {}'.format(df['ejection_fraction'].values.min()),
#      'max {}'.format(df['ejection_fraction'].values.max()))

#axs[1, 0].hist(df['platelets'].values, bins = n_bins)
#axs[1, 0].set_title('platelets')
#print('min {}'.format(df['platelets'].values.min()),
#      'max {}'.format(df['platelets'].values.max()))

#axs[1, 1].hist(df['serum_creatinine'].values, bins = n_bins)
#axs[1, 1].set_title('serum_creatinine')
#print('min {}'.format(df['serum_creatinine'].values.min()),
#      'max {}'.format(df['serum_creatinine'].values.max()))

#axs[1, 2].hist(df['serum_sodium'].values, bins = n_bins)
#axs[1, 2].set_title('serum_sodium')
#print('min {}'.format(df['serum_sodium'].values.min()),
#      'max {}'.format(df['serum_sodium'].values.max()))

#plt.show()

data = df.to_numpy(dtype='float')

#print('до стандартизации')
#print (np.mean(data[:150, :], axis=0))
#print (np.std(data[:150, :], axis=0))

#стандартизация
scaler = preprocessing.StandardScaler().fit(data)
data_scaled = scaler.transform(data)

min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)*15-5

quantile_transformer = preprocessing.PowerTransformer().fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

kbin_discret_transformer = preprocessing.KBinsDiscretizer([3,4,3,10,2,4], encode='ordinal').fit(data)
data_kbin_discret = kbin_discret_transformer.transform(data)

fig, axs = plt.subplots(2,3)

axs[0, 0].hist(data_kbin_discret[:,0], bins = n_bins)
axs[0, 0].set_title('age')

axs[0, 1].hist(data_kbin_discret[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')

axs[0, 2].hist(data_kbin_discret[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')

axs[1, 0].hist(data_kbin_discret[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')

axs[1, 1].hist(data_kbin_discret[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')

axs[1, 2].hist(data_kbin_discret[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')

print('')
print(kbin_discret_transformer.bin_edges_)
#print('после стандартизации')
#print (np.mean(data_scaled, axis=0))
#print (np.std(data_scaled, axis=0))
#print ('mean scaler', scaler.mean_)
#print ('var scaler', scaler.var_)
#print ('min', min_max_scaler.data_min_)
#print ('max', min_max_scaler.data_max_)

plt.show()

