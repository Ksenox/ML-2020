#!/usr/bin/env python
# coding: utf-8

# # Лабораторная №1

# ## Загрузка данных

# In[5]:


import pandas as pd
import numpy as np

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

df = df.drop(columns = ['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])
                        
print(df) # Вывод датафрейма с данными для лаб. работы. Должно быть 299 наблюдений и 6 признаков


# In[6]:


import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)

n_bins = 20

fig, axs = plt.subplots(2,3)

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

plt.show()


# Значения по гистограммам:
# 
# | | Age | CP | EF | Platelets | SC | Serim Sodium |
# | --- |--- | --- | --- | --- | --- | --- |
# | Приблизительный Диапазон | \[40, 96\] | \[0, 8000\] | \[0 , 80\] | \[ 0, 800000\] | \[0, 9\] | \[0, 150\] |
# | Значение, близкое к среднему | 60 | 200 | 38 | 220000 | 1 | 136 |

# In[7]:


data = df.to_numpy(dtype='float')


# ## Стандартизация данных

# In[8]:


from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(data[:150,:])


# In[9]:


data_scaled = scaler.transform(data)


# In[10]:


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

plt.show()


# In[11]:


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

plt.show()


# ### Что изменилось и почему?
# 
# Изменились мат ожидания на 0, а также дисперсии стали сравнимы. Это произошло потому что у кажжого значения набора отняли мат ожидание а и разделили его на СКО. Нормализация произведена с целью, чтобы статистически сравнить величины разных размерностей и порядков.

# Расчет параметров:

# In[12]:


mean_age = data[:, 0].mean()
std_age = data[:, 0].std()
mean_cp = data[:, 1].mean()
std_cp = data[:, 1].std()
mean_ef = data[:, 2].mean()
std_ef = data[:, 2].std()
mean_plat = data[:, 3].mean()
std_plat = data[:, 3].std()
mean_sc = data[:, 4].mean()
std_sc = data[:, 4].std()
mean_ss = data[:, 5].mean()
std_ss = data[:, 5].std()

print(f'Age mean (no scaling): {mean_age}')
print(f'Age std (no scaling): {std_age}')
print(f'Cp mean (no scaling): {mean_cp}')
print(f'Cp std (no scaling): {std_cp}')
print(f'Ef mean (no scaling): {mean_ef}')
print(f'Ef std (no scaling): {std_ef}')
print(f'Plat mean (no scaling): {mean_plat}')
print(f'Plat std (no scaling): {std_plat}')
print(f'Sc mean (no scaling): {mean_sc}')
print(f'Sc std (no scaling): {std_sc}')
print(f'Ss mean (no scaling): {mean_ss}')
print(f'Ss std (no scaling): {std_ss}')

mean_age = data_scaled[:, 0].mean()
std_age = data_scaled[:, 0].std()
mean_cp = data_scaled[:, 1].mean()
std_cp = data_scaled[:, 1].std()
mean_ef = data_scaled[:, 2].mean()
std_ef = data_scaled[:, 2].std()
mean_plat = data_scaled[:, 3].mean()
std_plat = data_scaled[:, 3].std()
mean_sc = data_scaled[:, 4].mean()
std_sc = data_scaled[:, 4].std()
mean_ss = data_scaled[:, 5].mean()
std_ss = data_scaled[:, 5].std()

print(f'Age mean: {mean_age}')
print(f'Age std: {std_age}')
print(f'Cp mean: {mean_cp}')
print(f'Cp std: {std_cp}')
print(f'Ef mean: {mean_ef}')
print(f'Ef std: {std_ef}')
print(f'Plat mean: {mean_plat}')
print(f'Plat std: {std_plat}')
print(f'Sc mean: {mean_sc}')
print(f'Sc std: {std_sc}')
print(f'Ss mean: {mean_ss}')
print(f'Ss std: {std_ss}')


# In[13]:


import math

mean_age = scaler.mean_[0]
std_age = math.sqrt(scaler.var_[0])
mean_cp = scaler.mean_[1]
std_cp = math.sqrt(scaler.var_[1])
mean_ef = scaler.mean_[2]
std_ef = math.sqrt(scaler.var_[2])
mean_plat = scaler.mean_[3]
std_plat = math.sqrt(scaler.var_[3])
mean_sc = scaler.mean_[4]
std_sc = math.sqrt(scaler.var_[4])
mean_ss = scaler.mean_[5]
std_ss = math.sqrt(scaler.var_[5])

print(f'Age mean: {mean_age}')
print(f'Age std: {std_age}')
print(f'Cp mean: {mean_cp}')
print(f'Cp std: {std_cp}')
print(f'Ef mean: {mean_ef}')
print(f'Ef std: {std_ef}')
print(f'Plat mean: {mean_plat}')
print(f'Plat std: {std_plat}')
print(f'Sc mean: {mean_sc}')
print(f'Sc std: {std_sc}')
print(f'Ss mean: {mean_ss}')
print(f'Ss std: {std_ss}')


# Формула нормализации:
# $$ z = {{x - u} \over {\sigma}} $$
# 
# где x - случайна величина, u - среднее, $\sigma$ - СКО

# In[14]:


data_scaled = preprocessing.StandardScaler().fit(data[:,:])
data_scaled = scaler.transform(data)


# In[15]:


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

plt.show()


# In[16]:


import math

mean_age = scaler.mean_[0]
std_age = math.sqrt(scaler.var_[0])
mean_cp = scaler.mean_[1]
std_cp = math.sqrt(scaler.var_[1])
mean_ef = scaler.mean_[2]
std_ef = math.sqrt(scaler.var_[2])
mean_plat = scaler.mean_[3]
std_plat = math.sqrt(scaler.var_[3])
mean_sc = scaler.mean_[4]
std_sc = math.sqrt(scaler.var_[4])
mean_ss = scaler.mean_[5]
std_ss = math.sqrt(scaler.var_[5])

print(f'Age mean: {mean_age}')
print(f'Age std: {std_age}')
print(f'Cp mean: {mean_cp}')
print(f'Cp std: {std_cp}')
print(f'Ef mean: {mean_ef}')
print(f'Ef std: {std_ef}')
print(f'Plat mean: {mean_plat}')
print(f'Plat std: {std_plat}')
print(f'Sc mean: {mean_sc}')
print(f'Sc std: {std_sc}')
print(f'Ss mean: {mean_ss}')
print(f'Ss std: {std_ss}')


# ## Приведение к диапазону

# ### MinMaxScaler

# In[17]:


min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)


# In[18]:


fig, axs = plt.subplots(2,3)

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

plt.show()


# In[19]:


print(min_max_scaler.data_min_[0])
print(min_max_scaler.data_max_[0])
print('\n')
print(min_max_scaler.data_min_[1])
print(min_max_scaler.data_max_[1])
print('\n')
print(min_max_scaler.data_min_[2])
print(min_max_scaler.data_max_[2])
print('\n')
print(min_max_scaler.data_min_[3])
print(min_max_scaler.data_max_[3])
print('\n')
print(min_max_scaler.data_min_[4])
print(min_max_scaler.data_max_[4])
print('\n')
print(min_max_scaler.data_min_[5])
print(min_max_scaler.data_max_[5])


# ### MaxAbsScaler

# In[20]:


max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaled = max_abs_scaler.transform(data)


# In[21]:


fig, axs = plt.subplots(2,3)

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

plt.show()


# MaxAbsScaler делит данные на максимальное обсолютное значение, при это разброс не меняется.

# ### RobustScaler

# In[22]:


robust_scaler = preprocessing.RobustScaler().fit(data)
data_robust_scaled = robust_scaler.transform(data)


# In[23]:


fig, axs = plt.subplots(2,3)

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

plt.show()


# `RobustScaler` похож на `StandardScaler`, но для стандартизации использует не среднее и СКО, а медиану и IQR (Intequartile range, расстояние между первым и третьи квартилем, т.е. "ширину места, вероятность в которое попасть 50%"). Этот метод прведения более устойчив к выбросам, чем `StandardScaler`, поэтому так и называется.

# ### Приведение к промежутку \[-5, 10\]

# In[24]:


min_max_scaler = preprocessing.MinMaxScaler((-5, 10)).fit(data)
data_min_max_scaled = min_max_scaler.transform(data)


# In[25]:


fig, axs = plt.subplots(2,3)

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

plt.show()


# ## Нелинейные преобразования

# In[26]:


quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100,random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)


# In[27]:


fig, axs = plt.subplots(2,3)

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

plt.show()


# In[28]:


quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 10,random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)


# In[29]:


fig, axs = plt.subplots(2,3)

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

plt.show()


# `n_quantiles` влияет на то, насколько хорошо приведется исходные данные к равномерному распредлению (в данным случае). Это значение используется для задания размера шага при дискретизации оценочной CDF для исходных данных (CDF используется для приведения к равномерному распределению в `QuantileTransformer`). 

# ### Приведение к номарльному распределению

# In[30]:


quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100,random_state=0, output_distribution='normal').fit(data)
data_quantile_scaled = quantile_transformer.transform(data)


# In[31]:


fig, axs = plt.subplots(2,3)

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

plt.show()


# ### `PowerTransformer`

# In[32]:


power_transformer = preprocessing.PowerTransformer().fit(data)
data_power_scaled = power_transformer.transform(data)


# In[33]:


fig, axs = plt.subplots(2,3)

axs[0, 0].hist(data_power_scaled[:,0], bins = n_bins)c
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

plt.show()


# ## Дискретизация признаков

# In[58]:


biner = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal')
data_bined = biner.fit_transform(data)


# In[59]:


fig, axs = plt.subplots(2,3)

axs[0, 0].hist(data_bined[:,0])
axs[0, 0].set_title('age')

axs[0, 1].hist(data_bined[:,1])
axs[0, 1].set_title('creatinine_phosphokinase')

axs[0, 2].hist(data_bined[:,2])
axs[0, 2].set_title('ejection_fraction')

axs[1, 0].hist(data_bined[:,3])
axs[1, 0].set_title('platelets')

axs[1, 1].hist(data_bined[:,4])
axs[1, 1].set_title('serum_creatinine')

axs[1, 2].hist(data_bined[:,5])
axs[1, 2].set_title('serum_sodium')

plt.show()


# Количество столбиков = количество бинов (промежутков), на которые мы разбили данные.

# In[60]:


print(biner.bin_edges_)

