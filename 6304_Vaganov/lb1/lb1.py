import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import stats


df = pd.read_csv('lab1/heart_failure_clinical_records_dataset.csv')
df = df.drop(columns =
['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])
print(df) #Вывод датафрейма с данными для лаб. работы. Должно быть 299наблюдений и 6 признаков

#Вывод гистограмм
def plot_hist(data):
    n_bins = 20
    fig, axs = plt.subplots(2,3, figsize=(15, 10))
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
    plt.tight_layout()
    plt.show()


#Вывод интервалов значений и моды
for column in df.columns:
    print(df[column].name)
    print("interval [{},{}], mode value {}\n".format(df[column].min(), df[column].max(), df[column].mode()))


data = df.to_numpy(dtype='float')
print(data)


scaler = preprocessing.StandardScaler().fit(data[:150,:])
data_scaled = scaler.transform(data)
plot_hist(data_scaled)
print(data_scaled.std(axis=0))
print(data_scaled.mean(axis=0))


for column in range(0, 6):
    print("interval [{},{}], mode value {}\n".format(data_scaled[:,column].min(), data_scaled[:,column].max(), stats.mode(data_scaled[:,column]).mode))


scaler = preprocessing.StandardScaler()
data_scaled = scaler.fit_transform(data)
plot_hist(data_scaled)
print(data_scaled.std(axis=0))
print(data_scaled.mean(axis=0))


scaler = preprocessing.MinMaxScaler().fit(data)
data_scaled = scaler.transform(data)
plot_hist(data_scaled)
print(scaler.data_min_)
print(scaler.data_max_)


scaler = preprocessing.MaxAbsScaler().fit(data)
data_scaled = scaler.transform(data)
plot_hist(data_scaled)


scaler = preprocessing.RobustScaler().fit(data)
data_scaled = scaler.transform(data)
plot_hist(data_scaled)


def range_5_10(data):
    custom_scaler = preprocessing.MinMaxScaler().fit(data)
    return custom_scaler.transform(data)*15-5
data_scaled = range_5_10(data)
plot_hist(data_scaled)


quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100,
random_state=0, output_distribution='normal').fit(data)
data_scaled = quantile_transformer.transform(data)
plot_hist(data_scaled)


transformer = preprocessing.PowerTransformer().fit(data)
data_scaled = transformer.transform(data)
plot_hist(data_scaled)


discretizer = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal')
data_scaled = discretizer.fit_transform(data)
plot_hist(data_scaled)
print(discretizer.bin_edges_)