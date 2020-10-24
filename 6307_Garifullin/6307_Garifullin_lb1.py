import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def showPlot(data):
  n_bins = 20
  fig, axs = plt.subplots(2,3)
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
  plt.show()

def showDataInfo(data, dataName):
  print(dataName, ': ')
  print('mean: ', np.mean(data, 0))
  print('std: ', np.std(data, 0))
  showPlot(data)

def getData():
  df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
  df = df.drop(columns = ['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])
  print(df)
  return df.to_numpy()

def stdScaleData(data, scaler, scalerName):
  scaledData = scaler.transform(data)
  print(scalerName, ':')
  print('mean_: ', scaler.mean_)
  print('var_: ', scaler.var_)
  showDataInfo(scaledData, scalerName)

def rangeScaleData(data, scaler):
  scaledData = scaler.transform(data)
  showPlot(scaledData)

def scalefromMinus5to10(data):
  scaledData = preprocessing.MinMaxScaler(feature_range=(-5, 10)).fit_transform(data)
  return scaledData

data = getData()

showDataInfo(data, 'Data')
standardScaler150 = preprocessing.StandardScaler().fit(data[:150,:])
stdScaleData(data, standardScaler150, 'standardScaler150')

standardScaler = preprocessing.StandardScaler().fit(data)
stdScaleData(data, standardScaler, 'standardScaler')

minMaxScaler = preprocessing.MinMaxScaler().fit(data)
rangeScaleData(data, minMaxScaler)
print('min:', minMaxScaler.data_min_)
print('max:', minMaxScaler.data_max_)

maxAbsScaler = preprocessing.MaxAbsScaler().fit(data)
rangeScaleData(data, maxAbsScaler)

robustScaler = preprocessing.RobustScaler().fit(data)
rangeScaleData(data, robustScaler)

showPlot(scalefromMinus5to10(data)) 

showPlot(preprocessing.QuantileTransformer(n_quantiles = 100, random_state=0).fit_transform(data))
showPlot(preprocessing.QuantileTransformer(n_quantiles = 10, random_state=0).fit_transform(data))
showPlot(preprocessing.QuantileTransformer(n_quantiles = 100, random_state=0, output_distribution='normal').fit_transform(data))
showPlot(preprocessing.PowerTransformer().fit_transform(data))

kBinsDiscretizer = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal').fit(data)
print(kBinsDiscretizer.bin_edges_)
showPlot(kBinsDiscretizer.transform(data))