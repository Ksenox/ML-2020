import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def draw_data(data, title=''):
    n_bins = 20
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].hist(data[:, 0], bins=n_bins)
    axs[0, 0].set_title('age')
    axs[0, 1].hist(data[:, 1], bins=n_bins)
    axs[0, 1].set_title('creatinine_phosphokinase')
    axs[0, 2].hist(data[:, 2], bins=n_bins)
    axs[0, 2].set_title('ejection_fraction')
    axs[1, 0].hist(data[:, 3], bins=n_bins)
    axs[1, 0].set_title('platelets')
    axs[1, 1].hist(data[:, 4], bins=n_bins)
    axs[1, 1].set_title('serum_creatinine')
    axs[1, 2].hist(data[:, 5], bins=n_bins)
    axs[1, 2].set_title('serum_sodium')
    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    df = df.drop(columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT'])
    data = df.to_numpy(dtype='float')

    draw_data(data, 'data')

    # Стандартизация данных
    scaler = preprocessing.StandardScaler()
    standard_data = scaler.fit_transform(data)
    draw_data(standard_data, 'standard scale')
    print(scaler.mean_, np.mean(data, axis=0), scaler.var_, np.var(data, axis=0), sep='\n')
    print()
    print(np.mean(standard_data, axis=0), np.var(standard_data, axis=0), sep='\n')
    print()

    # mean = np.mean(data, axis=0)
    # var = np.var(data, axis=0)
    # for i in range(len(mean)):
    #     print('zi = (xi - ', mean[i], ') / ', var[i], sep='')

    # Приведение к диапазону
    scaler = preprocessing.MinMaxScaler()
    draw_data(scaler.fit_transform(data), '[0, 1] scale')
    print(scaler.data_range_, scaler.data_min_, scaler.data_max_, sep='\n', end='\n\n')

    draw_data(preprocessing.MinMaxScaler(feature_range=[-5, 10]).fit_transform(data), '[-5, 10] scale')
    draw_data(preprocessing.MaxAbsScaler().fit_transform(data), 'max abs scale')
    draw_data(preprocessing.RobustScaler().fit_transform(data), 'robust scale')

    # Нелинейные преобразования
    draw_data(preprocessing.QuantileTransformer(n_quantiles=100, random_state=0).fit_transform(data),
              'uniform quant distribution')
    draw_data(
        preprocessing.QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal').fit_transform(
            data),
        'normal quant distribution')
    draw_data(preprocessing.PowerTransformer().fit_transform(data), 'normal power distribution')

    # Дискретизация признаков
    discretizer = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal')
    draw_data(discretizer.fit_transform(data), 'discr')
    print(discretizer.bin_edges_)
