import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import matplotlib.pyplot as plt
from math import isnan

all_data = pd.read_csv('dataset_group.csv', header=None)
# В файле нет строки с названием столбцов, поэтому параметр header равен None.
# Интерес представляет информация об id покупателя - столбец с названием 1
# Название купленного товара хранится в столбце с названием 2

unique_id = list(set(all_data[1]))
# print(len(unique_id)) #Выведем количество id

items = list(set(all_data[2]))
# print(len(items)) #Выведем количество товаров

dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in
            items] for id in unique_id]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# print(df)

results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))  # добавление размера набора
# print(results)

results = apriori(df, min_support=0.3, use_colnames=True, max_len=1)
# print(results)

results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
results = results[results['length'] == 2]
# print(results)
# print('\nCount of result itemstes = ',len(results))

# min_support_range = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1,]
min_support_range = np.arange(0.05, 0.5, 0.01)
lens_apriori = []

max_level = None
results_point = []
results_min_level = []
for current_support in min_support_range:
    results = apriori(df, min_support=current_support, use_colnames=True)
    lens_apriori.append(len(results))
    results['length'] = results['itemsets'].apply(lambda x: len(x))

    if max_level is None:
        max_level = results['length'].max()
    else:
        while max_level > 0 and len(results[results['length'] == max_level]) == 0:
            # print(len(results))
            results_point.append(len(results))
            results_min_level.append(current_support)
            max_level -= 1

# plt.plot(lens_apriori)
# plt.show()

plt.plot(results_min_level, results_point, 'ro')
plt.plot(min_support_range, lens_apriori)
plt.show()

# 6. Построим датасет только из тех элементов, которые попадают в наборы размером 1 при
# уровне поддержки 0.38
results = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
new_items = [list(elem)[0] for elem in results['itemsets']]
new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in
                new_items] for id in unique_id]
new_dataset_2 = [[elem for elem in all_data[all_data[1] == id][2] if elem not in
                  new_items] for id in unique_id]

te = TransactionEncoder()
te_ary = te.fit(new_dataset).transform(new_dataset)
df_new = pd.DataFrame(te_ary, columns=te.columns_)

results = apriori(df_new, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
# print(results)
# print('\nCount of result itemstes = ', len(results))

# < 15%
results = apriori(df_new, min_support=0.15, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))  # добавление размера набора
results = results[results['length'] > 1]
results = results[results['itemsets'].apply(lambda x: ('yogurt' in x) or ('waffles' in x))]

# print(results)
# print('\nCount of result itemstes = ', len(results))

# Постройте датасет, из тех элементов, которые не попали в датасет в п. 6 и приведите его к
# удобному для анализа виду
te = TransactionEncoder()
te_ary = te.fit(new_dataset_2).transform(new_dataset_2)
df_new = pd.DataFrame(te_ary, columns=te.columns_)

# 11. Проведите анализ aprioti для полученного датасета
results = apriori(df_new, min_support=0.15, use_colnames=True)
#results['length'] = results['itemsets'].apply(lambda x: len(x))  # добавление размера набора
#print(results)
#print('\nCount of result itemstes = ', len(results))

# 12. Напишите правило, для вывода всех наборов, в которых хотя бы два элемента начинаются на 's'
#results = results[results['itemsets'].apply(lambda x: len([el for el in x if el.startswith('s')]) >= 2)]
#print(results)
#print('\nCount of result itemstes = ', len(results))

# 13. Напишите правило, для вывода всех наборов, для которых уровень поддержки изменяется от 0.1 до 0.25
results = results[results['support'].apply(lambda x: 0.1 <= x <= 0.25)]
print(results)
print('\nCount of result itemstes = ', len(results))