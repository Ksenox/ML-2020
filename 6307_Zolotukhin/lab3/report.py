#!/usr/bin/env python
# coding: utf-8

# ## Загрузка данных

# In[53]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams['figure.facecolor'] = 'white'


# In[54]:


all_data = pd.read_csv('dataset_group.csv', header=None)
# В файле нет строки с названием столбцов, поэтому параметр header равен None.
# Интерес представляет информация об id покупателя - столбец с названием 1
# Название купленного товара хранится в столбце с названием 2


# In[55]:


unique_id = list(set(all_data[1]))
print(f'Buyers: {len(unique_id)}') # Выведем количество id


# In[56]:


items = list(set(all_data[2]))
print(f'Items: {len(items)}') # Выведем количество товаров


# In[57]:


dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in
items] for id in unique_id]


# ## Подготовка данных

# In[58]:


from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)


# In[59]:


print(df)


# Исходные данные были представлены в виде таблицы с колонками: дата, id покупателя и наименование покупки.
# Теперь данные представлены так, что для каждого покупателя (наименование ряда) существуют колонки, каждая
# из которых представляет собой один из продуктов, а значения для каждого покупателя - купил он этот продукт
# или нет. Проще говоря получилась "тепловая карта" продуктов для покупателей.

# ## Ассоциативный анализ с использованием алгоритма Apriori

# In[60]:


from mlxtend.frequent_patterns import apriori

results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x)) # Добавление размера набора
print(results)


# С помощью алгоритма apriori мы выделили часто встречающиеся сочетания (наборы) продуктов, которые покупают люди
# (в данном датасете), в том числе наборы из одного продукта. Частоту, которая считается "частой", мы задали параметром
# min_support. Частота = количество строк с всеми True для набора / общее количество строк.

# In[61]:


results = apriori(df, min_support=0.3, use_colnames=True, max_len=1)
print(results)


# In[62]:


results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
results = results[results['length'] == 2]
print(results)
print('\nCount of result itemstes = ',len(results))


# Посчитаем количество наборов при различных уровнях поддержки. Начальное значение
# поддержки 0.05, шаг 0.01. Построим график зависимости количества наборов от уровня
# поддержки

# In[63]:


min_support_range = np.arange(0.05, 0.8, 0.01)

itemsets_lengths = []
for min_support in min_support_range:
    results = apriori(df, min_support=min_support, use_colnames=True)
    itemsets_lengths.append(len(results))

plt.figure()

plt.plot(min_support_range.tolist(), itemsets_lengths)

plt.show()


# Как видно из графика количество наборов значительно уменьшается с повышением границы, что
# логично.

# Определим значения уровня поддержки при котором перестают генерироваться наборы
# размера 1,2,3, и.т.д. Отметим полученные уровни поддержки на графике построенном
# выше.

# In[64]:


from math import isnan

itemsets_lengths = []
threshold_supports = []
threshold_lengths = []

last_itemset_len = len(df.columns)

for min_support in min_support_range:
    results = apriori(df, min_support=min_support, use_colnames=True)
    itemsets_lengths.append(len(results))

    results['length'] = results['itemsets'].apply(lambda x: len(x))
    current_itemset_max_len = results['length'].max()

    if isnan(current_itemset_max_len):
        current_itemset_max_len = 0

    if current_itemset_max_len < last_itemset_len:
        last_itemset_len = current_itemset_max_len
        threshold_supports.append(min_support)
        threshold_lengths.append(len(results))

plt.figure()

plt.plot(min_support_range.tolist(), itemsets_lengths)
plt.plot(threshold_supports, threshold_lengths, 'ro')

plt.show()


# Построим датасет только из тех элементов, которые попадают в наборы размером 1 при
# уровне поддержки 0.38

# In[65]:


results = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
new_items = [ list(elem)[0] for elem in results['itemsets']]
new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in
new_items] for id in unique_id]


# Приведём полученный датасет к формату, который можно обработать

# In[66]:


te = TransactionEncoder()
te_ary = te.fit(new_dataset).transform(new_dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print(df)


# Проведём ассоциативный анализ при уровне поддержки 0.3 для нового датасета.

# In[67]:


results = apriori(df, min_support=0.3, use_colnames=True)
print(results)


# Как видно в подборке присутствуют лишь 28 элементов, вместо 52 ранее, потому
# что из датасета были исключены элементы с частотой менее 0.38 и размером
# набора 1. Те элементы, что были получены присутствуют также в старой подборке.

# Проведём ассоциативный анализ при уровне поддержки 0.15 для нового датасета.

# In[68]:


results = apriori(df, min_support=0.15, use_colnames=True)
print(results)


# Выведем все наборы размер которых больше 1 и в котором есть 'yogurt' или 'waffles'

# In[69]:


results['length'] = results['itemsets'].apply(lambda x: len(x))
results = results[results['length'] > 1]
results = results[results['itemsets'].apply(lambda x: ('yogurt' in x) or ('waffles' in x))]
print(results)


# Построим датасет, из тех элементов, которые не попали в датасет в п. 6 и приведём его к
# удобному для анализа виду.

# In[70]:


# All data dataframe
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
all_df = pd.DataFrame(te_ary, columns=te.columns_)

# print(all_df)
# print(new_items) # From #6

diff_df = all_df[all_df.columns.difference(new_items)]
print(diff_df)

# print(new_items)
# print(diff_df.columns)


# Проведём анализ apriori для полученного датасета для уровней поддержки 0.3 и 0.15

# In[71]:


results = apriori(diff_df, min_support=0.3, use_colnames=True)
print(results)


# In[72]:


results = apriori(diff_df, min_support=0.15, use_colnames=True)
print(results)


# Напишем правило, для вывода всех наборов, в которых хотя бы два элемента начинаются
# на 's'

# In[73]:


# All data dataframe
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
all_df = pd.DataFrame(te_ary, columns=te.columns_)

results = apriori(all_df, min_support=0.1, use_colnames=True)
# print(results)
results = results[results['itemsets'].apply(
    lambda x: np.fromiter(
        map(lambda y: y.startswith('s'), x), dtype=bool
    ).sum() >= 2
)]
print(results)


# Напишем правило, для вывода всех наборов, для которых уровень поддержки изменяется
# от 0.1 до 0.25

# In[88]:


# All data dataframe
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
all_df = pd.DataFrame(te_ary, columns=te.columns_)

results = apriori(all_df, min_support=0.1, use_colnames=True)

results = results[np.logical_and(results.support <= 0.25, results.support >= 0.1)]
print(results)

