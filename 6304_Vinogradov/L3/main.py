import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# Loading
all_data = pd.read_csv('dataset_group.csv', header=None)
# В файле нет строки с названием столбцов, поэтому параметр header равен None.
# Интерес представляет информация об id покупателя - столбец с названием 1
# Название купленного товара хранится в столбце с названием 2
unique_id = list(set(all_data[1]))
print('Кол-во id', len(unique_id))  # Выведем количество id
items = list(set(all_data[2]))
print('Кол-во товаров', len(items))  # Выведем количество товаров
dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in items] for id in unique_id]

# Transformation
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

# Analyse Task 1
results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))  # добавление размера набора
print(results)

# Analyse Task 2
results = apriori(df, min_support=0.3, use_colnames=True, max_len=1)
results['length'] = results['itemsets'].apply(lambda x: len(x))
print(results)

# Analyse Task 3
results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
results = results[results['length'] == 2]
print(results)
print('\nCount of result itemstes = ', len(results))

# Analyse Task 4, 5
items = np.array([])
rng = np.arange(0.05, 1, 0.01)
levels = np.array([])
results = apriori(df, min_support=0.05)
results['length'] = results['itemsets'].apply(lambda x: len(x))
max_len = results.max()['length']

for i in rng:
    results = apriori(df, min_support=i)
    results['length'] = results['itemsets'].apply(lambda x: len(x))
    if results.max()['length'] < max_len:
        max_len = results.max()['length']
        levels = np.append(i, levels)
    items = np.append(items, len(results))

fig, ax = plt.subplots()
ax.plot(rng, items)
for level in range(len(levels)):
    ax.axvline(x=levels[level], color='red', label='k = ' + str(level + 2) + ' -> ' + str(level + 1))
    ax.annotate(text='k = ' + str(level + 2) + ' -> ' + str(level + 1) + '\nthreshold = ' + str('{:.2f}').format(levels[level]),
                xy=(levels[level] + 0.01, 1000 / levels[level]))
plt.show()
plt.close(fig)

# Analyse Task 6, 7
results = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
new_items = [list(elem)[0] for elem in results['itemsets']]
new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in new_items] for id in unique_id]

te_ary = te.fit(new_dataset).transform(new_dataset)
df_new = pd.DataFrame(te_ary, columns=te.columns_)

# Analyse Task 8
results = apriori(df_new, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
print(results)

# Analyse Task 9
results = apriori(df_new, min_support=0.15, use_colnames=True)
results = results[results['itemsets'].apply(lambda x:
                                            x.intersection(frozenset(['yogurt', 'waffles'])) != frozenset([]) and
                                            len(x) > 1)]
# print(results)

# Analyse Task 10
new_dataset2 = [[elem for elem in all_data[all_data[1] == id][2] if elem not in new_items] for id in unique_id]
te_ary = te.fit(new_dataset2).transform(new_dataset2)
df_new2 = pd.DataFrame(te_ary, columns=te.columns_)

# Analyse Task 11
results = apriori(df_new2, min_support=0.03, use_colnames=True)

# Analyse Task 12
results = results[results['itemsets'].apply(lambda x:
                                            len([word for word in x if word.startswith('s')]) > 1)]
# print(results)

# Analyse Task 13
results = apriori(df_new2, min_support=0.03, use_colnames=True)
results = results[results['support'].apply(lambda x: 0.1 <= x <= 0.25)]
print(results)
