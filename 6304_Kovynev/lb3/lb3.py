import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

all_data = pd.read_csv('dataset_group.csv', header=None)
# В файле нет строки с названием столбцов, поэтому параметр header равен None.
# Интерес представляет информация об id покупателя - столбец с названием 1
# Название купленного товара хранится в столбце с названием 2

unique_id = list(set(all_data[1]))
# print('unique_id', len(unique_id)) #Выведем количество id

items = list(set(all_data[2]))
# print('items', len(items)) #Выведем количество товаров


dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in items] for id in unique_id]

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


min_sups = np.arange(0.05, 0.7, 0.01)
counts_of_itemesets = []
apriori_results = []

for min_sup in min_sups:
    apriori_results.append(apriori(df, min_support=min_sup, use_colnames=True))
    counts_of_itemesets.append(apriori_results[-1].shape[0])
counts_of_itemesets = np.array(counts_of_itemesets)

max_len_of_itemsets = np.fromiter(map(lambda x: x.itemsets.map(len).max(), apriori_results), dtype=int)
print(max_len_of_itemsets)

df_ = pd.DataFrame({'min_sup': min_sups, 'count_of_itemeset': counts_of_itemesets})

minimal_min_sup_for_itemsets_count = []
unique_max_len_of_items = np.unique(max_len_of_itemsets)
for count in unique_max_len_of_items:
    minimal_min_sup_for_itemsets_count.append(
        np.where(max_len_of_itemsets == count)[0][0]
    )

print(minimal_min_sup_for_itemsets_count)
end_of_generating_n_length_itemsets = list(reversed(minimal_min_sup_for_itemsets_count))[1:]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df_.min_sup, df_.count_of_itemeset, linewidth=2, color='red')
for i in end_of_generating_n_length_itemsets:
    plt.axvline(df_.min_sup[i] - 0.005, color='green')
    print(df_.min_sup[i] - 0.005)
ax.set_xlabel('support level')
ax.set_ylabel('Count itemset')
# plt.show()


results = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
new_items = [list(elem)[0] for elem in results['itemsets']]
new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in new_items] for id in unique_id]

te = TransactionEncoder()
te_ary = te.fit_transform(new_dataset)
df_new = pd.DataFrame(te_ary, columns=te.columns_)

results_new = apriori(df_new, min_support=0.3, use_colnames=True)
results_new['length'] = results_new['itemsets'].apply(lambda x: len(x))
print(results_new)

results = apriori(df_new, min_support=0.15, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
results = results[
    results.apply(lambda d: d['length'] > 1 and ('yougurt' in d['itemsets'] or 'waffles' in d['itemsets']),
                  axis=1)].reset_index(drop=True)
print(results)

diff = set(list(df)) - set(list(df_new))
diff_items = [list(elem)[0] for elem in results['itemsets']]
diff_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem not in diff_items] for id in unique_id]
te = TransactionEncoder()
te_ary = te.fit_transform(diff_dataset)
df_new = pd.DataFrame(te_ary, columns=te.columns_)
print(apriori(df, min_support=0.3, use_colnames=True))

two_elems_starts_with_s = lambda df: df[df['itemsets'].apply(
    lambda x:
    np.fromiter(map(lambda y: y.startswith('s'), x), dtype=bool).sum() >= 2
)
]
two_elems_starts_with_s = lambda df: df[df['itemsets'].apply(
    lambda x:
    np.fromiter(map(lambda y: y.startswith('s'), x), dtype=bool).sum() >= 2
)
]
print(two_elems_starts_with_s(apriori_results[0]))

subset_10_25 = lambda df: df[np.logical_and(df.support >= 0.1, df.support <= 0.25)]
print(subset_10_25(apriori_results[0]))
