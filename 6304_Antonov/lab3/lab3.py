import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import matplotlib.pyplot as plt


all_data = pd.read_csv('dataset_group.csv',header=None)
print(all_data)

unique_id = all_data[1].unique()
print(unique_id.size)


items = all_data[2].unique()
print(items.size)


dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in items] for id in unique_id]


te = TransactionEncoder()
te_ary = te.fit_transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

# 1
results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
print(results)


results_orig = apriori(df, min_support=0.3, use_colnames=True, max_len=1)
results_orig['length'] = results_orig['itemsets'].apply(lambda x: len(x))
print(results_orig)


results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
results_2 = results[results['length'] == 2]
print(results_2)
print('\nCount of result itemstes = ',len(results_2))


min_sups = np.arange(0.05, 0.7, 0.01)
counts_of_itemesets = []
apriori_results = []


for min_sup in min_sups:
    apriori_results.append(apriori(df, min_support=min_sup, use_colnames=True))
    counts_of_itemesets.append(apriori_results[-1].shape[0])
counts_of_itemesets = np.array(counts_of_itemesets)


max_len_of_itemsets = np.fromiter(map(lambda x: x.itemsets.map(len).max(), apriori_results), dtype=int)


df_ = pd.DataFrame({'min_sup': min_sups, 'count_of_itemeset': counts_of_itemesets})


minimal_min_sup_for_itemsets_count = []
unique_max_len_of_items = np.unique(max_len_of_itemsets)
for count in unique_max_len_of_items:
    minimal_min_sup_for_itemsets_count.append(
        np.where(max_len_of_itemsets == count)[0][0]
    )


# unique_max_len_of_items
print(minimal_min_sup_for_itemsets_count)


end_of_generating_n_length_itemsets = list(reversed(minimal_min_sup_for_itemsets_count))[1:]


# 4-5
fig, ax = plt.subplots(figsize=(8,6))
fig.suptitle('Зависимость количества наборов от уровня поддержки', fontsize=16)
ax.plot(df_.min_sup, df_.count_of_itemeset, linewidth=2)
for i in end_of_generating_n_length_itemsets:
    plt.axvline(df_.min_sup[i] - 0.005, color='red')
ax.set_yscale('log')
ax.set_xlabel('Уровень поддержки')
ax.set_ylabel('Количество наборов')
plt.show()


# 6
results = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
new_items = [list(elem)[0] for elem in results['itemsets']]
new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in new_items] for id in unique_id]


# 7
te = TransactionEncoder()
te_ary = te.fit_transform(new_dataset)
df_new = pd.DataFrame(te_ary, columns=te.columns_)


# 8
results_new = apriori(df_new, min_support=0.3, use_colnames=True)
results_new['length'] = results_new['itemsets'].apply(lambda x: len(x))
print('8', results_new)


# 9
low_support = apriori(df_new, min_support=0.15, use_colnames=True)
low_support['length'] = low_support['itemsets'].apply(lambda x: len(x))

only_yogurt_waffles = lambda x: x['length'] > 1 and ('yogurt' in x['itemsets'] or 'waffles' in x['itemsets'])

print(low_support[low_support.apply(only_yogurt_waffles, axis=1)])


# 10
diff = set(list(df)) - set(list(df_new))
diff_items = [ list(elem)[0] for elem in results['itemsets']]
diff_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem not in diff_items] for id in unique_id]
te = TransactionEncoder()
te_ary = te.fit_transform(diff_dataset)
df_new = pd.DataFrame(te_ary, columns=te.columns_)
print('10')
print(df_new)

# 11
print(apriori(df, min_support=0.3, use_colnames=True))


# 12
def two_s_elements(row):
    s_elements = 0
    for item in row['itemsets']:
        if str(item).startswith('s'):
            s_elements = s_elements + 1
        if s_elements == 2:
            return True
    return False


results_for_rules = apriori(df, min_support=0.15, use_colnames=True)
results_for_rules['length'] = results_for_rules['itemsets'].apply(lambda x: len(x))
print(results_for_rules[results_for_rules.apply(two_s_elements, axis=1)])


# 13
subset_10_25 = lambda df: df[np.logical_and(df.support>=0.1, df.support <= 0.25)]
print(subset_10_25(apriori_results[0]))
