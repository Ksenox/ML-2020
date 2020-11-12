import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import matplotlib.pyplot as plt


saveFigure = False
plot_base_path = 'plots/lab3/'


all_data = pd.read_csv('datasets/dataset_group.csv',header=None)
# all_data['date'] = pd.to_datetime(all_data[0])


all_data.dtypes


unique_id = all_data[1].unique()
print(unique_id.size)


items = all_data[2].unique()
print(items.size)


dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in items] for id in unique_id]


te = TransactionEncoder()
te_ary = te.fit_transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)


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


min_sups = np.arange(0.05, 0.74, 0.01)
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
minimal_min_sup_for_itemsets_count


end_of_generating_n_length_itemsets = list(reversed(minimal_min_sup_for_itemsets_count))[1:]


# 4+5
fig, ax = plt.subplots(figsize=(8,6))
fig.suptitle('Зависимость количества наборов от уровня поддержки', fontsize=16)
ax.plot(df_.min_sup, df_.count_of_itemeset, linewidth=2)
for i in end_of_generating_n_length_itemsets:
    plt.axvline(df_.min_sup[i] - 0.005, color='red')
ax.set_yscale('log')
ax.set_xlabel('Уровень поддержки')
ax.set_ylabel('Количество наборов')
if saveFigure:
    plt.savefig(plot_base_path+'count_of_sets_supp.png')
plt.show()


# 6
results = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
new_items = [ list(elem)[0] for elem in results['itemsets']]
new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in new_items] for id in unique_id]


# 7
te = TransactionEncoder()
te_ary = te.fit_transform(new_dataset)
df_new = pd.DataFrame(te_ary, columns=te.columns_)


# 8
results_new = apriori(df_new, min_support=0.3, use_colnames=True)
results_new['length'] = results_new['itemsets'].apply(lambda x: len(x))
print(results_new)


# 9
results = apriori(df_new, min_support=0.15, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
print(results)


# 10
diff = set(list(df)) - set(list(df_new))
diff_items = [ list(elem)[0] for elem in results['itemsets']]
diff_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem not in diff_items] for id in unique_id]
te = TransactionEncoder()
te_ary = te.fit_transform(diff_dataset)
df_new = pd.DataFrame(te_ary, columns=te.columns_)


# 11
print(apriori(df, min_support=0.3, use_colnames=True))


# 12
def two_elems_starts_with_s(df, threshold=2):
    return df[
        df['itemsets'].apply(
            lambda x: np.fromiter(
                map(lambda y: y[0]=='s', x), dtype=bool
            ).sum()>=threshold
        )
    ]


print(two_elems_starts_with_s(apriori_results[0]))


# 13
def subset_10_25(df):
    return df[np.logical_and(df.support>=0.1, df.support <= 0.25)]


print(subset_10_25(apriori_results[0]))


