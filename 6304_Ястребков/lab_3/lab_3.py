import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

# To make plots on Linux-based systems
matplotlib.use('TkAgg')

# Data preparation
all_data = pd.read_csv('../../../dataset_group.csv', header=None)
print(all_data.head(7))

unique_id = list(set(all_data[1]))
print(all_data.shape, len(unique_id))
items = list(set(all_data[2]))
print(len(items))

dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in items] for id in unique_id]
te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_array, columns=te.columns_)

print(df.head())

# Apriori
results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(lambda x: len(x))
print(results)

results = results[results['length'] == 2]
print(results)
print('\nCount of result itemstes = ', len(results))

# Number of sets
support = np.arange(0.05, 0.8, 0.01)
set_number = []
max_set_size = 4
colors = ['r', 'm', 'y', 'g']

for s in support:
    rslt = apriori(df, min_support=s, use_colnames=True)
    set_number.append(rslt.shape[0])
    rslt['length'] = rslt['itemsets'].apply(lambda x: len(x))
    if rslt.shape[0] == 0 and max_set_size != 0:
        plt.scatter(s, rslt.shape[0], s=30, c='r')
        max_set_size = 0
        print(f"No sets with size 1; support={s}")
    if rslt.shape[0] != 0 and max_set_size != max(rslt['length']):
        plt.scatter(s, rslt.shape[0], s=30, c=colors[max_set_size - 1])
        print(f"No sets with size {max_set_size}; support={s}")
        max_set_size = max_set_size - 1

plt.plot(support, set_number)
plt.xlabel('Support level')
plt.ylabel('Number of sets')

# Support = 0.38
results = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
new_items = [list(elem)[0] for elem in results['itemsets']]
new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in new_items] for id in unique_id]

te_2 = TransactionEncoder()
te_array = te_2.fit(new_dataset).transform(new_dataset)
new_df = pd.DataFrame(te_array, columns=te_2.columns_)

print(new_df.head())

new_results = apriori(new_df, min_support=0.3, use_colnames=True)
new_results['length'] = new_results['itemsets'].apply(lambda x: len(x))
print(new_results)

# Filter yogurt and waffles
low_support = apriori(new_df, min_support=0.15, use_colnames=True)
low_support['length'] = low_support['itemsets'].apply(lambda x: len(x))

only_yogurt_waffles = lambda x: x['length'] > 1 and ('yogurt' in x['itemsets'] or 'waffles' in x['itemsets'])

print(low_support[low_support.apply(only_yogurt_waffles, axis=1)])

# Other items
other_items = [item for item in items if item not in new_items]
other_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in other_items] for id in unique_id]

te_3 = TransactionEncoder()
te_array = te_3.fit(other_dataset).transform(other_dataset)
other_df = pd.DataFrame(te_array, columns=te_3.columns_)

print(other_df.head())

other_results = apriori(other_df, min_support=0.3, use_colnames=True)
other_results['length'] = other_results['itemsets'].apply(lambda x: len(x))

print(other_results)


# Rules
# Filter 's'
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


# 0.1 < support < 0.25
results_for_rules = apriori(df, min_support=0.07, use_colnames=True)
results_for_rules['length'] = results_for_rules['itemsets'].apply(lambda x: len(x))

print(results_for_rules[lambda df: (df['support'] > 0.1) & (df['support'] < 0.25)])
