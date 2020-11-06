import pandas as pd
import numpy as np

all_data = pd.read_csv('../data/dataset_group.csv', header=None)

unique_id = list(set(all_data[1]))
print(len(unique_id))

items = list(set(all_data[2]))
print(len(items))

dataset = [
    [
        elem for elem in all_data[all_data[1] == id][2]
        if elem in items
    ]
    for id in unique_id
]
print(dataset[:2])

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit_transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
pd.set_option('display.max_columns', 7)

with open('./output/df.txt', 'w') as f:
    f.write(str(df))

df

from mlxtend.frequent_patterns import apriori

results = apriori(df, min_support=0.3, use_colnames=True)
results['length'] = results['itemsets'].apply(len)
old_results = results

with open('./output/results.txt', 'w') as f:
    f.write(str(results))
    
with open('./output/results_gt2.txt', 'w') as f:
    f.write(str(results[results['length'] > 1]))
    
results

support_data = []

for min_support in np.arange(0.05, 1, 0.01):
    r = apriori(df, min_support=min_support)
    r['length'] = r['itemsets'].apply(len)
    max_len = np.max(r['length'])
    if np.isnan(max_len):
        break
    datum = {
        'min_support': min_support,
        'total': len(r),
        **{n: 0 for n in range(1, max_len + 1)}
    }
    for res in r.itertuples(index=False):
        datum[res.length] += 1
    
    if min_support % 0.1 == 1:
        print('Analyzing', min_support)
        
    support_data.append(datum)

support_df = pd.DataFrame(support_data).fillna(value=0)
support_df

min_indices = support_df[[1, 2, 3, 4]].idxmin()
min_support = pd.Series(list(support_df['min_support'][min_indices]), index=min_indices.index)
min_support

from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['figure.facecolor'] = '1'

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
cmap = mpl.cm.get_cmap('Set1')

support_df.plot(ax=ax, logy='sym', x='min_support', colormap=cmap)
for i, value in min_support.iteritems():
    ax.axvline(value, label=f'{i} (min)', lw=1, linestyle='--', color=cmap(i / len(min_support)))
ax.legend()
pass

results = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
new_items = [list(elem)[0] for elem in results['itemsets']]
new_dataset = [
    [
        elem for elem in all_data[all_data[1] == id][2]
        if elem in new_items
    ]
    for id in unique_id
]
print(new_dataset[:2])

te = TransactionEncoder()
te_ary = te.fit_transform(new_dataset)

new_df = pd.DataFrame(te_ary, columns=te.columns_)

with open('./output/new_df.txt', 'w') as f:
    f.write(str(new_df))

new_df

new_results = apriori(new_df, min_support=0.3, use_colnames=True)
new_results['length'] = new_results['itemsets'].apply(len)

with open('./output/new_results.txt', 'w') as f:
    f.write(str(new_results))

new_results

new2_results = apriori(new_df, min_support=0.15, use_colnames=True)
new2_results['length'] = new2_results['itemsets'].apply(len)
new2_results = new2_results[
    new2_results.apply(
        lambda d: d['length'] > 1 and ('yougurt' in d['itemsets'] or 'waffles' in d['itemsets']), axis=1
    )
].reset_index(drop=True)

with open('output/new2_results.txt', 'w') as f:
    f.write(str(new2_results))

new2_results

new3_items = [elem for elem in items if elem not in new_items]
new3_dataset = [
    [
        elem for elem in all_data[all_data[1] == id][2]
        if elem in new3_items
    ]
    for id in unique_id
]

te = TransactionEncoder()
te_ary = te.fit_transform(new3_dataset)

new3_df = pd.DataFrame(te_ary, columns=te.columns_)

with open('./output/new3_df.txt', 'w') as f:
    f.write(str(new3_df))

new3_df

new3_results = apriori(new3_df, min_support=0.1, use_colnames=True)
new3_results

s_results = new3_results[new3_results.apply(
    lambda r: len([e for e in r['itemsets'] if e.startswith('s')]) > 1,
    axis=1
)]

with open('output/s_results.txt', 'w') as f:
    f.write(str(s_results))
    
s_results

# new3_results[new3_results['support'] > 0.1 | new3_results['support'] < 0.25]
ss_results = new3_results[(new3_results['support'] > 0.1) & (new3_results['support'] < 0.25)]

with open('output/ss_results.txt', 'w') as f:
    f.write(str(ss_results))
    
ss_results
