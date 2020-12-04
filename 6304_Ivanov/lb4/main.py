import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, fpmax, association_rules
import random as rand
from numpy import random as npr
import seaborn as sns


all_data = pd.read_csv('groceries - groceries.csv')
print(all_data.shape)

np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem,str)] for row in np_data]

unique_items = set(np.unique(np.concatenate(np_data)))
print(unique_items)
print(len(unique_items))


## FPGrowth и FPMax

# 1
te = TransactionEncoder()
te_ary = te.fit(np_data).transform(np_data)
data = pd.DataFrame(te_ary, columns=te.columns_)
print(data)

# 2
result_fpgrowth = fpgrowth(data, min_support=0.03, use_colnames = True)
result_fpgrowth['length'] = np.fromiter(map(len, result_fpgrowth['itemsets']),dtype=int)
print(result_fpgrowth.sort_values('support',ascending=False))

# 3
print(result_fpgrowth.groupby('length').support.min())
print(result_fpgrowth.groupby('length').support.max())

# 4
result_fpmax = fpmax(data, min_support=0.03, use_colnames = True)
result_fpmax['length'] = np.fromiter(map(len, result_fpmax['itemsets']),dtype=int)
print(result_fpmax.groupby('length').support.min())
print(result_fpmax.groupby('length').support.max())
print(result_fpmax.sort_values('support',ascending=False))

# 6
plt.figure(figsize=(8,6))
count_of_items = data.sum()
count_of_items.nlargest(10).plot.bar()

plt.figure(figsize=(8,6))
data_ = result_fpgrowth[result_fpgrowth.length==1].sort_values('support',ascending=False).set_index('itemsets').support
data_.nlargest(10).plot.bar()

# 7
items = ['whole milk', 'yogurt', 'soda', 'tropical fruit', 'shopping bags', 'sausage', 'whipped/sour cream', 'rolls/buns', 'other vegetables', 'root vegetables', 'pork', 'bottled water', 'pastry', 'citrus fruit', 'canned beer', 'bottled beer']
np_data_f = all_data.to_numpy()
np_data_f = [[elem for elem in row[1:] if isinstance(elem,str) and elem in items] for row in np_data_f]

te_f = TransactionEncoder()
te_ary_f = te_f.fit(np_data_f).transform(np_data_f)
data_f = pd.DataFrame(te_ary_f, columns=te_f.columns_)

# 8.1
result_fpgrowth_f = fpgrowth(data_f, min_support=0.03, use_colnames = True)
result_fpgrowth_f['length'] = np.fromiter(map(len, result_fpgrowth_f['itemsets']),dtype=int)
print(result_fpgrowth_f.groupby('length').support.min())
print(result_fpgrowth_f.groupby('length').support.max())
print(result_fpgrowth_f.sort_values('support', ascending=False))

# 8.2
result_fpmax_f = fpmax(data_f, min_support=0.03, use_colnames = True)
result_fpmax_f['length'] = np.fromiter(map(len, result_fpmax_f['itemsets']),dtype=int)
print(result_fpmax_f.groupby('length').support.min())
print(result_fpmax_f.groupby('length').support.max())

# 9
min_sups = np.arange(0.03, 0.26, 0.01)
counts_of_itemesets = []
results = []

for min_sup in min_sups:
    fpgrowth_res = fpgrowth(data, min_support=min_sup, use_colnames=True)
    fpmax_res = fpmax(data, min_support=min_sup, use_colnames=True)
    results.append((fpgrowth_res, fpmax_res))

sizes = []
max_len = []
for min_sup_vals in results:
    sizes.append((min_sup_vals[0].shape[0], min_sup_vals[1].shape[0]))
    max_len.append(len(min_sup_vals[0].iloc[-1].itemsets))
sizes = np.array(sizes)

minimal_min_sup_for_itemsets_count = []
unique_max_len_of_items = np.unique(max_len)
for count in unique_max_len_of_items:
    minimal_min_sup_for_itemsets_count.append(np.where(max_len == count)[0][0])

end_of_generating_n_length_itemsets = list(reversed(minimal_min_sup_for_itemsets_count))[1:]

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(min_sups, sizes[:, 0], linewidth=2, label='FPGrowth')
ax.plot(min_sups, sizes[:, 1], linewidth=2, label='FPMax')
for i in end_of_generating_n_length_itemsets:
    print(min_sups[i])
    plt.axvline(min_sups[i], color='red')
ax.set_xlabel('Уровень поддержки')
ax.set_ylabel('Количество наборов')
plt.legend()
plt.show()

## Ассоциативные правила

# 1
np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem,str) and elem in
items] for row in np_data]
np_data = [row for row in np_data if len(row) > 1]

# 2
result = fpgrowth(data, min_support=0.05, use_colnames = True)
print(result.sort_values('support'))

# 3
rules = association_rules(result, min_threshold = 0.3)

# 4 - confidence
result_fpgrowth = fpgrowth(data, min_support=0.03, use_colnames = True)
result_fpgrowth['length'] = np.fromiter(map(len, result_fpgrowth['itemsets']),dtype=int)
association_rules_res = association_rules(result_fpgrowth, metric='confidence', min_threshold = 0.34)
print(association_rules_res)
print(association_rules(result_fpgrowth, metric='lift', min_threshold = 1.75))
print(association_rules(result_fpgrowth, metric='leverage', min_threshold = 0.016))
print(association_rules(result_fpgrowth, metric='conviction', min_threshold = 1.18))

# 6
print(association_rules_res.iloc[:,2:].describe())
result = fpgrowth(data, min_support=0.03, use_colnames = True)
result[np.fromiter(map(len, result_fpgrowth['itemsets']),dtype=int) == 2]
rules = association_rules(result, min_threshold = 0.4, metric='confidence')
print(rules)

digraph = nx.DiGraph()
for i in range(rules.shape[0]):
    digraph.add_edge(
        rules.iloc[i].antecedents,
        rules.iloc[i].consequents,
        weight=rules.iloc[i].support,
        label=round(rules.iloc[i].confidence,3)
    )

rand.seed(10)
npr.seed(10)
plt.figure(figsize=(15, 8))
pos = nx.spring_layout(digraph)
nx.draw(
    digraph,
    pos,
    labels={node: ','.join(node) for node in digraph.nodes()},
    width=[digraph[u][v]['weight']*100 for u,v in digraph.edges()],
    node_size=2000,
    node_color='#00b4ff',
    font_size=20
)
nx.draw_networkx_edge_labels(
    digraph,
    pos,
    edge_labels=nx.get_edge_attributes(digraph, 'label'),
    font_size=20
)
plt.show()


rules_pivot = rules.pivot(index='antecedents', columns='consequents', values='confidence')
rules_pivot.index = list(map(lambda x: list(x)[0], rules_pivot.index))
rules_pivot.columns = list(map(lambda x: list(x)[0], rules_pivot.columns))
print(rules_pivot)
sns.heatmap(rules_pivot, cmap='turbo')