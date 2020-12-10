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
print(all_data)  # Видно, что датафрейм содержит NaN значения

np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem, str)] for row in np_data]

unique_items = set()
for row in np_data:
    for elem in row:
        unique_items.add(elem)

print(len(unique_items), unique_items)

te = TransactionEncoder()
te_ary = te.fit(np_data).transform(np_data)
data = pd.DataFrame(te_ary, columns=te.columns_)

result_fpgrowth = fpgrowth(data, min_support=0.03, use_colnames=True)
result_fpgrowth['length'] = np.fromiter(map(len, result_fpgrowth['itemsets']), dtype=int)
print(result_fpgrowth.sort_values('support', ascending=False))

print(result_fpgrowth.groupby('length').support.min())
print(result_fpgrowth.groupby('length').support.max())

result_fpmax = fpmax(data, min_support=0.03, use_colnames=True)
result_fpmax['length'] = np.fromiter(map(len, result_fpmax['itemsets']), dtype=int)
print(result_fpmax.sort_values('support', ascending=False))

print(result_fpmax.groupby('length').support.min())
print(result_fpmax.groupby('length').support.max())

plt.figure(figsize=(6, 6))

count_of_items = data.sum()
count_of_items.nlargest(10).plot.barh(align='center')
# plt.show()

plt.figure(figsize=(6, 6))
data_ = result_fpgrowth[result_fpgrowth.length == 1].sort_values('support', ascending=False).set_index(
    'itemsets').support
data_.nlargest(10).plot.barh()
# plt.show()

items = ['whole milk', 'yogurt', 'soda', 'tropical fruit', 'shopping bags', 'sausage', 'whipped/sour cream', 'rolls/buns', 'other vegetables', 'root vegetables', 'pork', 'bottled water', 'pastry', 'citrus fruit', 'canned beer', 'bottled beer']
np_data_f = all_data.to_numpy()
np_data_f = [[elem for elem in row[1:] if isinstance(elem,str) and elem in items] for row in np_data_f]


te_f = TransactionEncoder()
te_ary_f = te_f.fit(np_data_f).transform(np_data_f)
data_f = pd.DataFrame(te_ary_f, columns=te_f.columns_)

result_fpgrowth_f = fpgrowth(data_f, min_support=0.03, use_colnames = True)
result_fpgrowth_f['length'] = np.fromiter(map(len, result_fpgrowth_f['itemsets']),dtype=int)
print(result_fpgrowth_f.groupby('length').support.min())
print(result_fpgrowth_f.groupby('length').support.max())
print(result_fpgrowth_f.sort_values('support', ascending=False))

result_fpmax_f = fpmax(data_f, min_support=0.03, use_colnames = True)
result_fpmax_f['length'] = np.fromiter(map(len, result_fpmax_f['itemsets']),dtype=int)
print(result_fpmax_f.groupby('length').support.min())
print(result_fpmax_f.groupby('length').support.max())


sup_data = []

for min_support in np.linspace(0.001, 0.02, num=10):
    results = fpgrowth(data, min_support=min_support, use_colnames=True)
    results['length'] = results['itemsets'].apply(lambda x: len(x))
    max_len_curr = np.max(results['length'])
    if (np.isnan(max_len_curr)):
        break
    grouped_count = results.groupby('length').itemsets.count()

    lens_dict = {
        'Общее кол-во': 0,
        'min_support': min_support
    }
    for i in range(1, len(grouped_count) + 1):
        lens_dict.setdefault(f'Набор длины {i}', grouped_count[i])
    lens_dict['Общее кол-во'] = len(results)
    sup_data.append(lens_dict)

df_count_by_lens = pd.DataFrame(sup_data).fillna(value=0)
fig, ax = plt.subplots(figsize=(12, 8))
df_count_by_lens.plot(ax=ax, x='min_support')
ax.set_axisbelow(True)
ax.set_ylabel('Количество наборов')
ax.set_xlabel('Уровень поддержки')
fig.tight_layout()

plt.show()


np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem,str) and elem in items] for row in np_data]
np_data = [row for row in np_data if len(row) > 1]

result = fpgrowth(data, min_support=0.05, use_colnames = True)
rules = association_rules(result, min_threshold = 0.3)
print(rules)

result_fpgrowth = fpgrowth(data, min_support=0.03, use_colnames = True)
result_fpgrowth['length'] = np.fromiter(map(len, result_fpgrowth['itemsets']),dtype=int)
association_rules_res = association_rules(result_fpgrowth, metric='confidence', min_threshold = 0.34)
print(association_rules_res)
print(association_rules(result_fpgrowth, metric='lift', min_threshold = 1.75))
print(association_rules(result_fpgrowth, metric='leverage', min_threshold = 0.016))
print(association_rules(result_fpgrowth, metric='conviction', min_threshold = 1.18))


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
    node_color='#ff00ff',
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
rules_pivot.index = ['\n'.join(ind) for ind in rules_pivot.index]
rules_pivot.columns = ['\n'.join(col) for col in rules_pivot.columns]
sns.heatmap(rules_pivot, cmap='rainbow')
plt.tight_layout()
plt.show()

print(rules)