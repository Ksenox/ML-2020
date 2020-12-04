import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, fpmax, association_rules
import matplotlib.pyplot as plt
import networkx as nx


def paintPlot(_data, _support_values, _axie, _function):
    if len(support_values) < 1:
        return
    _res = _function(_data, min_support=_support_values[0], use_colnames=True)
    _level_items = _res['itemsets'].apply(lambda x: len(x)).value_counts().sort_index()
    _ndim = max(_level_items.index)
    _matrix = np.zeros((1, _ndim))
    np.put(_matrix, _level_items.index - 1, _level_items)
    for index in range(1, len(_support_values)):
        _res = _function(_data, min_support=_support_values[index], use_colnames=True)
        _level_items = _res['itemsets'].apply(lambda x: len(x)).value_counts().sort_index()
        _new_line = np.zeros((1, _ndim))
        np.put(_new_line, _level_items.index - 1, _level_items.to_numpy())
        _matrix = np.append(_matrix, _new_line, axis=0)
    for size in range(_matrix.shape[1]):
        _axie.plot(_support_values, _matrix[:, size])
    _axie.legend(np.arange(1, _ndim + 1))


all_data = pd.read_csv('groceries - groceries.csv')
print(all_data)  # Видно, что датафрейм содержит NaN значения

np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem, str)] for row in np_data]

unique_items = set([item for row in np_data for item in row])

print('Items: ', unique_items)
print('Items amount: ', len(unique_items))

te = TransactionEncoder()
te_ary = te.fit(np_data).transform(np_data)
data = pd.DataFrame(te_ary, columns=te.columns_)

result = fpgrowth(data, min_support=0.03, use_colnames=True)
print(result)

minmax = {'Max': {size: max(result[result['itemsets'].apply(lambda x: len(x) == size)]['support'])
                  for size in range(1, max(result['itemsets'].apply(lambda x: len(x))) + 1)},
          'Min': {size: min(result[result['itemsets'].apply(lambda x: len(x) == size)]['support'])
                  for size in range(1, max(result['itemsets'].apply(lambda x: len(x))) + 1)}}

print(minmax)

hist_result = result.sort_values(by='support', ascending=False).head(10)

result = fpmax(data, min_support=0.03, use_colnames=True)
print(result)

minmax = {'Max': {sz: max(result[result['itemsets'].apply(lambda x: len(x) == sz)]['support'])
                  for sz in range(1, max(result['itemsets'].apply(lambda x: len(x))) + 1)},
          'Min': {sz: min(result[result['itemsets'].apply(lambda x: len(x) == sz)]['support'])
                  for sz in range(1, max(result['itemsets'].apply(lambda x: len(x))) + 1)}}

print(minmax)

fig, ax = plt.subplots()
n, bins, patches = ax.hist([[val] for val in np.arange(hist_result['support'].size)],
                           weights=[[val] for val in hist_result['support']],
                           label=hist_result['itemsets'], histtype='stepfilled', bins=hist_result['support'].size)
ax.set_xticks([(bins[i + 1] + bins[i]) / 2 for i in range(bins.size - 1)])
ax.set_xticklabels(hist_result['itemsets'].apply(lambda x: ', '.join([val for val in x])),
                   rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()
plt.close(fig)

items = ['whole milk', 'yogurt', 'soda', 'tropical fruit', 'shopping bags', 'sausage', 'whipped/sour cream',
         'rolls/buns', 'other vegetables', 'root vegetables', 'pork', 'bottled water', 'pastry', 'citrus fruit',
         'canned beer', 'bottled beer']
np_data_chosen = all_data.to_numpy()
np_data_chosen = [[elem for elem in row[1:] if isinstance(elem, str) and elem in items] for row in np_data_chosen]

te_ary_chosen = te.fit(np_data_chosen).transform(np_data_chosen)
data_chosen = pd.DataFrame(te_ary_chosen, columns=te.columns_)

result = fpgrowth(data_chosen, min_support=0.03, use_colnames=True)
print(result)

minmax = {'Max': {size: max(result[result['itemsets'].apply(lambda x: len(x) == size)]['support'])
                  for size in range(1, max(result['itemsets'].apply(lambda x: len(x))) + 1)},
          'Min': {size: min(result[result['itemsets'].apply(lambda x: len(x) == size)]['support'])
                  for size in range(1, max(result['itemsets'].apply(lambda x: len(x))) + 1)}}

print(minmax)

result = fpmax(data_chosen, min_support=0.03, use_colnames=True)
print(result)

minmax = {'Max': {sz: max(result[result['itemsets'].apply(lambda x: len(x) == sz)]['support'])
                  for sz in range(1, max(result['itemsets'].apply(lambda x: len(x))) + 1)},
          'Min': {sz: min(result[result['itemsets'].apply(lambda x: len(x) == sz)]['support'])
                  for sz in range(1, max(result['itemsets'].apply(lambda x: len(x))) + 1)}}

print(minmax)

support_values = np.arange(0.005, 0.4, 0.005)

fig_full_growth, ax_full_growth = plt.subplots()
paintPlot(data, support_values, ax_full_growth, fpgrowth)
plt.grid()
plt.title('FPGrowth full data')
plt.show()
plt.close(fig_full_growth)

fig_chosen_growth, ax_chosen_growth = plt.subplots()
paintPlot(data_chosen, support_values, ax_chosen_growth, fpgrowth)
plt.grid()
plt.title('FPGrowth chosen data')
plt.show()
plt.close(fig_chosen_growth)

fig_full_max, ax_full_max = plt.subplots()
paintPlot(data, support_values, ax_full_max, fpmax)
plt.grid()
plt.title('FPMax full data')
plt.show()
plt.close(fig_full_max)

fig_chosen_max, ax_chosen_max = plt.subplots()
paintPlot(data_chosen, support_values, ax_chosen_max, fpmax)
plt.grid()
plt.title('FPMax chosen data')
plt.show()
plt.close(fig_chosen_max)

# Association rules

np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem, str) and elem in items] for row in np_data]
np_data = [row for row in np_data if len(row) > 1]

result = fpgrowth(data, min_support=0.04, use_colnames=True)

rules = association_rules(result, min_threshold=0.4)
print(rules)

metrics = ['confidence', 'lift', 'leverage', 'conviction']

rules = association_rules(result, min_threshold=0.1)
print(rules)
for metr in metrics:
    print(metr, ' - \tMean: ', np.mean(rules[metr].to_numpy()),
          ' \tMedian: ', np.median(rules[metr].to_numpy()), ' \tMSD: ', np.sqrt(np.var(rules[metr].to_numpy())))

# Graph
rules = association_rules(result, min_threshold=0.4, metric='confidence')
nodes = set(rules['antecedents'].apply(lambda x: ', '.join([elem for elem in x])).tolist() +
            rules['consequents'].apply(lambda x: ', '.join([elem for elem in x])).tolist())
edges = rules[['antecedents', 'consequents']] \
    .apply(lambda x: pd.Series([', '.join([elem for elem in fset]) for fset in x]))
edges['confidence'] = rules['confidence']
edges = edges.values.tolist()

fig, axie = plt.subplots()
graph = nx.DiGraph()
graph.add_nodes_from(nodes)
graph.add_weighted_edges_from(edges)
pos = nx.spring_layout(graph)
nx.draw(graph, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='red', alpha=0.9,
        labels={node: node for node in graph.nodes()})
nx.draw_networkx_edge_labels(graph, pos, edge_labels={(edge[0], edge[1]): "{:.2f}".format(edge[2]) for edge in edges},
                             font_color='black')
plt.axis('off')
plt.show()
plt.close(fig)
