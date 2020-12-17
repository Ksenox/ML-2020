# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Цель работы
# Ознакомиться с методами ассоциативного анализа из библиотеки MLxtend
# # Ход работы
# ## Загрузка данных

# %%
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, fpmax, association_rules
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


# %%
all_data = pd.read_csv('groceries - groceries.csv')
all_data #Видно, что датафрейм содержит NaN значения


# %%
np_data = all_data.to_numpy()
np_data = [[elem for elem in row[1:] if isinstance(elem,str)] for row in
np_data]


# %%
unique_items = set()
for row in np_data:
    for elem in row:
        unique_items.add(elem)
print('Количество товаров', len(unique_items), unique_items)


# %%
te = TransactionEncoder()
te_ary = te.fit(np_data).transform(np_data)
data = pd.DataFrame(te_ary, columns=te.columns_)
data


# %%
fpg_result = fpgrowth(data, min_support=0.03, use_colnames = True).sort_values('support', ascending=False)
fpg_result


# %%
def printMinMaxSupport(result):
    curr_len = 1
    while True:
        sups = result[result['itemsets'].apply(lambda r: len(r) == curr_len)]['support']
        if len(sups) == 0:
            break
        print('Длина набора {len}: поддержка [{min}, {max}]'.format(len=curr_len, min=round(np.min(sups), 5), max=round(np.max(sups), 5)))
        curr_len += 1
printMinMaxSupport(fpg_result)


# %%
fpm_result = fpmax(data, min_support=0.03, use_colnames = True).sort_values('support', ascending=False)
fpm_result


# %%
printMinMaxSupport(fpm_result)


# %%
plt.xlabel('Количество попаданий товара в транзакцию')
data.sum().nlargest(10).sort_values().plot.barh()


# %%
plt.xlabel('Уровень поддержки')
fpg_result.set_index('itemsets')['support'].nlargest(10).sort_values().plot.barh()


# %%
plt.xlabel('Уровень поддержки')
fpm_result.set_index('itemsets')['support'].nlargest(10).sort_values().plot.barh()


# %%
items = ['whole milk', 'yogurt', 'soda', 'tropical fruit', 'shopping bags', 'sausage', 'whipped/sour cream', 'rolls/buns', 'other vegetables', 'root vegetables', 'pork', 'bottled water', 'pastry', 'citrus fruit', 'canned beer', 'bottled beer']
np_data_new = all_data.to_numpy()
np_data_new = [[elem for elem in row[1:] if isinstance(elem,str) and elem in items] for row in np_data_new]


# %%
te_new = TransactionEncoder()
te_ary_new = te_new.fit_transform(np_data_new)
data_new = pd.DataFrame(te_ary_new, columns=te_new.columns_)
data_new


# %%
fpg_result_new = fpgrowth(data_new, min_support=0.03, use_colnames = True).sort_values('support', ascending=False)
fpg_result_new


# %%
fpm_result_new = fpmax(data_new, min_support=0.03, use_colnames = True).sort_values('support', ascending=False)
fpm_result_new


# %%
printMinMaxSupport(fpg_result_new)
printMinMaxSupport(fpm_result_new)


# %%
min_supports = np.arange(0.0, 1, 0.01)
sup_data = []

for min_support in np.logspace(-3, 0, num=20):
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
df_count_by_lens.plot(ax=ax, x='min_support', logy='sym', logx=True, colormap='rainbow')
ax.set_axisbelow(True)
ax.grid(0.6)
ax.set_ylabel('Количество наборов')
ax.set_xlabel('Уровень поддержки')
fig.tight_layout()

# %% [markdown]
# ## Ассоциативные правила

# %%
np_data_a = all_data.to_numpy()
np_data_a = [[elem for elem in row[1:] if isinstance(elem,str) and elem in items] for row in np_data_a]
np_data_a = [row for row in np_data_a if len(row) > 1]
te_a = TransactionEncoder()
te_ary_a = te_a.fit_transform(np_data_a)
data_a = pd.DataFrame(te_ary_a, columns=te_a.columns_)
data_a


# %%
result = fpgrowth(data_a, min_support=0.05, use_colnames = True)
result


# %%
rules_conf = association_rules(result, min_threshold = 0.3)
rules_conf


# %%
rules_sup = association_rules(result, min_threshold = 0.01, metric='support')
rules_sup


# %%
rules_lift = association_rules(result, min_threshold = 0.01, metric='lift')
rules_lift


# %%
rules_leverage = association_rules(result, min_threshold = 0.01, metric='leverage')
rules_leverage


# %%
rules_conviction = association_rules(result, min_threshold = 0.01, metric='conviction')
rules_conviction


# %%
def getStatistic(rules, metric):
    return f'mean = {round(rules[metric].mean(), 5)}', f'median = {round(rules[metric].median(), 5)}', f'std = {round(rules[metric].std(), 5)}'


# %%
print('support', getStatistic(rules_sup, 'support'))
print('confidence', getStatistic(rules_conf, 'confidence'))
print('lift', getStatistic(rules_lift, 'lift'))
print('leverage', getStatistic(rules_leverage, 'leverage'))
print('conviction', getStatistic(rules_conviction, 'conviction'))


# %%
rules_ = association_rules(result, min_threshold=0.4, metric='confidence')
rules_


# %%
digraph = nx.DiGraph()
for rule in rules_.itertuples(index=False):
    digraph.add_edge(rule.antecedents, rule.consequents, weight=rule.support, label=round(rule.confidence, 3))
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(digraph)
nx.draw(digraph, pos,
    labels={node: '\n'.join(node) for node in digraph.nodes()},
    width=[digraph[u][v]['weight']*30 for u,v in digraph.edges()],
    node_size=2000
)
nx.draw_networkx_edge_labels(digraph, pos, edge_labels=nx.get_edge_attributes(digraph, 'label'))
plt.axis('off')
plt.show()


# %%
rules_pivot = rules_.pivot(index='antecedents', columns='consequents', values='confidence')
rules_pivot.index = ['\n'.join(ind) for ind in rules_pivot.index]
rules_pivot.columns = ['\n'.join(col) for col in rules_pivot.columns]
sns.heatmap(rules_pivot, cmap='rainbow')
plt.tight_layout()
plt.show()


