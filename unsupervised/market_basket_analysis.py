"""
Machine Learning (CSP 584 Assignment 2) question 2
Authored by Saransh Kumar (A20424637)
Environment: Python 3.7.1
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt


groceries_df: DataFrame = pd.read_csv("~/Desktop/machine learning/assignments/Groceries.csv")

unique_customers = groceries_df['Customer'].unique()  # list of unique customers
print(f"Unique customers in market basket: {len(unique_customers)}")

print(f"Unique items in market basket across all Customers: {len(groceries_df['Item'].unique())}")

grouped_groceries_by_cust = groceries_df.groupby('Customer')  # grouping groceries by customer


records = []
for cust in unique_customers:
    records.append((cust, len(grouped_groceries_by_cust.get_group(cust)['Item'].unique())))

groceries_cust_item_count = DataFrame.from_records(
    records, columns=['Customer', 'Unique item count'], index='Customer'
)  # a dataframe of customer id and number of unique items purchased.


def compute_beautified_bin_width(col_data):
    """
    Trying to compute suggested and beautified bin width.
    The same has not been used and a bin width of 1 has been taken to produce histograms.
    """
    summary = col_data.describe()
    print(summary)
    suggested_bin_width = 2 * (summary['75%'] - summary['25%']) / \
                          summary['count'] ** (1 / 3)
    print(f"Suggested bin width: {suggested_bin_width}")
    u = np.log10(suggested_bin_width)
    v = np.sign(u) * np.ceil(abs(u))
    beautified_bin_width = 10 ** v
    print(f"Suggested bin width rounded to a nice value: {beautified_bin_width}")
    print("Suggested bin width is not used. "
          "A bin width of 1 has been used to display frequency for each unique count")


def compute_bin_edges_for_bin_width(col_data: Series, bin_width: int):
    """
    Computes bin edges for the provided bin width for the data provided
    :param col_data: Column data/ Series for which histogram has to be drawn.
    :param bin_width:
    """
    max = int(col_data.max())
    min = int(col_data.min())
    edge = min
    while edge <= max:
        yield edge
        edge = edge + bin_width


compute_beautified_bin_width(groceries_cust_item_count['Unique item count'])

# since, the suggested bin width and the beautified bin width gives value of 0.373 and 0.1 respectively, which is not
# appropriate as item counts vary as integer which would lead to empty bins.
# Therefore, following bin sizes are tried.

for bin_width in [1]:  # can be tried with other bin widths of 4,5 to get a better understanding of distribution.
    # Plotting with bin width 1 to plot frequency against each item count.
    plt.figure()
    axs = plt.gca()
    item_count_series = groceries_cust_item_count['Unique item count']
    bin_edges = list(compute_bin_edges_for_bin_width(item_count_series, bin_width))
    retval = axs.hist(
        item_count_series,
        bins=bin_edges,
        label=None,
    )
    axs.set_title(f"Histogram with bin width {bin_width}")
    axs.set_ylabel("Unique item counts")
    axs.set_xlabel("Frequency")

    plt_height = max(retval[0])
    # Plotting vertical lines to demonstrate Q1, Q2 and Q3
    axs.vlines(x=item_count_series.quantile(q=0.25), ymin=0, ymax=plt_height+100, label="Q1", linestyles='solid')
    axs.vlines(x=item_count_series.quantile(q=0.5), ymin=0, ymax=plt_height+100, label="Median", linestyles='dashed')
    axs.vlines(x=item_count_series.quantile(q=0.75), ymin=0, ymax=plt_height+100, label="Q3", linestyles='dotted')
    axs.legend()

# Viewing the plots, a bin width of 4 is decided and the same is reported.
plt.show()

item_list = list(grouped_groceries_by_cust['Item'].apply(list))  # converts the data into Item list format.

te = TransactionEncoder()
te_ary = te.fit(item_list).transform(item_list)  # converts to item indicator format.
trainData = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets: DataFrame = apriori(
    trainData, min_support=75/len(unique_customers),
    use_colnames=True,
    max_len=np.median(groceries_cust_item_count['Unique item count'].values)
)  # generates frequent itemset against min support and max length itemset provided.

print(f"\nTotal frequent itemsets: {frequent_itemsets['itemsets'].count()}")

max_len = 0
for itemset in frequent_itemsets['itemsets']:
    max_len = max(max_len, len(itemset))

print(f"Maximum length frequent itemset: {max_len}")


conf_itemset: DataFrame = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.01)
print(f"Association rules found: {len(conf_itemset[['antecedents', 'consequents']])}")

plt.figure()
axs = plt.gca()
p = axs.scatter(conf_itemset['confidence'], conf_itemset['support'], s=conf_itemset['lift'], c=conf_itemset['lift'])
axs.set_title(f"Support vs Confidence")
axs.set_xlabel("Confidence")
axs.set_ylabel("Support")
axs.grid(True)
fig = plt.gcf()
cbar = fig.colorbar(p)  # generates a colorbar for the plot. This is used to show color indicators for Lift as used by
# the plot
cbar.set_label("Lift")
plt.show()


itemset_with_conf_gt_sixty_pct = conf_itemset.loc[
    conf_itemset['confidence'] >= 0.6]

print("Rule produced with atleast 60% confidence:\n")
print(itemset_with_conf_gt_sixty_pct)

consequent_list = list(itemset_with_conf_gt_sixty_pct['consequents'].apply(list))

# computing common consequent
cons_set = set(consequent_list[0])
for index in range(1, len(consequent_list)):
    cons_set = cons_set.intersection(set(consequent_list[index]))

print(f"Common items in consequents: {cons_set}")
