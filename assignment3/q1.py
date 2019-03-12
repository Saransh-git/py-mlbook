import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from itertools import combinations

car_ownership_df: DataFrame = pd.read_csv(
    "~/Desktop/machine learning/assignments/CustomerSurveyData.csv", index_col='CustomerID'
)[['CarOwnership', 'CreditCard', 'JobCategory']]

print(car_ownership_df['CarOwnership'].unique())

# car_ownership_df = car_ownership_df[car_ownership_df['CarOwnership'] != 'None']
print(car_ownership_df.head())
print(car_ownership_df['CreditCard'].unique())
print(car_ownership_df['JobCategory'].unique())
car_ownership_df.loc[car_ownership_df['JobCategory'].isna(), 'JobCategory'] = 'Missing'


def compute_entropy_and_node_count(node):
    entropy_node = 0
    num_obs = node.shape[0]
    for class_label in node['CarOwnership'].unique():
        label_cnt = node.loc[node['CarOwnership'] == class_label].shape[0]
        entropy_node += label_cnt / num_obs * np.log2(label_cnt / num_obs)
    return -1 * entropy_node, num_obs


print(f"Entropy at root: {compute_entropy_and_node_count(car_ownership_df)[0]}")

unique_credit_cards = car_ownership_df['CreditCard'].unique()
print(f"Total number of binary splits possible from the CredictCard predictor: "
      f"{2 ** (len(unique_credit_cards) - 1) -1}")


def split_generator(col_data: Series):
    """
    Generates the left set of the binary split (as right split is redundant)
    :param col_data: Accepts column data like car_ownership_df['CreditCard']
    :return: A list of left split sets
    """
    binary_splits = []
    unique_items = col_data.unique()  # fetch unique column data
    for item in unique_items:
        binary_splits.append([item])  # add single items

    for root_idx in range(len(unique_items)):  # iterate over to have the first item of set
        for num_combinations in range(1, len(unique_items) - root_idx - 1):  # produce combinations over the remaining
            # elements to the right. (Only right elements to be considered as considering left elements would produce
            # redundant sets.)
            for item in list(combinations(range(root_idx + 1, len(unique_items)), num_combinations)):
                split = [unique_items[root_idx]]
                split = split + unique_items[
                    list(item)
                ].tolist()  # append elements to split set
                binary_splits.append(split)  # append the split set

    # Now, many redundant sets might have been generated meaning, that a set represents the right split set of a set
    # already present in the binary splits. Remove such splits.
    for i in range(len(binary_splits)):
        for j in range(i+1, len(binary_splits)):
            try:
                if sorted(binary_splits[i] + binary_splits[j]) == sorted(unique_items.tolist()):
                    binary_splits.pop(j)
            except IndexError:
                break

    return binary_splits


credit_card_binary_split = split_generator(car_ownership_df['CreditCard'])
print(len(credit_card_binary_split))
records = []

for index, split in enumerate(credit_card_binary_split):
    left_split = car_ownership_df.loc[car_ownership_df['CreditCard'].isin(split)]
    right_split = car_ownership_df.loc[~car_ownership_df['CreditCard'].isin(split)]
    left_entropy, left_node_count = compute_entropy_and_node_count(left_split)
    right_entropy, right_node_count = compute_entropy_and_node_count(right_split)
    total_count = left_node_count + right_node_count
    split_entropy = left_node_count/total_count * left_entropy + right_node_count/total_count * right_entropy
    records.append((index, split, split_entropy))


credit_card_split_df = DataFrame.from_records(records, columns=['Index', 'Split', 'Split Entropy'])
credit_card_split_df.set_index('Index')
print(credit_card_split_df)

print(f"Split with the minimum value:\n"
      f"{credit_card_split_df.iloc[credit_card_split_df['Split Entropy'].idxmin(), :]}")

unique_job_categories = car_ownership_df['JobCategory'].unique()
print(unique_job_categories)
print(f"Total number of binary splits possible from the JobCategory predictor: "
      f"{2 ** (len(unique_job_categories) - 1) -1}")


job_category_split = split_generator(car_ownership_df['JobCategory'])

print(len(job_category_split))

records = []
for index, split in enumerate(job_category_split):
    left_split = car_ownership_df.loc[car_ownership_df['JobCategory'].isin(split)]
    right_split = car_ownership_df.loc[~car_ownership_df['JobCategory'].isin(split)]
    left_entropy, left_node_count = compute_entropy_and_node_count(left_split)
    right_entropy, right_node_count = compute_entropy_and_node_count(right_split)
    total_count = left_node_count + right_node_count
    split_entropy = left_node_count/total_count * left_entropy + right_node_count/total_count * right_entropy
    records.append((index, split, split_entropy))

job_category_split_df = DataFrame.from_records(records, columns=['Index', 'Split', 'Split Entropy'])
job_category_split_df.set_index('Index')
print(job_category_split_df)

print(f"Job Category Split with the minimum value:\n"
      f"{job_category_split_df.iloc[job_category_split_df['Split Entropy'].idxmin(), :]}")
