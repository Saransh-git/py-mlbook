"""
Machine Learning (CSP 584 Assignment 2) question 1
Authored by Saransh Kumar (A20424637)
Environment: Python 3.7.1
"""
from itertools import combinations  # generates combination of items

items = ['A', 'B', 'C', 'D', 'E', 'F', 'G']


print(f"Total number of itemsets possible: {2 ** len(items) - 1}\n")
print(f"1-itemset: {list(combinations(items, 1))}")
print(f"2-itemset: {list(combinations(items, 2))}")
print(f"3-itemset: {list(combinations(items, 3))}")
print(f"4-itemset: {list(combinations(items, 4))}")
print(f"5-itemset: {list(combinations(items, 5))}")
print(f"6-itemset: {list(combinations(items, 6))}")
print(f"7-itemset: {list(combinations(items, 7))}")

