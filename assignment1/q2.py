"""
Machine Learning (CSP 584 Assignment 1) question 2
Authored by Saransh Kumar (A20424637)
Environment: Python 3.7.1
"""

import matplotlib.pyplot as plt
import pandas as pd

# read dataset
nm_samp_df = pd.read_csv(
    "~/Desktop/machine learning/assignments/NormalSample.csv",
    index_col="i"
)

x_desc = nm_samp_df['x'].describe()  # Empirical Stats on x

print(f"Five-number summary:\n{x_desc}")
iqr = x_desc['75%'] - x_desc['25%']  # computing IQR: Q3-Q1

whisk_x = (
    max(x_desc['25%'] - 1.5*iqr, nm_samp_df['x'].min()),
    min(x_desc['75%'] + 1.5*iqr, nm_samp_df['x'].max())
)  # computing the whiskers for X (un-grouped)

print(f"Whiskers for X:{whisk_x}")
print(nm_samp_df.groupby(['group'])['x'].describe())  # produce empirical stats on 'x' grouped by 'group'

plt.figure(1)  # Figure-1 to depict boxplot for un-grouped x.

axs = plt.gca()

axs.set_title("Boxplot for X")
axs.set_xlabel("X")
axs.set_ylabel("")
axs.boxplot(nm_samp_df['x'], vert=False, whis=1.5)  # produce boxplot for un-grouped x.
axs.vlines([
    x_desc['25%'], x_desc['75%'],
    max(
        x_desc['25%'] - 1.5*iqr, nm_samp_df['x'].min()
    ),
    min(
        x_desc['75%'] + 1.5*iqr, nm_samp_df['x'].max()
    )],
    ymin=0, ymax=5, linestyles='dotted'
)  # generate vertical lines on left and right whiskers and on Q1 and Q3.

# label the vertical lines
axs.text(x_desc['25%'], 0.7, "Q1")
axs.text(x_desc['75%'], 0.7, "Q3")
axs.text(x_desc['25%'] - 1.5*iqr, 0.7, "Q1 - 1.5*IQR")
axs.text(x_desc['75%'] + 1.5*iqr, 0.7, "Q3 + 1.5*IQR")

plt.figure(2)  # second figure to depict combined box plots for un-grouped and grouped x.
axs = plt.gca()  # fetch current axes.
axs.grid(True)  # show grid lines.
x_grp_0 = nm_samp_df.groupby(['group']).get_group(0)['x']  # fetch group 0 elements.

x_grp_1 = nm_samp_df.groupby(['group']).get_group(1)['x']  # fetch group 1 elements.

axs.boxplot([nm_samp_df['x'], x_grp_0, x_grp_1], vert=False, whis=1.5)  # plot combined box plots.
axs.set_title("Boxplots")
axs.set_xlabel("X values")
axs.set_yticklabels(['X', 'X group 0', 'X group 1'])  # set labels against different box plots.
plt.show()

x_grp_0_desc = x_grp_0.describe()  # produce empirical stats on 'x' for group 0.
x_grp_0_iqr = x_grp_0_desc['75%'] - x_grp_0_desc['25%']  # compute iqr for group 0.
whisk_x_grp_0 = (
    max(x_grp_0_desc['25%'] - 1.5 * x_grp_0_iqr, x_grp_0.min()),
    min(x_grp_0_desc['75%'] + 1.5 * x_grp_0_iqr, x_grp_0.max())
)  # compute whiskers for group 0 x.

x_grp_1_desc = x_grp_1.describe()  # produce empirical stats on 'x' for group 1.
x_grp_1_iqr = x_grp_1_desc['75%'] - x_grp_1_desc['25%']  # compute iqr for group 1.
whisk_x_grp_1 = (
    max(x_grp_1_desc['25%'] - 1.5 * x_grp_1_iqr, x_grp_1.min()),
    min(x_grp_1_desc['75%'] + 1.5 * x_grp_1_iqr, x_grp_1.max())
)  # compute whiskers for group 1 x.

print(f"Whiskers for X group 0: {whisk_x_grp_0}")
print(f"Whiskers for X group 1: {whisk_x_grp_1}")

outlying_x = pd.concat(
    [nm_samp_df['x'][nm_samp_df['x'] < whisk_x[0]],
     nm_samp_df['x'][nm_samp_df['x'] > whisk_x[1]]],
    ignore_index=False
)  # compute outliers for un-grouped X.

outlying_x_grp0 = pd.concat(
    [x_grp_0[x_grp_0 < whisk_x_grp_0[0]],
     x_grp_0[x_grp_0 > whisk_x_grp_0[1]]],
    ignore_index=False
)  # compute outliers for group-0 X.

outlying_x_grp1 = pd.concat(
    [x_grp_1[x_grp_1 < whisk_x_grp_1[0]],
     x_grp_1[x_grp_1 > whisk_x_grp_1[1]]],
    ignore_index=False
)  # compute outliers for group-1 X.
print(f"Outlying when all X considered:\n{outlying_x}")
print(f"Outlying X group 0:\n{outlying_x_grp0}")
print(f"Outlying X group 1:\n{outlying_x_grp1}")
