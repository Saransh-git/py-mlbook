"""
Machine Learning (CSP 584 Assignment 1) question 1
Authored by Saransh Kumar (A20424637)
Environment: Python 3.7.1
"""

import matplotlib.pyplot as plt
import pandas as pd

# loading the dataset
nm_samp_df = pd.read_csv(
    "~/Desktop/machine learning/assignments/NormalSample.csv",
    index_col="i"
)

# Empirical Stats on x
x_summary = nm_samp_df['x'].describe()

print(x_summary)
x_iqr = x_summary['75%'] - x_summary['25%']  # calculating iqr = Q3 - Q1
print(f'IQR: {x_iqr}')


rec_bin_width = round(2 * x_iqr * int(nm_samp_df['x'].count()) ** (-1/3), 4) # calculating recommended bin width


print(f"Recommended bin width: {rec_bin_width}")


def seq_generator(start, end, step=1):
    """
    A function to generate sequence from start to end with a step provided by the step parameter
    :param start: start of sequence (inclusive)
    :param end: end of sequence (inclusive)
    :param step: step for sequence valuesb
    :return: generator for sequence
    """
    while start <= end:
        yield start
        start = round(start + step, 2)


bin_widths = [0.1, 0.5, 1, 2]  # histogram bin widths as given in question
for i, bin_width in enumerate(bin_widths):
    plt.figure(i)  # generate different figures numbered on index.
    axs = plt.gca()  # fetch current axes.
    bin_edges = list(seq_generator(26.0, 36.0, bin_width))  # compute the bin edges.
    md_pts = [edge + bin_width / 2 for edge in list(seq_generator(26, 36, bin_width))][:-1]  # compute the mid points.
    axs.set_title(f"Histogram with bin width {bin_width}")
    axs.set_xlabel("X")
    axs.set_ylabel("density")
    # axs.set_xticks(md_pts) This is to label the x axis with the mispoints, but doing so makes the plot clumsier and
    # dirty and thus, has been avoided.
    ret_val = axs.hist(nm_samp_df['x'], bins=bin_edges, density=True)  # returns a tuple of y vals, x vals and
    # other custom parameters
    yvals = ret_val[0]  # fetching the density values
    print(bin_edges)
    print(f"Co-ordinates for density estimator with bin-width {bin_width}: "
          f"{[(round(md_pts[j],2), round(yvals[j],6)) for j in range(len(md_pts))]}")  # prints the co-ordinates of
    # density estimator.
    '''
    Note: This block is to label the graph with the co-ordinates of density estimator, but again makes the histograms
    clumsier and thus, has been avoided.
    for j in range(len(md_pts)):
        axs[i // 2][i % 2].text(md_pts[j] - bin_width / 2, yvals[j] + 0.001, round(yvals[j], 3))
    '''

plt.show()
