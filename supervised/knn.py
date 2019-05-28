"""
Machine Learning (CSP 584 Assignment 1) question 3
Authored by Saransh Kumar (A20424637)
Environment: Python 3.7.1
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier as kNN


# read the dataset
fraud_df = pd.read_csv(
    "~/Desktop/machine learning/assignments/Fraud.csv",
    index_col="CASE_ID"
)

fraud = fraud_df.groupby('FRAUD').get_group(1)  # fetch the fraudulent observations.
print(f"Percentage of investigations found to be fraudulent: "
      f"{round(fraud['FRAUD'].count()/fraud_df['FRAUD'].count() * 100, 4)}%"
)  # compute the fraudulent observations percentage


def box_plotters():
    """
    A function to plot the box-plots for interval attributes/ variables.
    """
    for col in ['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']:
        fraud_df.boxplot(column=col, by='FRAUD', vert=False)
        axs = plt.gca()
        fig = plt.gcf()
        axs.set_xlabel(col)
        axs.set_ylabel("Fraud")
        fig.suptitle("")
    plt.show()


box_plotters()  # plot box plots for interval variables in a go.

data_inp = np.asmatrix(fraud_df[list(fraud_df.columns)[1:]].to_numpy())  # transform input observations to numpy matrix
# with target variable excluded.

xtx = data_inp.transpose() * data_inp  # compute covariance matrix.
evals, evecs = LA.eigh(xtx)  # compute eigenvalues and eigenvectors for covariance matrix.

print(f"Eigenvalues {evals}")  # list the eigenvalues
print(f"Eigenvectors {evecs}")  # list the eigenvectors

transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)))  # compute the transformation matrix
print("Transformation Matrix = \n", transf)

transf_x = data_inp * transf  # compute the transformed matrix
print("The Transformed x = \n", transf_x)

# Proof for orthogonalization
xtx = transf_x.transpose() * transf_x  # this should yield an identity matrix as orthogonalized variables would
# have zero correlation between them.
print("Expect an Identity Matrix = \n", xtx)

kNNSpec = kNN(n_neighbors=5, algorithm='brute')  # specify the parameters for KNN classifier.
knn_fit = kNNSpec.fit(transf_x, fraud_df['FRAUD'])  # fit the model.
print(f"Accuracy for model: {round(knn_fit.score(transf_x, fraud_df['FRAUD']), 4)}")  # compute accuracy of model on
# training data.

new_inp = [
    [7500, 15, 3, 127, 2, 2]
]  # test input.

transf_inp = new_inp * transf  # transform test input.
transf_inp_nbrs = knn_fit.kneighbors(transf_inp, return_distance=False)  # compute the nearest neighbors for the
# test input, the distance values are not needed and hence, has been omitted.

# print the input and output values for nearest neighbors.
for nbr in transf_inp_nbrs.tolist()[0]:
    print(f"Neighbor indexed {nbr} with\n input: {transf_x[nbr].tolist()[0]} and output: {fraud_df['FRAUD'][nbr]}")

print(f"Predicted output: {knn_fit.predict(transf_inp)}")  # predict class for the test input.
