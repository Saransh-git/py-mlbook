"""
Machine Learning (CSP 584 Assignment 3) question 2
Authored by Saransh Kumar (A20424637)
Environment: Python 3.7.1
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
import statsmodels.api as stats

purchase_df: DataFrame = pd.read_csv("~/Desktop/machine learning/assignments/Purchase_Likelihood.csv")

purchase_df['A'] = purchase_df['A'].astype('category')

print(f"Parameters in model with only intercept terms: {len(purchase_df['A'].unique()) -1}")
freq_tab_by_a = pd.crosstab(purchase_df['A'],['Count'], rownames=['A'], margins=True).reset_index()  # generate a
# marginal frequency table

print("Marginal Counts:\n")
print(freq_tab_by_a)

total_count = int(freq_tab_by_a.loc[freq_tab_by_a['A'] == 'All']['Count'])
log_likelihood_value = 0
probs = []
for category in freq_tab_by_a['A']:
    if category == 'All':
        continue
    marginal_count = int(freq_tab_by_a['Count'][category])
    pred_prob = marginal_count/total_count  # compute predicted probability
    probs.append(pred_prob)
    print(f"Predicted probability of category {category}: {pred_prob}")
    log_likelihood_value += marginal_count*np.log(pred_prob)  # computing log likelihood value


print(f"Log likelihood of intercept only model: {log_likelihood_value}")

for idx in range(len(probs)):
    print(f"Beta-{idx + 1}-{0}: {np.log(probs[idx]/probs[0])}")  # computing MLE intercepts


group_size = purchase_df['group_size'].values
homeowner = purchase_df['homeowner'].values
married_couple = purchase_df['married_couple'].values
a = purchase_df['A'].values


freq_tab: DataFrame = pd.crosstab(
    [group_size, homeowner, married_couple], a, rownames=['group_size', 'homeowner','married_couple'],
    colnames=['A']
)  # generating the contingency table by frequency

percentTable = freq_tab.div(freq_tab.sum(1), axis='index')*100  # converting the contingency table to percentage

print(f"Grouped contingency table:\n"
      f"{percentTable}")

X = pd.get_dummies(purchase_df['group_size'], prefix='group_size')
X = X.join(pd.get_dummies(purchase_df['homeowner'], prefix='homeowner'))
X = X.join(pd.get_dummies(purchase_df['married_couple'], prefix='married_couple'))
X = stats.add_constant(X, prepend=True)  # adjustment required for intercept term


logit = stats.MNLogit(
    purchase_df['A'], X
)

print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)


thisFit = logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
thisParameter: DataFrame = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)  # Computing model parameters
print("Model Log-Likelihood Value: ", logit.loglike(thisParameter.values))  # Computing the log likelihood from
# losgistic model

logit_categ_1 = X.dot(thisParameter.iloc[:, 0])  # logit of categ_1 to ref categ_0
logit_categ_2 = X.dot(thisParameter.iloc[:, 1])  # logit of categ_2 to ref categ_0

prob_categ_1_to_0 = np.exp(logit_categ_1)  # odds of categ_1 to ref categ_0

print(f"Values of group_size, homeowner, married_couple for which odd Prob(A=1)/Prob(A = 0) will attain its maximum"
      f" value of {prob_categ_1_to_0.max()}:\n")
print(X.iloc[prob_categ_1_to_0.idxmax(), 1:])  # fetching sub-population where odds is maximum

print(f"Odds ratio for group size=3 versus group size=1, and A=2 versus A=0: "
      f"{np.exp(thisParameter.loc['group_size_3'][1] - thisParameter.loc['group_size_1'][1])}")  # Computing the odds
# ratio for group size=3 versus group size=1, and A=2 versus A=0

params_categ_2_to_categ_1 = thisParameter.iloc[:,1] - thisParameter.iloc[:,0]  # Computing the parameter making ref
# category as 1

print(f"Odds ratio for group size=1 versus group size=3, and A=2 versus A=1: "
      f"{np.exp(params_categ_2_to_categ_1.loc['group_size_1'] - params_categ_2_to_categ_1.loc['group_size_3'])}")
# Odds ratio for group size=1 versus group size=3, and A=2 versus A=1
