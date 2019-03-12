import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas.core.generic import NDFrame
import statsmodels.api as stats
from math import log

purchase_df: DataFrame = pd.read_csv("~/Desktop/machine learning/assignments/Purchase_Likelihood.csv")

purchase_df['A'] = purchase_df['A'].astype('category')

freq_tab_by_a = pd.crosstab(purchase_df['A'],['Count'], rownames=['A'], margins=True).reset_index()

print("Marginal Counts:\n")
print(freq_tab_by_a)

total_count = int(freq_tab_by_a.loc[freq_tab_by_a['A'] =='All']['All'])
log_likelihood_value = 0
probs = []
for category in freq_tab_by_a['A']:
    if category == 'All':
        continue
    marginal_count = int(freq_tab_by_a['Count'][category])
    pred_prob = marginal_count/total_count
    probs.append(pred_prob)
    print(f"Predicted probability of category {category}: {pred_prob}")
    log_likelihood_value += marginal_count*np.log(pred_prob)


print(f"Log likelihood of intercept only model: {log_likelihood_value}")

for idx in range(len(probs)):
    print(f"Beta-{idx}-{0}: {np.log(probs[idx]/probs[0])}")


group_size = purchase_df['group_size'].values
homeowner = purchase_df['homeowner'].values
married_couple = purchase_df['married_couple'].values
a = purchase_df['A'].values


freq_tab: DataFrame = pd.crosstab(
    [group_size, homeowner, married_couple], a, rownames=['group_size', 'homeowner','married_couple'],
    colnames=['A']
)

percentTable = freq_tab.div(freq_tab.sum(1), axis='index')*100

X = pd.get_dummies(purchase_df['group_size'], prefix='group_size')
X = X.join(pd.get_dummies(purchase_df['homeowner'], prefix='homeowner'))
X = X.join(pd.get_dummies(purchase_df['married_couple'], prefix='married_couple'))
X = stats.add_constant(X, prepend=True)


logit = stats.MNLogit(
    purchase_df['A'], X
)

print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)


thisFit = logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
thisParameter: DataFrame = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value: ", logit.loglike(thisParameter.values))

logit_categ_1 = X.dot(thisParameter.iloc[:, 0])
logit_categ_2 = X.dot(thisParameter.iloc[:, 1])

prob_categ_1_to_0 = np.exp(logit_categ_1)

print(f"Values of group_size, homeowner, married_couple for which odd Prob(A=1)/Prob(A = 0) will attain its maximum"
      f" value of {prob_categ_1_to_0.max()}:\n")
print(X.iloc[prob_categ_1_to_0.idxmax(), 1:])  # part(i)

print(f"Odds ratio for group size=3 versus group size=1, and A=2 versus A=0: "
      f"{np.exp(thisParameter.loc['group_size_3'][1] - thisParameter.loc['group_size_1'][1])}")  # part(j)

params_categ_2_to_categ_1 = thisParameter.iloc[:,1] - thisParameter.iloc[:,0]

print(f"Odds ratio for group size=1 versus group size=3, and A=2 versus A=1: "
      f"{np.exp(params_categ_2_to_categ_1.loc['group_size_1'] - params_categ_2_to_categ_1.loc['group_size_3'])}")
# part(k)
