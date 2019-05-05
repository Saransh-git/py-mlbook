import concurrent
import math
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as stats
from matplotlib.axes import Axes
from pandas import DataFrame
import time

start = time.time()

five_ring_df: DataFrame = pd.read_csv("~/Desktop/machine learning/assignments/FiveRing.csv")
five_ring_df['ring'] = five_ring_df['ring'].astype('category')
lt_input = five_ring_df[['x', 'y']]
lt_input = stats.add_constant(lt_input, prepend=True)

y = pd.get_dummies(five_ring_df['ring'], prefix='rings')


def _fit_logistic(train_X, train_Y):
    logit = stats.MNLogit(
        train_Y, train_X
        # mnlogit treats every distinct value as a separate category and therefore, need not
        # pass the dummies in exogenous variable.
    )
    fit = logit.fit(full_output=True, maxiter=1000)
    return fit, fit.params


thisParameter: DataFrame

thisFit, thisParameter = _fit_logistic(lt_input, five_ring_df['ring'])
print("Model Parameter Estimates:\n", np.round(thisFit.params, 4))  # Computing model parameters

odds: DataFrame = np.exp(lt_input.dot(thisParameter))
prob_ring_0 = 1/(1+ odds.sum(axis=1))

records = []

for obs_idx in range(odds.shape[0]):
    """
    Computing probabilities, additionally can be done via thisFit.predict(lt_input)
    """
    prob_0 = prob_ring_0[obs_idx]
    record = [prob_0]
    record = record + (prob_0 * odds.iloc[obs_idx, :]).to_list()
    records.append(record)

probs: DataFrame = pd.DataFrame.from_records(records, columns=[0, 1, 2, 3, 4])


def _compute_misclassification_rate_and_rase_and_labels(true_label, pred_probs):
    pred_labels = pred_probs.idxmax(axis=1)
    misclassification_count = sum(true_label != pred_labels)
    rase = 0
    for obs_idx in range(pred_labels.shape[0]):
        true_ = true_label[obs_idx]
        prob_obs_ = pred_probs.iloc[obs_idx, :]

        for col in pred_probs.columns:
            if true_ == col:
                rase += (1 - prob_obs_[col]) ** 2
            else:
                rase += (0 - prob_obs_[col]) ** 2
    rase = math.sqrt(rase/(true_label.shape[0] * 2))
    return misclassification_count/true_label.shape[0], rase, pred_labels


def _plot_rings(labels, bootstrap_trial_num):
    cmap = ['orange', 'green', 'blue', 'black', 'red']
    plt.figure()
    axs: Axes = plt.gca()
    for col in range(5):
        members = five_ring_df[labels == col]
        axs.scatter(members['x'], members['y'], color=cmap[col], label=col, s=0.5)
    axs.set_xlabel("X")
    axs.set_ylabel("Y")
    axs.set_title(f"Y vs X with bootstrap trials {bootstrap_trial_num}")
    axs.grid(True)
    axs.legend()
    plt.show()


records.clear()  # for storing a dict of bootstrap trial and corresponding metrics
misclassification_rate, rase, pred_labels = _compute_misclassification_rate_and_rase_and_labels(
    five_ring_df['ring'], probs
)
print(f"Misclassification rate: {misclassification_rate}")
print(f"RASE: {rase}")
records.append({'trial': 0, 'misclassification_rate': misclassification_rate, 'rase': rase})
_plot_rings(pred_labels, 0)

bootstrap_trials = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
random.seed(20190430)
bootstrap_probs = []
for trial in bootstrap_trials:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=trial)  # add concurrency of num_trials as well as
    # parallelism equal to the number of CPU cores.
    pred_probs: Optional[DataFrame] = None
    exec_res = {}
    for _ in range(trial):
        idxs = random.choices(lt_input.index, k=lt_input.shape[0])
        samp = lt_input.iloc[idxs]
        exec_res[
            executor.submit(_fit_logistic, samp, five_ring_df['ring'][idxs])
        ] = samp
    executor.shutdown(wait=False)  # asynchronous shutdown here
    done, not_done = concurrent.futures.wait(exec_res, timeout=20)  # allow a maximum of 20 seconds for logistic
    # model to fit itself.

    # import ipdb;ipdb.set_trace()
    for exec_ in done:
        fit, params = exec_.result()
        if pred_probs is None:
            pred_probs = fit.predict(lt_input)
        else:
            pred_probs += fit.predict(lt_input)
    pred_probs = pred_probs/trial
    bootstrap_probs.append(pred_probs)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(bootstrap_trials))  # again doing a concurrent
# computation of misclassification rate and rase for different trials.
exec_res = {}
for boot_prob, trial in zip(bootstrap_probs, bootstrap_trials):
    exec_res[
        executor.submit(_compute_misclassification_rate_and_rase_and_labels, five_ring_df['ring'], boot_prob)
    ] = trial
executor.shutdown(wait=False)  # asynchronous shutdown here

done, not_done = concurrent.futures.wait(exec_res, timeout=100)  # allow a maximum of 100 seconds for
# metrics computation

for exec_ in done:
    misclassification_rate, rase, pred_labels = exec_.result()
    trial = exec_res[exec_]
    print(f"Misclassification rate for trial {trial}: {misclassification_rate}")
    print(f"RASE for trial {trial}: {rase}")
    records.append({'trial': trial, 'misclassification_rate': misclassification_rate, 'rase': rase})
    _plot_rings(pred_labels, trial)

metrics: DataFrame = pd.DataFrame.from_records(records)
metrics.sort_values('trial', inplace=True)
metrics.set_index('trial', inplace=True)
print(metrics)
end = time.time()

print(end - start)
