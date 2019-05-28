import concurrent
import math
import random
from functools import partial

import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import tree

five_ring_df: DataFrame = pd.read_csv("~/Desktop/machine learning/assignments/FiveRing.csv")

five_ring_df['ring'] = five_ring_df['ring'].astype('category')


def _fit_tree(train_X, train_Y, sample_weights=None):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=20190415)
    fit = classTree.fit(train_X, train_Y, sample_weights)
    return fit


def _compute_misclassification_rate_and_rase_and_labels(true_label, pred_probs, boosting_max_iter=0):
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
    rase = math.sqrt(rase / (true_label.shape[0] * 2))
    misclassification_rate = misclassification_count / true_label.shape[0]
    print(f"Misclassification rate for boosting max iterations {boosting_max_iter}: {misclassification_rate}")
    print(f"RASE for boosting max iterations {boosting_max_iter}: {rase}")
    return misclassification_rate, rase, pred_labels


def _plot_rings(labels, boosting_iterations=0):
    cmap = ['orange', 'green', 'blue', 'black', 'red']
    plt.figure()
    axs: Axes = plt.gca()
    for col in range(5):
        members = five_ring_df[labels == col]
        axs.scatter(members['x'], members['y'], color=cmap[col], label=col, s=0.5)
    axs.set_xlabel("X")
    axs.set_ylabel("Y")
    axs.set_title(f"Y vs X with boosting iterations {boosting_iterations}")
    axs.grid(True)
    axs.legend()
    plt.show()


metrics_records = []
fit: tree.DecisionTreeClassifier = _fit_tree(five_ring_df[['x', 'y']], five_ring_df['ring'])
misclassification_rate, rase, pred_labels = _compute_misclassification_rate_and_rase_and_labels(
    five_ring_df['ring'], pd.DataFrame(fit.predict_proba(five_ring_df[['x', 'y']]))
)
metrics_records.append({'boosting_iter': 0, 'misclassification_rate': misclassification_rate, 'rase': rase})

_plot_rings(pred_labels)


def plot_async_callback(ft, boosting_iterations):
    misclassification_rate, rase, pred_labels = ft.result()
    metrics_records.append(
        {'boosting_iter': boosting_iterations, 'misclassification_rate': misclassification_rate, 'rase': rase}
    )
    _plot_rings(pred_labels, boosting_iterations)


boosting_max_iters = list(range(100, 1001, 100))
train_X = five_ring_df[['x', 'y']]
train_Y = five_ring_df['ring']
boosting_probs = []


exec_res = []
def _fit_boosting(train_X, train_Y, max_iter):
    print(f"Computing for max_iter {max_iter}")
    num_obs = train_X.shape[0]
    pred_probs_overall = None
    weights = np.ones(num_obs)
    accuracy_sum_over_iterations = 0
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=20190415)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # for asynchronous computation of metrics
    # and plotting
    boosting_iterable = iter(boosting_max_iters)
    try:
        next_stop = boosting_iterable.__next__()  # record metrics/ predicted probabilities at stops
    except StopIteration:
        next_stop = None

    for iter_ in range(1, max_iter+1, 1):
        print(f"On iteration {iter_}...")
        fit: tree.DecisionTreeClassifier = classTree.fit(train_X, train_Y, weights)
        pred_prob_iter: DataFrame = pd.DataFrame(fit.predict_proba(train_X))
        accuracy = fit.score(train_X, train_Y, weights)
        # print(f"reached iter {iter_} with accuracy {accuracy}")
        pred_label_iter = pred_prob_iter.idxmax(axis=1)
        accuracy_sum_over_iterations += accuracy
        if pred_probs_overall is None:
            pred_probs_overall = accuracy * pred_prob_iter
        else:
            pred_probs_overall += accuracy * pred_prob_iter

        if (iter_ == next_stop) or (accuracy == 1):
            boosting_probs.append(pred_probs_overall/accuracy_sum_over_iterations)
            ft = executor.submit(
                _compute_misclassification_rate_and_rase_and_labels, train_Y,
                pred_probs_overall/accuracy_sum_over_iterations, next_stop
            )
            ft.add_done_callback(partial(plot_async_callback, boosting_iterations=next_stop))
            exec_res.append(ft)
            try:
                next_stop = boosting_iterable.__next__()
            except StopIteration:
                pass

        if accuracy == 1:
            break

        for i in range(num_obs):
            prob_obs = pred_prob_iter.iloc[i, :]
            case_error = 0
            for j in pred_prob_iter.columns:
                if(train_Y[i] == j):
                    case_error += abs(1 - prob_obs[j])
                else:
                    case_error += abs(0 - prob_obs[j])

            if train_Y[i] == pred_label_iter[i]:
                weights[i] = 1/5 * case_error
            else:
                weights[i] = 1 + (1 / 5 * case_error)

    executor.shutdown(wait=False)


_fit_boosting(train_X, train_Y, max(boosting_max_iters))
concurrent.futures.wait(exec_res, timeout=100)  # allow all futures to complete within 100 seconds


for iter_ in boosting_max_iters[len(boosting_probs):]:  # extend the metrics if the iterations
    # converged on a smaller metric
    boosting_probs.append(boosting_probs[-1])
    metrics_records.append(
        {'boosting_iter': iter_, 'misclassification_rate': metrics_records[-1]['misclassification_rate'],
         'rase': metrics_records[-1]['rase']}
    )

metrics_df = pd.DataFrame.from_records(metrics_records)
print(metrics_df)
