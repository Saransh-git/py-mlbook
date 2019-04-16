import sys

import matplotlib.pyplot as plt
import numpy as np
from math import log, exp

from .Eval import Eval
from .imdb import IMDBdata


class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA  # smoothing parameter
        self.data = data  # training data
        # self.data.X = self.data.X[0:50, :]
        # self.data.Y = self.data.Y[0:50]
        self.vocab_len = self.data.vocab.GetVocabSize()
        self.pos_indices = np.argwhere(self.data.Y == 1).flatten()  # find the indices of positively labelled docs
        self.neg_indices = np.argwhere(self.data.Y == -1).flatten()  # find the indices of negative labelled docs

        self.pos_reviews = self.data.X[self.pos_indices]  # positive reviews
        self.num_positive_reviews = self.pos_reviews.shape[0]  # number of positive reviews
        self.neg_reviews = self.data.X[self.neg_indices]  # negative reviews
        self.num_negative_reviews = self.neg_reviews.shape[0]  # number of negative reviews

        self.total_positive_words = np.sum(self.pos_reviews)  # sum all the array elements in positive reviews array
        self.total_negative_words = np.sum(self.neg_reviews)  # sum all the array elements in negative reviews array

        self.P_positive = self.num_positive_reviews/(self.num_positive_reviews + self.num_negative_reviews)
        # Probability of positive review docs
        self.P_negative = self.num_negative_reviews/(self.num_positive_reviews + self.num_negative_reviews)
        # Probability of negative review docs

        self.deno_pos = self.vocab_len * ALPHA + self.total_positive_words  # denominator in conditional probs of
        # features in positive reviews
        self.deno_neg = self.vocab_len * ALPHA + self.total_negative_words  # denominator in conditional probs of
        # features in negative reviews

        self.count_positive = np.zeros(self.vocab_len)  # stores the count of features in positively marked docs.
        self.count_negative = np.zeros(self.vocab_len)  # stores the count of features in negatively marked docs.

        self.prob_features_pos_label = np.zeros(self.vocab_len)  # stores the probability of features given
        # class label positive
        self.prob_features_neg_label = np.zeros(self.vocab_len)  # stores the probability of features given
        # class label negative

        self.Train()

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self):
        """Training model parameters (conditional feature probs)"""
        for word_id in range(self.pos_reviews.shape[1]):
            self.count_positive[word_id] = np.sum(self.pos_reviews[:, word_id])  # update count
            self.prob_features_pos_label[word_id] = (self.ALPHA + self.count_positive[word_id])/self.deno_pos
            # update probs

        for word_id in range(self.neg_reviews.shape[1]):
            self.count_negative[word_id] = np.sum(self.neg_reviews[:, word_id])  # update count
            self.prob_features_neg_label[word_id] = (self.ALPHA + self.count_negative[word_id])/self.deno_neg
            # update probs
        
        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X, probThresh=0.5):
        """
        Predicts label based on the threshold provided. Takes 0.5 as default threshold. Predicts positive for
        probability greater than threshold, else negative.
        """
        pred_labels = []  # to store the predicted labels of new reviews.
        pred_probs = self.PredictProb(X)  # Predict probability

        for prob in pred_probs:
            if prob > probThresh:  # Predict positive if prob greater than threshold.
                pred_labels.append(+1.0)
            else:
                pred_labels.append(-1.0)

        '''
        for index in range(X.shape[0]):
            doc = X[index, :]
            log_prob_pos, log_prob_neg = self.compute_log_probability_numerator(doc)

            if log_prob_pos > probThresh:
                pred_labels.append(+1.0)
            else:
                pred_labels.append(-1.0)
        '''
        return pred_labels

    def compute_log_probability_numerator(self, doc):
        """
        Computes the numerator of log probability i.e. log(P(X/Y)P(Y)) for different class labels for the provided doc.
        :param doc:
        :return:
        """
        log_prob_pos = log(self.P_positive)  # log of positively labelled reviews
        log_prob_neg = log(self.P_negative)  # log of negatively labelled reviews
        for word_id in doc.nonzero()[1]:
            count = doc[0, word_id]
            log_prob_pos += count * log(self.prob_features_pos_label[word_id])
            log_prob_neg += count * log(self.prob_features_neg_label[word_id])

        return log_prob_pos, log_prob_neg

    def LogSum(self, logx, logy):
        # Return log(x+y), avoiding numerical underflow/overflow. Log-exp sum trick
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))

    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, X, indexes=None):
        """
        Computes the predicted probability (positive) of given docs.
        """
        if not indexes:
            indexes = range(X.shape[0])
        predicted_probs = []
        for i in indexes:
            # TO DO: Predict the probability of the i_th review in test being positive review
            # TO DO: Use the LogSum function to avoid underflow/overflow
            doc = X[i, :]
            log_prob_pos, log_prob_neg = self.compute_log_probability_numerator(doc)  # compute the log probability
            # numerator
            log_prob_deno = self.LogSum(log_prob_pos, log_prob_neg)  # compute the log denominator

            log_predicted_prob_positive = log_prob_pos - log_prob_deno
            predicted_prob_positive = exp(log_predicted_prob_positive)  # compute probability from log prob
            predicted_probs.append(predicted_prob_positive)

            '''
            # A change for Part (c) of the assignment: Precision vs Recall

            log_predicted_prob_neg = log_prob_neg - log_prob_deno
            predicted_prob_negative = exp(log_predicted_prob_neg)
            if log_prob_pos > log_prob_neg:
                predicted_label = 1.0
            else:
                predicted_label = -1.0
            
            #print test.Y[i], test.X_reviews[i]
            # TO DO: Comment the line above, and uncomment the line below
            print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])
            '''
        return predicted_probs

    # Evaluate performance on test data 
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()

    def EvalPrecision(self, test, threshold=0.5, class_label=1.0):
        """
        Computes precision for the given threshold
        """
        Y_pred = np.array(self.PredictLabel(test.X, threshold))
        true_pos_pred_indx = np.argwhere(Y_pred[Y_pred == test.Y] == class_label)
        return len(true_pos_pred_indx)/len(np.argwhere(Y_pred == class_label))

    def EvalRecall(self, test, threshold=0.5, class_label=1.0):
        """
        Computes recall for the given threshold
        """
        Y_pred = np.array(self.PredictLabel(test.X, threshold))
        true_pos_pred_indx = np.argwhere(Y_pred[Y_pred == test.Y] == class_label)
        return len(true_pos_pred_indx)/len(np.argwhere(test.Y == class_label))

    def EvalPrecisionAndRecall(self, test, threshold=0.5, class_label=1.0):
        """
        Computes precision and recall in one go and return a tuple of precision, recall
        """
        Y_pred = np.array(self.PredictLabel(test.X, threshold))
        true_pred_indx = np.argwhere(Y_pred == test.Y)
        true_ct = 0
        for index in true_pred_indx:
            if Y_pred[index] == class_label:
                true_ct += 1
        return (
            true_ct / len(np.argwhere(Y_pred == class_label)),
            true_ct / len(np.argwhere(test.Y == class_label))
        )


def plot_precision_recall_curve(precision_recall_arr, label=''):
    precision = []
    recall = []

    for item in precision_recall_arr:
        precision.append(item[0])
        recall.append(item[1])

    plt.figure()
    axs = plt.gca()
    axs.plot(recall, precision, 'o--')
    axs.set_title(f"Precision vs Recall {label} label")
    axs.set_xlabel("Recall")
    axs.set_ylabel("Precision")


if __name__ == "__main__":
    
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)    
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    print("Test Accuracy: ", nb.Eval(testdata))

    thresholds = np.arange(0.1, 1.0, 0.1)
    prec_rec_pos_label = []
    prec_rec_neg_label = []
    # Compute Precision and Recall
    for threshold in thresholds:
        prec_rec_pos_label.append(
            nb.EvalPrecisionAndRecall(testdata, threshold, 1.0)
        )

        prec_rec_neg_label.append(
            nb.EvalPrecisionAndRecall(testdata, threshold, -1.0)
        )
    plot_precision_recall_curve(prec_rec_pos_label, label='+ve')  # plot the Precision Recall curve for positive class
    plot_precision_recall_curve(prec_rec_neg_label, label='-ve')  # plot the Precision Recall curve for negative class
    plt.show()
    pos_influential_word_ids = np.argsort(nb.prob_features_pos_label)[::-1][:20]
    neg_influential_word_ids = np.argsort(nb.prob_features_neg_label)[::-1][:20]

    result_str = ""
    for id in pos_influential_word_ids:
        result_str = result_str + f"{nb.data.vocab.GetWord(id)}  {nb.prob_features_pos_label[id]}  "
    print(result_str)

    result_str = ""
    for id in neg_influential_word_ids:
        result_str = result_str + f"{nb.data.vocab.GetWord(id)}  {nb.prob_features_neg_label[id]}  "
    print(result_str)
