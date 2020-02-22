#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

'''
class EvaluationMetricsHandler main goal is to compute metrics for each class
from expected and predicted results, and a list of classes
'''
class EvaluationMetricsHandler:

    """
    :method __init__
    :argument expected_results are the real labels of each value x \in X, the training set
    :argument predicted_results are the classification resulting labels for each x \in X
    :argument classes is the label list
    """
    def __init__(self, expected_results, predicted_results, classes):
        self.expected_results = expected_results
        self.predicted_results = predicted_results
        self.classes = classes

    """
    :method init_values initialize all values to compute metrics   
    """
    def init_values(self):
        self.precision = 0.0  # initialize all global metrics
        self.recall = 0.0
        self.agree_rate = 0.0
        self.fp_rate = 0.0
        self.f1_measure = 0.0
        self.mse = 0.0

        self.precision_by_class = {}    # create all per class metrics
        self.recall_by_class = {}
        self.agree_rate_by_class = {}
        self.fp_rate_by_class = {}
        self.f1_measure_by_class = {}
        self.mse_by_class = {}
        self.class_no = {}

        for i, c in enumerate(self.classes):    # initialize every metrics per class
            self.class_no.update({c: i+1})
            self.precision_by_class.update({c: 0})
            self.recall_by_class.update({c: 0})
            self.agree_rate_by_class.update({c: 0})
            self.fp_rate_by_class.update({c: 0})
            self.f1_measure_by_class.update({c: 0})
            self.mse_by_class.update({c: 0})

    """
    :method: compute_metrics_per_label  
    
    We use True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN) per each class where: 
    
    for each label l \in classes do 
        TP: predicted_list(i) iff predicted_list(i) == real_list(i) and predicted_list(i) == l
        FP: predicted_list(i) iff predicted_list(i) != real_list(i) and predicted_list(i) == l
        
        TN: predicted_list(i) iff predicted_list(i) == real_list(i) and predicted_list(i) != l
        FN: predicted_list(i) iff predicted_list(i) != real_list(i) and predicted_list(i) != l
                    
    precision_by_class = TP / (TP + FP)  
    recall_by_class = TP / (TP + FN)
    agree_rate_by_class = (TP + TN) / (TP + TN + FP + FN) 
    fp_rate_by_class = FP / (TP + FN)
    f1_measure_by_class = 2 * ((precision_by_class * recall_by_class) / (precision_by_class + recall_by_class))
    mse_by_class = values_per_label(predicted_list(i)) - values_per_label(real_list(i)) ^ 2
    
    """
    def compute_metrics_per_label(self):
        true_p = {}  # first we initialize metrics base values TP, FP, FN, TN
        false_p = {}
        false_neg = {}
        true_neg = {}
        n_by_class = {}  # this means number of samples per class

        self.init_values()  # we initialize all metric values
        class_no = self.class_no  #

        n = 0  # we initialize the total number of samples
        for i, c in enumerate(self.classes):
            true_p.update({c: 0})  # initialize basic metric values per class
            false_p.update({c: 0})
            true_neg.update({c: 0})
            false_neg.update({c: 0})
            n_by_class.update({c: 0})

            for idx, value in enumerate(self.predicted_results):  # for any predicted result

                error = (class_no[self.predicted_results[idx]] - class_no[
                    self.expected_results[idx]])  # we compute the error

                if value == c:  # positive cases
                    if value == self.expected_results[idx]:
                        true_p[c] += 1
                    else:
                        false_p[c] += 1
                    self.mse_by_class[c] += error * error
                    n_by_class[c] += 1
                else:  # negatives cases
                    if value == self.expected_results[idx]:
                        true_neg[c] += 1
                    else:
                        false_neg[c] += 1

            div = true_p[c] + true_neg[c] + false_p[c] + false_neg[c]
            div = 1 if div == 0 else div
            self.agree_rate_by_class[c] = (true_p[c] + true_neg[c]) / div

            div = (true_p[c] + false_p[c])
            div = 1 if div == 0 else div
            self.precision_by_class[c] += (true_p[c]) / div

            div = (true_p[c] + false_neg[c])
            div = 1 if div == 0 else div
            self.recall_by_class[c] = (true_p[c]) / div

            div = (true_p[c] + false_neg[c])
            div = 1 if div == 0 else div
            self.fp_rate_by_class[c] = (false_p[c]) / div

            div = self.precision_by_class[c] + self.recall_by_class[c]
            if div == 0.0:
                div = 1
            self.f1_measure_by_class[c] += 2 * ((self.precision_by_class[c] * self.recall_by_class[c]) / (div))

            n += n_by_class[c]

        return n

    """
    :method compute_metrics this compute all gobal metrics taking in count all per-label metrics
       
    Global metrics are computed by the average of each per-label metrics
    """
    def compute_metrics(self):
        n = self.compute_metrics_per_label()        # we compute every metric per label and returns total number of samples

        for c in self.classes:
            self.agree_rate += self.agree_rate_by_class[c]
            self.precision += self.precision_by_class[c]
            self.recall += self.recall_by_class[c]
            self.fp_rate += self.fp_rate_by_class[c]
            self.mse += self.mse_by_class[c]
            self.f1_measure += self.f1_measure_by_class[c]

        num_classes = len(self.classes)
        if not num_classes:
            num_classes = 1

        self.agree_rate /= num_classes
        self.precision /= num_classes
        self.recall /= num_classes
        self.fp_rate /= num_classes
        self.f1_measure /= num_classes
        self.mse /= n

        for l in self.classes:
            print ("\n\n**--*-*-*-*-*-*-*-*-*-*-*\n agree_rate: {0}\nprecision: {1}\nrecall: {2}\nfp_rate: {3}\nmse: {4}\nf1_measure: {5}".format(self.agree_rate, self.precision, self.recall, self.fp_rate, self.mse, self.f1_measure))
            print("\n\n**{6}: --*-*-*-*-*-*-*-*-*-*-*\n agree_rate: {0}\nprecision: {1}\nrecall: {2}\nfp_rate: {3}\nmse: {4}\nf1_measure: {5}".format(
                    self.agree_rate_by_class[l],
                    self.precision_by_class[l],
                    self.recall_by_class[l],
                    self.fp_rate_by_class[l],
                    self.mse_by_class[l],
                    self.f1_measure_by_class[l], l))

    def save_to_file(self, fmetrics):
        fmetrics.write("\n\n**--*-*-*-*-*-*-*-*-*-*-*\n agree_rate: {0}\nprecision: {1}\nrecall: {2}\nfp_rate: {3}\nmse: {4}\nf1_measure: {5}".format(
                self.agree_rate, self.precision, self.recall, self.fp_rate, self.mse, self.f1_measure))
        for l in self.classes:
            fmetrics.write ("\n\n**--*-*-*-*-*-*-*-*-*-*-*\n agree_rate: {0}\nprecision: {1}\nrecall: {2}\nfp_rate: {3}\nmse: {4}\nf1_measure: {5}".format(self.agree_rate, self.precision, self.recall, self.fp_rate, self.mse, self.f1_measure))
            fmetrics.write("\n\n**{6}: --*-*-*-*-*-*-*-*-*-*-*\n agree_rate: {0}\nprecision: {1}\nrecall: {2}\nfp_rate: {3}\nmse: {4}\nf1_measure: {5}".format(
                    self.agree_rate_by_class[l],
                    self.precision_by_class[l],
                    self.recall_by_class[l],
                    self.fp_rate_by_class[l],
                    self.mse_by_class[l],
                    self.f1_measure_by_class[l], l))
        fmetrics.flush()