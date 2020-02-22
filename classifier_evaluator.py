#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import os

import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix

import utils

from evaluation_metrics_handler import EvaluationMetricsHandler
from classifier_hyperplanes_plotter import ClassifierHyperplanesPlotter


class ClassifierEvaluator:
    def __init__(self, method_name = None, classes = None):

        self.method_name = method_name
        self.selection_type = ""
        if not method_name is None:
            self.selection_type = method_name.split("_")[-1]
        self.metrics_list = []
        self.classes = classes

    def add_metrics(self, expected_results, predicted_results):
        if not expected_results or not predicted_results:
            return

        metrics = EvaluationMetricsHandler(expected_results, predicted_results, self.classes)
        metrics.compute_metrics()
        self.metrics_list.append(metrics)

    def save_metrics(self):
        path = "{0}/docs/CLASSIFIER_EVALUATION/".format(utils.get_module_path())
        file_name = time.strftime(
            "{0}_{1}_{2}_%y.%m.%d__%H.%M.%S.txt".format(path, self.method_name, "metrics"))

        fmetrics = open(file_name, "w")
        for idx, metric in enumerate(self.metrics_list):
            metric.save_to_file(fmetrics)
        fmetrics.close()

    def get_values(self):
        y_true = []
        y_pred = []

        for metrics in self.metrics_list:
            y_true.extend(metrics.expected_results)
            y_pred.extend(metrics.predicted_results)

        return y_true, y_pred

    def plot_generalized_confusion_matrix(self, normalize = False, cmap = plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        the generalized confusion matrix compute the full predicted results over full real results
        """
        y_true, y_pred = self.get_values()

        if y_true is None or y_pred is None:
            return
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Only use the labels that appear in the data
        classes = list(self.classes)           # classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        self.save_picture(title)

        return ax

    def plot_roc_curve(self):
        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        labels = ["False Positive Rate", "True Positive Rate"]
        fig, ax = plt.subplots()

        colors = ["b", "r", "g"]
        markers = ["", "", ""]
        lines = [(0, ()), (0, (1, 1)), (0, (3, 1, 1, 1))]

        for i, c in enumerate(self.classes):
            Xs = []
            Ys = []
            for metric in self.metrics_list:
                Xs.append(metric.fp_rate_by_class[c])
                Ys.append(metric.recall_by_class[c])

            ax.plot(Xs, Ys, label="ROC Curve Class {0}".format(c.strip()), color=colors[i], marker=markers[i], linestyle=lines[i], linewidth=1.5)
        fig.suptitle('ROC Curve')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.legend()

        self.save_picture("ROC_Curve")

        return ax

    def plot_pr_curve(self):
        # Compute macro-average PR curve

        # First aggregate all false positive rates
        labels = ["Recall", "Precision"]
        fig, ax = plt.subplots()

        Xs = []
        Ys = []
        for metric in self.metrics_list:
            Xs.append(metric.recall)
            Ys.append(metric.precision)

        ax.plot(Xs, Ys)
        fig.suptitle('PR Curve')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.legend()

        self.save_picture("PR_Curve")

        return self.plot_2D_curve(Xs, Ys, labels, "PR Curve", "k", "*")

    def agree_rate_curve(self):
        Xs = list(range(1, 1+len(self.metrics_list)))
        Ys = [metric.agree_rate for metric in self.metrics_list]
        labels = ["Trials","Agree Rate"]

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,title="agree_rate_curve", color="y", marker="p")

    def precision_curve(self):
        Xs = list(range(1, 1+len(self.metrics_list)))
        Ys = [metric.precision for metric in self.metrics_list]
        labels = ["Trials", "Precision"]

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,title="precision_curve", color="c", marker="o")

    def recall_curve(self):
        Xs = list(range(1, 1+len(self.metrics_list)))
        Ys = [metric.recall for metric in self.metrics_list]
        labels = ["Trials", "Recall"]

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,title="recall_curve", color="g", marker=">")

    def fp_rate_curve(self):
        Xs = list(range(1, 1+len(self.metrics_list)))
        Ys = [metric.fp_rate for metric in self.metrics_list]
        labels = ["Trials", "FP Rate"]

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,title="fp_rate_curve", color="r", marker="<")

    def f1_measure_curve(self):
        Xs = list(range(1, 1+len(self.metrics_list)))
        Ys = [metric.f1_measure for metric in self.metrics_list]
        labels = ["Trials", "F1 Measure"]

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,title="f1_measure_curve", color="b", marker="o")

    def mse_curve(self):
        Xs = list(range(1, 1+len(self.metrics_list)))
        Ys = [metric.mse for metric in self.metrics_list]
        labels = ["Trials", "Mean Squared Error"]

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels, title="mse_curve", color="m", marker="^")

    def plot_2D_curve(self, Xs, Ys, labels, title, color, marker):

        fig, ax = plt.subplots()

        ax.plot(Xs, Ys, color=color,marker=marker, linewidth=2, label=title)
        fig.suptitle(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.legend()

        self.save_picture(title)

        return ax

    def plot_all(self, x_values = None):
        # self.agree_rate_curve()
        # self.precision_curve()
        # self.recall_curve()
        # self.fp_rate_curve()
        # self.f1_measure_curve()
        # self.mse_curve()
        # self.plot_roc_curve()
        # self.plot_pr_curve()
        # self.plot_generalized_confusion_matrix()
        self.plot_generalized_confusion_matrix(normalize=True)
        self.plot_hyperplanes(x_values)
        self.save_metrics()

    def save_picture(self, title):
        path = "{0}/docs/CLASSIFIER_EVALUATION/{1}/".format(utils.get_module_path(), self.selection_type)
        if not os.path.isdir(path):
            os.makedirs(path)
        file_name = time.strftime(
            "{0}_{1}_{2}_%y.%m.%d__%H.%M.%S.png".format(path, self.method_name, title))

        plt.title(title)
        plt.savefig(file_name)

    def plot_hyperplanes(self, x_values):
        if x_values is None:
            return

        y_real, y_pred = self.get_values()
        chp = ClassifierHyperplanesPlotter(x_values, y_pred, y_real, self.classes)
        #chp.execute()
        #self.save_picture("Decision Boundaries")
