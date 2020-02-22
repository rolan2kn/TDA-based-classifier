#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import utils
import random
import time as time
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll
from dataset_handler import DatasetHandler, IRIS, SWISSROLL, DAILY_AND_SPORTS, LIGHT_CURVES

from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification


class DatasetPlotter:
    def __init__(self, data_mgr = None):
        self.data = data_mgr \

        if data_mgr is None:
            self.data = DatasetHandler(IRIS)
            self.data.load_dataset()

    def draw_data(self):
        if self.data.is_dataset(IRIS):
            self.draw_iris()
        elif self.data.is_dataset(SWISSROLL):
            self.draw_swiss_roll()
        else:
            self.draw_iris()

    def draw_iris(self):
        data_A_sample = self.data.unify_dataset()

        fig = plt.figure()
        fig.set_size_inches(10, 8)
        ax = fig.add_subplot(111)

        tag = None

        ks = list(self.data.tags_set)

        points = {ks[0]: [[], []]}
        points.update({ks[1]: [[], []]})
        points.update({ks[2]: [[], []]})

        for i in self.data.tags_training:
            idx = int(i[1:-1])
            k = self.data.tags_training[i]

            points[k][0].append(data_A_sample[idx][0])
            points[k][1].append(data_A_sample[idx][1])

        for i in self.data.tags_test:
            idx = int(i[1:-1])
            k = self.data.tags_test[i]

            points[k][0].append(data_A_sample[idx][0])
            points[k][1].append(data_A_sample[idx][1])

        area = (15) ** 2
        for idx, c in enumerate(['r', 'b', 'g']):
            values = points[ks[idx]]

            l = self.data.labels[ks[idx]].strip()
            if l.find("setosa") != -1:
                l = "Setosa"
            elif l.find("versicolor") != -1:
                l = "Versicolor"
            elif l.find("virginica") != -1:
                l = "Virginica"

            ax.scatter(values[0], values[1], s=area, c=c, marker="o", label=l)

        ax.set_xlabel('Sepal length', size=15)
        ax.set_ylabel('Sepal width', size=15)
        ax.legend(fontsize=20)
        plt.savefig('DATA_GRAPHICS/iris.png')

    def draw_swiss_roll(self):
        fig = plt.figure()
        fig.set_size_inches(10, 8)
        ax = p3.Axes3D(fig)
        ax.view_init(7, -80)
        label = self.data.tags
        X = self.data.dataset
        for l in np.unique(label):
            ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
                       color=plt.cm.jet(np.float(l) / np.max(label + 1)),
                       s=20, edgecolor='k')
        plt.title('Swiss Roll')
        plt.savefig('DATA_GRAPHICS/swissroll.png')
        plt.show()

    def draw_hyperplanes(self, classifiers, names, scores):
        h = .02  # step size in the mesh

        #names = ["Nearest Neighbors", "TDA-Based Classifier (TDABC)"]

        figure = plt.figure(figsize=(27, 9))
        i = 1
        # iterate over datasets
        X = np.array(self.data.dataset)
        X_train = np.array(self.data.training)
        X_test = np.array(self.data.test)
        y_train = [self.data.tags_position[self.data.tags_training[i]] for i in self.data.tags_training]
        y_test = None
        if len(self.data.tags_test) > 0:
            y_test = [self.data.tags_position[self.data.tags_test[i]] for i in self.data.tags_test]

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(1, len(classifiers) + 1, i)
        ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        if y_test is not None:
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf, score in zip(names, classifiers, scores):
            ax = plt.subplot(1, len(classifiers) + 1, i)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            Z = np.array(clf)

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            if score is not None:
                ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                        size=15, horizontalalignment='right')

        plt.tight_layout()
        plt.show()
