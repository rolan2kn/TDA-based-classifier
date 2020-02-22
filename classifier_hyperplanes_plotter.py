#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
from dataset_handler import DatasetHandler


class ClassifierHyperplanesPlotter:
    def __init__(self, y_pred, dataset):
        self.classes = dataset.tags_set
        self.y_pred = dataset.unify_tags()
        self.x_values = dataset.unify_dataset()

        self.class_no = {}
        for i, l in enumerate(self.classes):
            self.class_no.update({l:i})

    def execute(self):
        if self.x_values is None or self.y_pred is None:
            return

        # Create color maps for 3-class classification problem, as with iris
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        X = [[_a, _b] for _a, _b, _, _ in self.x_values]  # we only take the first two features. We could
        # avoid this ugly slicing by using a two-dim dataset
        X = np.array(X)
        x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
        y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = np.c_(self.y_pred)

        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        y = [self.class_no[l] for l in self.class_no]
        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        plt.xlabel('sepal length (cm)')
        plt.ylabel('sepal width (cm)')
        plt.axis('tight')

