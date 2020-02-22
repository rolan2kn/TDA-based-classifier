#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import neighbors
import numpy as np

class kNNClassifier:
    def __init__(self, training, training_tags):
        self.training = training
        self.training_tags = training_tags
        self.new_data = []

    def execute(self, new_data):
        if new_data is None:
            return []
        n_neighbors = 15
        self.new_data = new_data

        # import some data to play with

        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(self.training, self.training_tags) # entrena el dataset con las etiquetas...

        result = clf.predict(np.array(new_data))

        return result
