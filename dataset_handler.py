#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import utils
import random
import time as time
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll


IRIS, SWISSROLL, DAILY_AND_SPORTS, LIGHT_CURVES = range(4)


class DatasetHandler:
    def __init__(self, dataset_type = IRIS, dimensions = 2):
        self.dataset_type = dataset_type                    # dataset
        self.dataset = []                                   # dataset points
        self.training = []                                  # training set S
        self.test = []                                      # testing set  X
        self.tags_set = set()                               # tag set a class list L
        self.labels = list()                                # list of class names
        self.tags = []                                      # tags list for access purposes
        self.tags_training = {}                             # association set T = {(s, l) | s \in S; l \in L}
        self.tags_test = {}                                 # incomplete association set T = {(x, l) | x \in X; l \in L}
        self.tags_position = {}
        self.dimensions = dimensions

    def load_dataset(self):
        if self.dataset:
            self.dataset.clear()
            del self.dataset
            self.dataset = []

        if self.dataset_type == IRIS:
            self.load_iris()
        elif self.dataset_type == SWISSROLL:
            self.load_swiss_roll()
        elif self.dataset_type == DAILY_AND_SPORTS:
            self.load_daily_and_sport_activities()
        elif self.dataset_type == LIGHT_CURVES:
            self.load_light_curves()
        else:
            self.from_csv_file("{0}/dataset/iris.csv".format(utils.get_module_path()))

        self.assign_tags()

    def load_iris(self):
        iris = datasets.load_iris()
        dim = iris.data.shape[1]
        _min = min(self.dimensions, dim)

        self.dataset = [[sample[d] for d in range(_min)] for sample in iris.data]
        # self.dataset = iris.data
        self.tags = iris.target
        self.labels = list(iris.target_names)
        self.tags_set = set(self.tags)

    def load_swiss_roll(self):
        n_samples = 1500
        noise = 0.05
        X, _ = make_swiss_roll(n_samples, noise)
        # Make it thinner
        X[:, 1] *= .5

        dim = X.shape[1]
        if dim > 3:
            _min = min(self.dimensions, dim)

            self.dataset = [[sample[d] for d in range(_min)] for sample in X]
        else:
            self.dataset = X

        ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
        self.tags = ward.labels_
        self.tags_set = set(self.tags)

    def load_daily_and_sport_activities(self):
        pass

    def load_light_curves(self):
        pass

    def from_csv_file(self, file_name):
        if not os.path.exists(file_name):
            return

        iris_csv = open(self.data_file_name)
        iris_csv.readline()  # to ignore headers

        for idx, line in enumerate(iris_csv.readlines()):
            # print(line)
            str_record = line.split(",")[:-1]
            float_record = [float(i) for i in str_record]

            self.dataset.append(float_record)
            self.tags_set.add(line.split(",")[-1])
            self.tags.append(line.split(",")[-1])

        iris_csv.close()

    def configure_external_testing_set(self, ext_set):
        self.test.clear()
        self.tags_test.clear()

        size = len(self.dataset)
        tcount = size - 1  # define the first element to classify

        size_2_clsfy = len(ext_set)
        for i in range(size_2_clsfy):  # we iterate the new testing set
            tcount += 1

            self.test.append([tcount, ext_set[i]])  # filling testing set

    def split_dataset(self, k=None, fold_position=None):
        self.clean()
        size = len(self.dataset)

        external_test = False

        if size == 0:  # initialize values
            return
        if k is None:
            external_test = True
        elif fold_position is None or (fold_position*k) > (size-1):
            value = int((size + k-1)/k)
            fold_position = random.randint(0, value-1)

        I = [i for i in range(size)]  # dataset-samples index list

        random.seed(time.perf_counter())  # make the index list distorted
        random.shuffle(I)

        count = -1

        if not external_test:
            tcount = size - k - 1  # and kfold count
            for i in range(0, size, k):  # we iterate the entire dataset by making kfold steps
                if i != fold_position * k:  # is current sample is outside the desired fold
                    for id in range(i, i + k):  # then we fill the training set, and we also associate tags to it
                        if id < size:
                            count += 1
                            self.training.append(self.dataset[I[id]])  # filling the training set
                            self.tags_training.update({str([count]): self.tags[I[id]]})  # associating tags
                else:
                    for id in range(i, i + k):  # but if we are in the desired fold
                        if id < size:  # we fill the testing set and we associate its tags
                            tcount += 1

                            self.test.append([tcount, self.dataset[I[id]]])  # filling testing set
                            self.tags_test.update({str([tcount]): self.tags[I[id]]})  # associating tags
        else:
            for i in range(size):  # we iterate the entire dataset normally
                count += 1
                self.training.append(self.dataset[I[i]])  # filling the training set
                self.tags_training.update({str([count]): self.tags[I[i]]})  # associating tags

    def assign_tags(self):
        for i, t in enumerate(self.tags_set):
            self.tags_position.update({t: i})

    def clean(self):
        if self.training:
            self.training.clear()
            del self.training
            self.training = []

        if self.test:
            self.test.clear()
            del self.test
            self.test = []

        if self.tags_training:
            self.tags_training.clear()
            del self.tags_training
            self.tags_training = {}

        if self.tags_test:
            self.tags_test.clear()
            del self.tags_test
            self.tags_test = {}

    def unify_dataset(self):
        S = []
        S.extend(self.training)
        for _, x in self.test:
            S.append(x)

        return S

    def unify_tags(self):
        size = len(self.dataset)
        all_tags = []
        for idx in range(size):
            key = "[{0}]".format(idx)
            if key in self.tags_training:
                all_tags.append(self.tags_training[key])
            else:
                all_tags.append(None)

        return all_tags

    def is_dataset(self, dataset_type):
        return self.dataset_type == dataset_type


class DailyAndSportsActivitiesController:
    def __init__(self):
        pass

    def load(self, filename="/mnt/D/PHD/Research/Datasets/Swiss Roll/preswissroll.dat"):
        X = []
        pre_swissroll_file = open(filename)
        for line in pre_swissroll_file.readlines():
            parts = line.split(' ')
            i = 0
            x = None
            y = None
            while y is None and i < len(parts):
                if len(parts[i]) > 0:
                    if x is not None:
                        y = float(parts[i])
                    else:
                        x = float(parts[i])
                i = i + 1

            if y is not None:
                X.append([x, y])

        return X
