#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import gudhi
import random
from sklearn.metrics.pairwise import euclidean_distances
import utils


class TDABCHyperParamEstimator:

    def __init__(self, data_file_name = None):
        path = utils.get_root_dir()
        self.data_file_name = data_file_name if data_file_name else "%s/dataset/iris.csv"%path
        self.simplex_tree = None
        self.dataset = []
        self.training = []
        self.test = []
        self.filtrations = None
        self.simplex_tree = None
        self.rips = None
        self.memory = None

    def save_size(self, filenameMem, k, n, r, q, mem_byte = None):

        mem_kb = mem_byte / 1024 if mem_byte else None
        mem_mb = mem_kb / 1024 if mem_byte else None
        mem_gb = mem_mb / 1024 if mem_byte else None

        memfile = open(filenameMem, "a")
        memfile.write("\nk={0},n={1},r={2},q={3},mem_bytes={4},mem_kb={5},mem_mb={6}, mem_gb={7}".format(k,
                                                                                                         n,
                                                                                                         r,
                                                                                                         q,
                                                                                                         mem_byte,
                                                                                                         mem_kb,
                                                                                                         mem_mb,
                                                                                                         mem_gb))
        print("\nk={0},n={1},r={2},q={3},mem_bytes={4},mem_kb={5},mem_mb={6}, mem_gb={7}".format(k,
                                                                                                 n,
                                                                                                 r,
                                                                                                 q,
                                                                                                 mem_byte,
                                                                                                 mem_kb,
                                                                                                 mem_mb,
                                                                                                 mem_gb))
        memfile.close()

    def get_object_size(self, filenameMem, k, n, r, q):
        print("\n#########: voy a calcular la memoria")

        self.memory = None
        self.memory = utils.get_obj_size(self)
        print ("memory size = {0}".format(self.memory))
        self.save_size(filenameMem, k, n, r, q, self.memory)

        print("\n#########: fin de calculo de memoria")

    def init_data_with_iris(self):
        if self.data_file_name.find("dataset/iris.csv") == -1:
            return

        if self.dataset:
            self.dataset.clear()
            del self.dataset
            self.dataset = []

        iris_csv = open(self.data_file_name)
        iris_csv.readline()  # to ignore headers

        for idx, line in enumerate(iris_csv.readlines()):
            # print(line)
            str_record = line.split(",")[:-1]
            float_record = [float(i) for i in str_record]

            self.dataset.append(float_record)

        iris_csv.close()

    def split_dataset(self, k = None, j = None):

        size = len(self.dataset)

        k = 5 if k is None else k
        j = random.randint(0, k) if j is None else j

        if self.training:
            self.training.clear()
            del self.training
            self.training = []

        if self.test:
            self.test.clear()
            del self.test
            self.test = []

        I = [i for i in range(size)]

        random.shuffle(I)

        for i in range(0, size, k):
            if i != j*k:
                for id in range(i, i+k):
                    self.training.append(self.dataset[I[id]])
            else:
                for id in range(i, i+k):
                    self.test.append(self.dataset[I[id]])

        # print("Training Data: ", self.training)
        # print("Test Data: ", self.test)

    def get_maximal_distance(self):
        distances = euclidean_distances(self.dataset, self.dataset)
        import numpy as np

        maxd = np.max(distances)

        return int(maxd+1)

    def build_vr_complex(self, filenameMem, k, n, r, q):
        S = []
        S.extend(self.training)
        S.extend(self.test)
        self.rips = gudhi.RipsComplex(points=S,
                                 max_edge_length=float(r))

        self.simplextree = self.rips.create_simplex_tree(max_dimension=q)
        # self.filtrations = self.simplex_tree.get_filtration()

        # resp2 = utils.exec_with_timeout(self.get_object_size, [filenameMem, k, n, r, q], 240)
        # if not resp2:
        #     self.save_size(filenameMem, None, k, n, r, q)

    def destroy(self):

        if self.filtrations:
            del self.filtrations
            self.filtrations = None
        if self.simplex_tree:
            del self.simplex_tree
        self.simplex_tree = None
        if self.rips:
            del self.rips
        self.rips = None

    def execute(self):
        self.init_data_with_iris()
        _D = self.get_maximal_distance()
        _Q0 = 1#int(len(self.dataset)/2)
        _Q = len(self.dataset)

        _now = time.strftime("%Y.%m.%d__%H.%M.%S")
        filename = "%s/results/hyper_params_%s.txt" % (utils.get_root_dir(), _now)
        filenameMem = "%s/results/hyper_params_MEMORIA_%s.txt" % (utils.get_root_dir(), _now)
        result_file = open(filename, "w")
        result_file_mem = open(filenameMem, "w")
        result_file_mem.close()

        result_file.write("\nself.rips = gudhi.RipsComplex(points=self.training,max_edge_length=r)\n\n")

        result_file.write("self.simplex_tree = self.rips.create_simplex_tree(max_dimension=q)")
        timeout = 24000
        for k in [5]:
            for n in [0]:
                self.split_dataset(k, n)
                result_file.write("\nCROSS VALIDATION VALUES k=%s, n=%s," % (k, n))
                for r in range(1, _D):
                    for q in range(_Q0, _Q):
                        try:
                            result_file.write("\nk=%s, n=%s, r=%s, q=%s," % (k, n, r, q))
                            t1 = time.time()
                            self.build_vr_complex(filenameMem, k, n, r, q)
                            # resp = utils.exec_with_timeout(self.build_vr_complex, [filenameMem, k, n, r, q], timeout*3)
                            t2 = time.time()

                            self.destroy()
                            # t = (t2 - t1) if resp else timeout
                            print("\nk=%s, n=%s, r=%s, q=%s, timing=%s seg" % (k, n, r, q, t2 - t1))
                        except BaseException as e:
                            t = None
                            m = None
                            wd = None
                            print("Error jejeje: {0}".format(e))

                        # result_file.write("timing=%s" % (t))

                        result_file.flush()

        result_file.close()

    def execute_with_params(self, r, q):
        self.init_data_with_iris()

        _now = time.strftime("%Y.%m.%d__%H.%M.%S")
        filename = "%s/results/hyper_params_%s.txt" % (utils.get_root_dir(), _now)
        filenameMem = "%s/results/hyper_params_MEMORIA_%s.txt" % (utils.get_root_dir(), _now)
        result_file = open(filename, "w")
        result_file_mem = open(filenameMem, "w")
        result_file_mem.close()

        result_file.write("\nself.rips = gudhi.RipsComplex(points=self.training,max_edge_length=r)\n\n")

        result_file.write("self.simplex_tree = self.rips.create_simplex_tree(max_dimension=q)")

        for k in [5, 10, 15]:
            for n in range(k):
                self.split_dataset(k, n)
                result_file.write("\nCROSS VALIDATION VALUES k=%s, n=%s," % (k, n))
                result_file.write("\nk=%s, q=%s, r=%s, q=%s,"%(k, n, r, q))
                try:
                    result_file.write("\nk=%s, q=%s, r=%s, q=%s," % (k, n, r, q))
                    t1 = time.time()
                    # self.build_vr_complex(r, q)
                    resp = utils.exec_with_timeout(self.build_vr_complex, [r, q], 120)
                    t2 = time.time()
                    if k == 5 and n == 0:
                        resp2 = utils.exec_with_timeout(self.get_object_size, [filenameMem, k, n, r, q], 120)
                        if not resp2:
                            self.save_size(filenameMem, None, k, n, r, q)
                    self.destroy()
                    t = (t2 - t1) if resp else None
                except BaseException as e:
                    t = None
                    m = None
                    wd = None
                    print ("Error jejeje: {0}".format(e))

                result_file.write("timing=%s" % (t))
                print("\nk=%s, n=%s, r=%s, q=%s, timing=%s seg" % (k, n, r, q, t))
                result_file.flush()
                break
        result_file.close()

