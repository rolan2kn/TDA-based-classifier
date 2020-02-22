#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import math
import gudhi
import random

import numpy as np

import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score


from knn_classifier import kNNClassifier
from classifier_evaluator import ClassifierEvaluator
from dataset_handler import DatasetHandler, IRIS, SWISSROLL, DAILY_AND_SPORTS, LIGHT_CURVES
from classifier_hyperplanes_plotter import ClassifierHyperplanesPlotter

RANDOMIZED, MAXIMAL, AVERAGE = range(3)
INCREMENTAL, DIRECT = range(2)


class TDABasedClassifier:

    def __init__(self, data_file_name=None, dataset_type = IRIS, algorithm_mode = DIRECT):
        self.data_file_name = data_file_name if data_file_name else "dataset/iris.csv"
        self.simplex_tree = None
        self.algorithm_mode = algorithm_mode
        self.dataset_handler = DatasetHandler(dataset_type, 4)
        self.classifier_evaluator = None
        self.filtrations = None
        self.simplex_tree = None
        self.complex = None
        self.memory = None

    def init_data(self):
        self.dataset_handler.load_dataset()

    def split_dataset(self, k=None, j=None):
        self.dataset_handler.split_dataset(k, fold_position=j)

    def unify_dataset(self):
        return self.dataset_handler.unify_dataset()

    def destroy(self):
        if self.filtrations:
            del self.filtrations
            self.filtrations = None
        if self.simplex_tree:
            del self.simplex_tree
        self.simplex_tree = None
        if self.complex:
            del self.complex
        self.complex = None

        self.dataset_handler.clean()

    '''
    get_link calcula el link(sigma) debido a que gudhi no computa esta funcion
    '''
    def get_link(self, sigma):
        """
        as gudhi SimplexTree dont have link method
        :param sigma:
        :return:
        """

        if self.simplex_tree is None:
            return set()

        link = set()

        if not (type(sigma) == list or type(sigma) == tuple):
            sigma = [sigma]
        try:
            size = len(sigma)
            _star = self.simplex_tree.get_star(sigma)

            for simplex, _ in _star:                # _ is the filtration value, its not necessary here
                # if len(sigma)-size == 1:
                simplex = set(simplex).difference(sigma)
                link = link.union(simplex)

            del _star
        except BaseException as e:
            print("ERROR en get_lik: {0}".format(e))

        print ("link({0}) = {1}".format(sigma, link))
        return link

    '''
    Psi es la funcion de asignacion que hace corresponder un conjunto de etiquetas t \in P(T) a cada simplice sigma \in K
    '''
    def Psi(self, sigma):
        if sigma is None:
            return []
        if not type(sigma) == list or not type(sigma) == tuple:
            sigma_key = str([sigma])
        else:
            sigma_key = str(sigma)

        if sigma_key in self.dataset_handler.tags_training:
            t = self.dataset_handler.tags_training[sigma_key]
            return t if type(t) in [list, tuple, dict, np.ndarray] else [t] # then t \neq None this may occure when ksimplex \in S,
            # or the computation was completed before

        card = self.Card(sigma)  # here we need to compute associations
        self.dataset_handler.tags_training.update({sigma_key: []})

        result = []
        if card == 1:  # then ksimplex \in X and t = None
            link = self.get_link(sigma)

            for tau in link:
                psi_val = self.Psi(tau)
                result.extend(psi_val)
        else:
            for tau in sigma:
                psi_val = self.Psi(tau)
                result.extend(psi_val)

        self.dataset_handler.tags_training.update({sigma_key: result})
        return result

    def Card(self, sigma):
        return len(sigma) if type(sigma) == list or type(sigma) == tuple else 1
    '''
    La funcion Gamma retorna un vector V, donde cada elemento 
    v_i \in V representa la cantidad de apariciones (o votos) obtenidos por la etiqueta 
    t_i \in T durante el calculo de Psi(\sigma).  
    '''
    def Gamma(self, sigma):
        card = self.Card(sigma)

        size_tags = len(self.dataset_handler.tags_set)
        V = [0]*size_tags
        if card == 1:
            _tags = self.Psi(sigma)
            for t in _tags: # como Psi(sigma) devuelve un set lo expando.
                _idx = self.G2(t)
                if _idx > -1:
                    V[_idx] += 1

        elif card > 1:
            for tau in sigma:
                V = list(map(sum, zip(V, self.Gamma(tau))))

        return V

    #Upsilon asigna a sigma la etiqueta con mayor cantidad de votos
    def Upsilon(self, sigma):
        V = self.Gamma(sigma)
        i = self.M(V)

        return self.G(i)

    # G es una funcion que dado un entero i devuelve la etiqueta
    # que ocupa la posicion i asumiento algun orden lexicografico sobre T
    def G(self, idx):
        if idx is None or idx >= len(self.dataset_handler.tags_set) or idx < 0:
            return None

        '''
        Naive code:
        
        for _idx, t in enumerate(self.tags_set):
            if idx == _idx:
                return t
        But if we convert the set in a list we can index it and return
        '''

        return list(self.dataset_handler.tags_set)[idx]

    def G2(self, tag):
        if tag not in self.dataset_handler.tags_position:
            return -1

        return self.dataset_handler.tags_position[tag]

    # M es una función que dado un vector V ∈ R^{|T|} devuelve un entero 0 <= i <= |T|,
    # donde i es la posicion de la componente de V con valor máximo
    def M(self, vector):
        size = len(vector)
        if size < 1:
            return 0

        major = vector[0]
        pos = 0
        for idx, element in enumerate(vector):
            if major < element:
                pos = idx
                major = element

        del major
        return pos

    # I es una función que dado una condicion retorna 1 si es verdadera y cero en otro caso
    def I(self, condition):
        return 1 if condition else 0

    def build_filtered_simplicial_complex(self):
        S = self.unify_dataset()

        # self.complex = gudhi.AlphaComplex(points=S)
        self.complex = gudhi.RipsComplex(points=S, max_edge_length=8.0)

        self.simplex_tree = self.complex.create_simplex_tree(max_dimension=3)
        # self.simplex_tree = self.complex.create_simplex_tree(max_alpha_square=2)
        # self.simplex_tree = self.complex.create_simplex_tree(max_dimension=3)
        # del self.complex
        # self.complex = None

        # self.simplex_tree.initialize_filtration()

        # diag = self.simplex_tree.persistence()

        # return diag

    def get_desired_persistence_interval2(self, choice=MAXIMAL):
        dimension = self.simplex_tree.dimension()
        print("\nDIMENSION := {0}\n".format(dimension))
        dimension -= 1
        pintervals = []
        while len(pintervals) == 0 and dimension > -1:
            pintervals = self.simplex_tree.persistence_intervals_in_dimension(dimension)
            dimension -= 1

        # get maximal persistence filtration
        if len(pintervals) == 0:
            return None

        intervals_count = len(pintervals)
        if choice == MAXIMAL:
            major = pintervals[0][1] - pintervals[0][0]
            desired_pos = 0
            for idx, interv in enumerate(pintervals):
                i = interv[1] - interv[0]
                if major < i and not math.isinf(i):
                    major = i
                    desired_pos = idx

            print("el mayor es ", major)
        elif choice == RANDOMIZED:                  # get randomized persistence filtration
            desired_pos = random.randint(int(intervals_count/2), intervals_count-1) # to maximize posibilities
            # desired_pos = random.randint(int(intervals_count/2), intervals_count-1) # to maximize posibilities

            print("\nLa duracion de vida seleccionado aleatoriamente es {0}\n".format(pintervals[desired_pos][1]-pintervals[desired_pos][0]))
        else:                                       # get average persistence filtration
            Avg = 0
            for interv in pintervals:
                Avg += interv[1] - interv[0]
            Avg /= intervals_count
            desired_pos = 0
            min_d = math.fabs((pintervals[0][1] - pintervals[0][0]) - Avg)
            for idx, interv in enumerate(pintervals):
                i = math.fabs((interv[1] - interv[0]) - Avg)
                if min_d > i and not math.isinf(i):
                    min_d = i
                    desired_pos = idx

        print("el intervalo de persistencia elegido es ", pintervals[desired_pos])
        inter = pintervals[desired_pos]
        del pintervals
        return inter

    def get_desired_persistence_interval(self, choice=MAXIMAL):
        dimension = self.simplex_tree.dimension()
        print ("\nDIMENSION := {0}\n".format(dimension))
        dimension -= 1
        pintervals = []
        while len(pintervals) == 0 and dimension > -1:
            pintervals = self.simplex_tree.persistence_intervals_in_dimension(dimension)
            dimension -= 1

        # get maximal persistence filtration
        if len(pintervals) == 0:
            return None

        major = pintervals[0][1] - pintervals[0][0]         # compute the persistent-interval with maximal lifetime
        desired_pos = 0
        for idx, interv in enumerate(pintervals):
            i = interv[1] - interv[0]
            if major < i and not math.isinf(i):
                major = i
                desired_pos = idx

        print("el mayor es ", major)

        if choice == MAXIMAL:
            return pintervals[desired_pos]
        else:
            high_lifetimes_pi = []      # We seek for all persistent-intervals which birth is greater than the birth of the maximal persistent interval
            max_pi = pintervals[desired_pos]
            lifetime = max_pi[1] - max_pi[0]
            for idx, interv in enumerate(pintervals):
                if interv[0] >= max_pi[0] and lifetime < (interv[1]-interv[0])*1.5:
                    high_lifetimes_pi.append(interv)

            intervals_count = len(pintervals)
            init = 0
            if len(high_lifetimes_pi) == 1:
                high_lifetimes_pi = pintervals
                init = int(intervals_count / 2)

            intervals_count = len(high_lifetimes_pi)

            if choice == RANDOMIZED:                  # get randomized persistence filtration
                desired_pos = random.randint(init, intervals_count-1) # to maximize posibilities

                print("\nLa duracion de vida seleccionado aleatoriamente es {0}\n".format(high_lifetimes_pi[desired_pos][1]-high_lifetimes_pi[desired_pos][0]))
                return high_lifetimes_pi[desired_pos]
            else:                                       # get average persistence filtration
                Avg = 0
                for interv in high_lifetimes_pi:
                    Avg += interv[1] - interv[0]
                if intervals_count > 0:
                    Avg /= intervals_count
                else:
                    return None

                desired_pos = 0
                min_d = math.fabs((high_lifetimes_pi[0][1] - high_lifetimes_pi[0][0]) - Avg) # we get the first persistent-interval superior tu average
                for idx, interv in enumerate(pintervals):
                    i = math.fabs((interv[1] - interv[0]) - Avg)
                    if min_d > i and not math.isinf(i):
                        min_d = i
                        desired_pos = idx

        print("el intervalo de persistencia elegido es ", high_lifetimes_pi[desired_pos])

        # return pintervals, desired_pos
        return high_lifetimes_pi[desired_pos]

    def execute(self):

        self.init_data()
        persistence_selector = {RANDOMIZED: "RANDOMIZED", MAXIMAL: "MAXIMAL", AVERAGE: "AVERAGE"}

        all_data = []
        size_data = len(self.dataset_handler.dataset)

        for selector in persistence_selector:

            self.classifier_evaluator = ClassifierEvaluator("TDABC_{0}".format(persistence_selector[selector]), classes=self.dataset_handler.tags_set)
            self.knn_classifier_evaluator = ClassifierEvaluator("kNN_{0}".format(persistence_selector[selector]), classes=self.dataset_handler.tags_set)
            for k in [5, 10, 15, 20, 25]:
                print("\n#####################################")
                print("\n#####################################")
                print("\nEXECUTING REPEATED CROSS VALIDATION")
                folds = int((size_data + k-1)/k)
                for j in range(folds):
                    print ("\nEXECUTE K-FOLD k={0}, n={1}".format(k, j))
                    self.split_dataset(k, j)

                    diag = self.build_filtered_simplicial_complex() # to compute simplicial complex and filtrations
                    print ("persistence diagrams: ", diag)

                    persistence_interval = self.get_desired_persistence_interval(choice=selector)

                    if persistence_interval is None:       # we ignore the process
                        self.destroy()
                        print("we destroy all simlicial complex information because we couldnt find any persistent interval")
                        continue

                    self.simplex_tree.prune_above_filtration(persistence_interval[0])

                    predicted_values = []
                    real_values = []
                    elems = []
                    ttags = [self.dataset_handler.tags_position[self.dataset_handler.tags_training[i]] for i in self.dataset_handler.tags_training]
                    ttraining = [e for e in self.dataset_handler.training]

                    for idx, x0 in self.dataset_handler.test:
                        idx_key = str([idx])
                        value = self.Upsilon(idx)
                        elems.append(x0)

                        predicted_values.append(value)
                        real_values.append(self.dataset_handler.tags_test[idx_key])

                    acc = accuracy_score(real_values, predicted_values)*100

                    self.classifier_evaluator.add_metrics(real_values, predicted_values)

                    self.destroy()
                    knn = kNNClassifier(ttraining, ttags)
                    in_values = knn.execute(elems)
                    all_data.extend(elems)
                    acc_knn = "None"
                    if len(in_values) > 0:
                        predicted_values2 = [self.G(i) for i in in_values]
                        self.knn_classifier_evaluator.add_metrics(real_values, predicted_values2)
                        acc_knn = accuracy_score(real_values, predicted_values2) * 100

                    print("\nTDABC accuracy = {0}".format(acc))
                    print("\nKNN accuracy = {0}".format(acc_knn))

            self.classifier_evaluator.plot_all()
            self.knn_classifier_evaluator.plot_all()

        plt.show()

    def draw_simplex_tree(self):
        path = "./docs/SIMPLEX_TREES"
        file_name = time.strftime("./docs/SIMPLEX_TREES/simplex_tree_%y.%m.%d__%H.%M.%S.txt")
        if not os.path.exists(path):
            os.makedirs(path)

        simplex_tree_file = open(file_name, "w")

        filtrations = self.simplex_tree.get_filtration()

        fmt = "%s:(%s):%.2f"
        points = self.unify_dataset()
        for filtered_value in filtrations:
            qsimplex = str(filtered_value[0])
            filt = filtered_value[1]
            point = ""

            inner_simplex = qsimplex[1:-1]
            if inner_simplex.find(",") == -1:
                point = points[int(inner_simplex)]

            line = fmt % tuple((qsimplex, point, filt))
            print(line)
            simplex_tree_file.write(str(line)+"\n")

