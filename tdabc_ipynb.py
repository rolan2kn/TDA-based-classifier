#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import math
import gudhi
import random
import numpy as np
import time as time
from sklearn import datasets
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll
from knn_classifier import kNNClassifier

RANDOMIZED, MAXIMAL, AVERAGE = range(3)
IRIS, SWISS_ROLL = range(2)


class TDABasedClassifier4IPYNB:

    def __init__(self, dataset_name=None, dim_2_read=4):
        self.dataset_name = IRIS if dataset_name is None else dataset_name
        self.simplex_tree = None
        self.dataset = []
        self.training = []
        self.test = []
        self.tags_set = set()
        self.training_tags = dict()
        self.tags_training = {}
        self.tags_test = {}
        self.tags_position = {}
        self.complex = None
        self.dims = dim_2_read

    def load_data(self):
        if self.dataset:
            self.dataset.clear()
            del self.dataset
            self.dataset = []

        if self.dataset_name == IRIS:
            self.load_iris()
        if self.dataset_name == SWISS_ROLL:
            self.load_swissroll()
        else:
            self.load_iris()
        self.assign_tags()

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

    def load_iris(self, dim=4):
        iris = datasets.load_iris()
        self.dataset = [[sample[i] for i in range(self.dims)] for sample in iris.data]
        # self.dataset = iris.data
        self.tags = iris.target
        self.labels = list(iris.target_names)
        self.tags_set = set(self.tags)

    def load_swissroll(self):
        n_samples = 1500
        noise = 0.05
        X, _ = make_swiss_roll(n_samples, noise)
        # Make it thinner
        X[:, 1] *= .5
        self.dataset = X

        ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
        self.tags = ward.labels_
        self.tags_set = set(self.tags)

    def draw_data(self):
        if self.dataset_name == IRIS:
            self.draw_iris()
        elif self.dataset_name == SWISS_ROLL:
            self.draw_swiss_roll()
        else:
            self.draw_iris()

    def draw_iris(self):
        data_A_sample = self.unify_dataset()

        fig = plt.figure()
        fig.set_size_inches(10, 8)
        ax = fig.add_subplot(111)

        tag = None

        ks = list(self.tags_set)

        points = {ks[0]: [[], []]}
        points.update({ks[1]: [[], []]})
        points.update({ks[2]: [[], []]})

        for i in self.tags_training:
            idx = int(i[1:-1])
            k = self.tags_training[i]

            points[k][0].append(data_A_sample[idx][0])
            points[k][1].append(data_A_sample[idx][1])

        for i in self.tags_test:
            idx = int(i[1:-1])
            k = self.tags_test[i]

            points[k][0].append(data_A_sample[idx][0])
            points[k][1].append(data_A_sample[idx][1])

        area = (15) ** 2
        for idx, c in enumerate(['r', 'b', 'g']):
            values = points[ks[idx]]

            l = self.labels[ks[idx]].strip()
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
        plt.savefig('iris.png')

    def draw_swiss_roll(self):
        fig = plt.figure()
        fig.set_size_inches(10, 8)
        ax = p3.Axes3D(fig)
        ax.view_init(7, -80)
        label = self.tags
        X = self.dataset
        for l in np.unique(label):
            ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
                       color=plt.cm.jet(np.float(l) / np.max(label + 1)),
                       s=20, edgecolor='k')
        plt.title('Swiss Roll')
        plt.savefig('swissroll.png')
        plt.show()

    def unify_dataset(self):
        S = []
        S.extend(self.training)
        for _, x in self.test:
            S.append(x)

        return S

    def configure_external_test(self, ext_set):
        self.test.clear()
        self.tags_test.clear()

        size = len(self.dataset)
        tcount = size - 1  # define the first element to classify

        size_2_clsfy = len(ext_set)
        for i in range(size_2_clsfy):  # we iterate the new testing set
            tcount += 1

            self.test.append([tcount, ext_set[i]])  # filling testing set
            self.tags_test.update({str([tcount]): ""})  # imcomplete associating tags

    def split_dataset(self, k=None, fold_position=None):
        self.clean()
        size = len(self.dataset)

        external_test = False

        if size == 0:  # initialize values
            return
        if k is None:
            external_test = True
        elif fold_position is None:
            fold_position = random.randint(0, k)

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

    def destroy(self):
        if self.simplex_tree:
            del self.simplex_tree
        self.simplex_tree = None
        if self.complex:
            del self.complex
        self.complex = None

        self.clean()

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
            _star = self.simplex_tree.get_star(sigma)
            size = len(sigma)
            for simplex, _ in _star:  # _ is the filtration value, its not necessary here
                if len(simplex) - size == 1:
                    simplex = set(simplex).difference(sigma)
                    link = link.union(simplex)

            del _star
        except BaseException as e:
            print("ERROR en get_lik: {0}".format(e))

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

        if sigma_key in self.tags_training:
            t = self.tags_training[sigma_key]
            return t if type(t) in [list, tuple, dict, np.ndarray] else [
                t]  # then t \neq None this may occure when ksimplex \in S,
            # or the computation was completed before

        card = self.Card(sigma)  # here we need to compute associations
        self.tags_training.update({sigma_key: []})

        result = []
        if card == 1:  # then ksimplex \in X and t = None
            link = self.get_link(sigma)

            for tau in link:
                # if not tau in self.test:
                psi_val = self.Psi(tau)
                result.extend(psi_val)
        else:
            for tau in sigma:
                psi_val = self.Psi(tau)
                result.extend(psi_val)

        self.tags_training.update({sigma_key: result})
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

        size_tags = len(self.tags_set)
        V = [0] * size_tags
        if card == 1:
            _tags = self.Psi(sigma)
            for t in _tags:  # como Psi(sigma) devuelve un set lo expando.
                # tags[_idx] = t
                _idx = self.G2(t)
                if _idx > -1:
                    V[_idx] += 1

            # for idx in range(size_tags):

        elif card > 1:
            for tau in sigma:
                V = list(map(sum, zip(V, self.Gamma(tau))))

        return V

    # Upsilon asigna a sigma la etiqueta con mayor cantidad de votos
    def Upsilon(self, sigma):
        V = self.Gamma(sigma)
        i = self.M(V)

        return self.G(i)

    # G es una funcion que dado un entero i devuelve la etiqueta
    # que ocupa la posicion i asumiento algun orden lexicografico sobre T
    def G(self, idx):
        if idx is None or idx >= len(self.tags_set) or idx < 0:
            return None

        '''
        Naive code:

        for _idx, t in enumerate(self.tags_set):
            if idx == _idx:
                return t
        But if we convert the set in a list we can index it and return
        '''

        return list(self.tags_set)[idx]

    def G2(self, tag):
        if tag not in self.tags_position:
            return -1

        return self.tags_position[tag]

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
        self.complex = gudhi.RipsComplex(points=S, max_edge_length=5.0)

        self.simplex_tree = self.complex.create_simplex_tree(max_dimension=3.0)
        # self.simplex_tree = self.complex.create_simplex_tree(max_alpha_square=36)
        # self.simplex_tree = self.complex.create_simplex_tree()

        self.simplex_tree.initialize_filtration()

        diag = self.simplex_tree.persistence()

        return diag

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

        major = pintervals[0][1] - pintervals[0][0]  # compute the persistent-interval with maximal lifetime
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
            high_lifetimes_pi = []  # We seek for all persistent-intervals which birth is greater than the birth of the maximal persistent interval
            max_pi = pintervals[desired_pos]
            lifetime = max_pi[1] - max_pi[0]
            for idx, interv in enumerate(pintervals):

                if interv[0] >= max_pi[0] and lifetime < (interv[1] - interv[0]) * 1.5:
                    high_lifetimes_pi.append(interv)

            intervals_count = len(pintervals)
            init = 0
            if len(high_lifetimes_pi) == 1:
                high_lifetimes_pi = pintervals
                init = int(intervals_count / 2)

            intervals_count = len(high_lifetimes_pi)

            if choice == RANDOMIZED:  # get randomized persistence filtration
                desired_pos = random.randint(init, intervals_count - 1)  # to maximize posibilities

                print("\nLa duracion de vida seleccionado aleatoriamente es {0}\n".format(
                    high_lifetimes_pi[desired_pos][1] - high_lifetimes_pi[desired_pos][0]))
                return high_lifetimes_pi[desired_pos]
            else:  # get average persistence filtration
                Avg = 0
                for interv in high_lifetimes_pi:
                    Avg += interv[1] - interv[0]
                if intervals_count > 0:
                    Avg /= intervals_count
                else:
                    return None

                desired_pos = 0
                min_d = math.fabs((high_lifetimes_pi[0][1] - high_lifetimes_pi[0][
                    0]) - Avg)  # we get the first persistent-interval superior tu average
                for idx, interv in enumerate(high_lifetimes_pi):
                    i = math.fabs((interv[1] - interv[0]) - Avg)
                    if min_d > i and not math.isinf(i):
                        min_d = i
                        desired_pos = idx

        print("el intervalo de persistencia elegido es ", high_lifetimes_pi[desired_pos])

        # return pintervals, desired_pos
        return high_lifetimes_pi[desired_pos]

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

        return pintervals[desired_pos]

    def execute(self, set_2_classify=None, split_data=None):
        pi_selectors = {MAXIMAL: "MAXIMAL", AVERAGE: "AVERAGE", RANDOMIZED: "RANDOMIZED"}
        tdabc_result_list = {}
        knn_result_list = {}
        for selector_id in pi_selectors:
            self.destroy()
            if set_2_classify is None:
                k, j = split_data if split_data is not None else 5, 3
                self.split_dataset(k, j)
            else:
                self.split_dataset()
                self.configure_external_test(set_2_classify)

                #####

            diag = self.build_filtered_simplicial_complex()  # to compute simplicial complex and filtrations
            print("persistence diagrams: ", diag)

            persistence_interval = self.get_desired_persistence_interval(choice=selector_id)

            if persistence_interval is None:  # we ignore the process
                self.destroy()
                return

            self.simplex_tree.prune_above_filtration(persistence_interval[0])

            tdabc_pred = []
            real_values = []
            elems = []
            ttags = [self.tags_position[self.tags_training[i]] for i in self.tags_training]
            ttraining = [e for e in self.training]

            for idx, x0 in self.test:
                idx_key = str([idx])
                value = self.Upsilon(idx)
                elems.append(x0)

                tdabc_pred.append(value)
                real_values.append(self.tags_test[idx_key])

            knn = kNNClassifier(ttraining, ttags)
            knn_pred = knn.execute(elems)
            tdabc_result_list.update({selector_id: (tdabc_pred)})
            knn_result_list.update({selector_id: (knn, knn_pred)})

        return tdabc_result_list, knn_result_list, pi_selectors

    def save_simplex_tree(self):
        path = "./docs/SIMPLEX_TREES"
        file_name = time.strftime("./docs/SIMPLEX_TREES/simplex_tree_%y.%m.%d__%H.%M.%S.txt")
        if not os.path.exists(path):
            os.makedirs(path)

        simplex_tree_file = open(file_name, "w")

        filtrations = self.simplex_tree.get_filtration()

        fmt = "%s --> %.2f"
        for filtered_value in filtrations:
            print(fmt % tuple(filtered_value))
            simplex_tree_file.write(str(filtered_value[0]) + "\n")