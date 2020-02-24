from tdabc_ipynb import TDABasedClassifier4IPYNB
from tda_based_classifier import TDABasedClassifier

import utils
import numpy as np
import time, os

from dataset_handler import DatasetHandler, IRIS, SWISSROLL, DAILY_AND_SPORTS, LIGHT_CURVES
from dataset_plotter import DatasetPlotter
from knn_classifier import kNNClassifier

from filtration_predicates import FiltrationPredicate, FiltrationEqualPredicate, \
    FiltrationLowerThanPredicate, FiltrationGreaterThanPredicate, FiltrationOnOpenIntervalPredicate
from simplex_tree_file_parser import SimplexTreeFileParser
from off_file_generator import OffFileGenerator


import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap


class JupyterIPYNBChoice:
    def __init__(self):
        pass

    def execute(self):
        tdabc_iris = TDABasedClassifier4IPYNB(dataset_name=IRIS, dim_2_read=2)
        # tdabc_iris.load_data()
        # y_real, y_pred = tdabc_iris.execute()
        #
        # print("y_real\n\n", y_real)
        # print("y_pred\n\n", y_pred)

        ##### modify test
        tdabc_iris.load_data()
        tdabc_iris.split_dataset()

        training = [i for i in tdabc_iris.training]
        test = [tdabc_iris.tags_position[tdabc_iris.tags_training[i]] for i in tdabc_iris.tags_training]

        # tdabc_iris.draw_data()
        X = np.array(tdabc_iris.dataset)  # we only take the first two features. We could
        # avoid this ugly slicing by using a two-dim dataset

        x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
        y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 5),
                             np.linspace(y_min, y_max, 5))
        v = np.c_[xx.ravel(), yy.ravel()]
        values = [[a, b] for a,b in v]

        print("values: =====>>> ", values)

        fold_sizes = [5, 10, 15, 20, 25, 30, 35]
        for split_data in fold_sizes:

            tdabc_results, knn_results, algorithms = tdabc_iris.execute(set_2_classify=values, split_data=split_data)

            ########################

            titles = ["TDABC decision boundaries", "k-NN decision boundaries"]

            # data_plotter = DatasetPlotter(tdabc_iris.dataset)
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
            values = np.array(values)
            _xxx = np.array(training)


            # Plot the predicted probabilities. For that, we will assign a color to
            # each point in the mesh [x_min, m_max]x[y_min, y_max].

            # plt.show()

            # just plot the dataset first

            for algorithm_idx in algorithms:
                plt.figure(figsize=(15, 5))
                algorithm_name = algorithms[algorithm_idx]
                tdabc_alg_result = tdabc_results[algorithm_idx]
                (knn_classifier, knn_alg_result) = knn_results[algorithm_idx]
                ttags = knn_classifier.training_tags
                ttraining = knn_classifier.training
                # ttraining.extend(knn_classifier.new_data)
                xxx = np.array(ttraining)

                # plt.figure(figsize=(10, 5))
                plt.subplot(1, 3, 1)

                Z = np.array(test)

                plt.scatter(_xxx[:, 0], _xxx[:, 1], c=np.array(['#FF0000', '#00FF00', '#0000FF'])[test],
                            edgecolors="k", cmap=cm_bright, label=tdabc_iris.labels)

                # Plot the testing points

                plt.xlabel('Sepal length')
                plt.ylabel('Sepal width')
                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
                plt.xticks(())
                plt.yticks(())
                plt.title("{0}".format("Iris Dataset"))

                plt.tight_layout()
                plt.title("Iris Dataset")

                for i, clf in enumerate((tdabc_alg_result, knn_alg_result)):
                    # Plot the predicted probabilities. For that, we will assign a color to
                    # each point in the mesh [x_min, m_max]x[y_min, y_max].
                    plt.subplot(1, 3, i + 2)

                    Z = np.array(clf)

                    # Put the result into a color plot
                    _Z = Z.reshape((xx.shape))
                    plt.contourf(xx, yy, _Z, cmap=cm, alpha=.8)

                    # Plot the training points

                    plt.scatter(xxx[:, 0], xxx[:, 1], c=np.array(['#FF0000', '#00FF00', '#0000FF'])[ttags],
                                edgecolors="k", cmap=cm_bright, label=tdabc_iris.labels)

                    # Plot the testing points

                    plt.scatter(values[:, 0], values[:, 1], c=np.array(["r", "g", "b"])[Z],
                                edgecolors="k", alpha=0.5, label=tdabc_iris.labels)
                    plt.xlabel('Sepal length')
                    plt.ylabel('Sepal width')
                    plt.xlim(xx.min(), xx.max())
                    plt.ylim(yy.min(), yy.max())
                    plt.xticks(())
                    plt.yticks(())
                    plt.title("{0}".format(titles[i]))

                plt.tight_layout()

                path = "{0}/docs/DATA_GRAPHICS/comparison/".format(utils.get_module_path())
                if not os.path.isdir(path):
                    os.makedirs(path)
                file_name = time.strftime(
                    "{0}_{1}_%y.%m.%d__%H.%M.%S.png".format(path, algorithm_name))

                plt.title(titles[i])
                # plt.legend(fontsize=20)
                plt.savefig(file_name)
                # plt.show()


class SimplexTreeViewer:
    def __init__(self):
        pass

    def execute(self):
        import gudhi

        # tdabc = TDABasedClassifier(data_file_name=IRIS)
        # tdabc.init_data()
        # tdabc.split_dataset(5, 0)
        # tdabc.build_filtered_simplicial_complex()
        # tdabc.draw_simplex_tree()

        # self.load_simplex_tree()
        self.create_off_file()
        # self.show_off(index_off = 0)

    def example(self):
        import gudhi
        points = [1, 1], [7, 0], [4, 6], [9, 6], [0, 14], [2, 19], [9, 17]
        alpha_complex = gudhi.AlphaComplex(points=[[1, 1], [7, 0], [4, 6], [9, 6], [0, 14], [2, 19], [9, 17]])

        simplex_tree = alpha_complex.create_simplex_tree()
        result_str = 'Alpha complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
                     repr(simplex_tree.num_simplices()) + ' simplices - ' + \
                     repr(simplex_tree.num_vertices()) + ' vertices.'
        print(result_str)
        fmt = '%s:(%s):%.2f'
        for filtered_value in simplex_tree.get_filtration():
            str_line = fmt % tuple((filtered_value[0], "a,b,c",filtered_value[1]))
            qsimplex, point, filt = str_line.split(":")

            inner_simplex = qsimplex[1:-1]
            if inner_simplex.find(",") != -1:
                point = ""

            print(fmt % tuple((filtered_value[0], point, filtered_value[1])))
            print("qsimplex = {0}, point = {1}, filt = {2}".format(qsimplex, point, filt) )

    def load_simplex_tree(self):
        # filename = "{0}/docs/SIMPLEX_TREES/{1}".format(utils.get_module_path(), "simplex_tree_20.02.19__14.27.30.txt")
        filename = "{0}/docs/SIMPLEX_TREES/{1}".format(utils.get_module_path(), "simplex_tree_20.02.19__16.20.47.txt")
        # filtration_predicate = FiltrationLowerThanPredicate(3)
        # filtration_predicate = FiltrationLowerThanPredicate(2)
        # filtration_predicate = FiltrationLowerThanPredicate(1)
        # filtration_predicate = FiltrationLowerThanPredicate(0.5)
        # filtration_predicate = FiltrationLowerThanPredicate(0.2)
        # filtration_predicate = FiltrationOnOpenIntervalPredicate(0.2, 0.5)
        # filtration_predicate = FiltrationOnOpenIntervalPredicate(0.3, 0.5)
        # filtration_predicate = FiltrationOnOpenIntervalPredicate(0.5, 1)
        filtration_predicate = FiltrationOnOpenIntervalPredicate(1, 1.5)

        st_parser = SimplexTreeFileParser(filename, filtration_predicate)
        st_parser.execute()

        return st_parser

    def create_off_file(self):
        path = "{0}/docs/SIMPLEX_TREES/".format(utils.get_module_path())
        if not os.path.isdir(path):
            os.makedirs(path)
        off_file_name = time.strftime(
            "{0}_{1}_%y.%m.%d__%H.%M.%S.off".format(path, "simplex_tree"))

        off_file_gen = OffFileGenerator(off_file_name)
        stfp = self.load_simplex_tree()
        off_file_gen.init_from_simplex_tree_parser(stfp)
        off_file_gen.execute()

    def show_off(self, index_off = 0):
        filepath = "{0}/docs/SIMPLEX_TREES/".format(utils.get_module_path())
        all_off_files = utils.get_all_filenames(root_path=filepath, file_pattern=".off")

        if len(all_off_files) == 0 or len(all_off_files) < index_off-1:
            return

        os.system("geomview {0}".format(all_off_files[index_off]))