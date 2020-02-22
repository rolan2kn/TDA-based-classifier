# !/usr/bin/env python
# -*- coding: utf-8 -*-

import utils
import numpy as np
import time, os

import argparse
from tda_based_classifier import TDABasedClassifier

from tdabc_hyperparam_estimator import TDABCHyperParamEstimator
from tdabc_hyperparam_file_parser import TDABCHyperParamFileParser
from tdabc_hyperparam_plotter import TDABCHyperParamPlotter
from dataset_handler import DatasetHandler, IRIS, SWISSROLL, DAILY_AND_SPORTS, LIGHT_CURVES
import tdabc_ipynb
from dataset_plotter import DatasetPlotter
from knn_classifier import kNNClassifier

import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap

from execution_choices.dataset_based_choices import *
from execution_choices.hyperparams_based_choices import *
from execution_choices.graphical_based_choices import *


"""
Por que programar al dedo con respecto a Gudhi:

1. Gudhi no permite manipular la construccion del simplex tree
2. Provoca una explosion de memoria, provocando que el S.O mate el proceso sin terminar.
    Para que esto no ocurra debe construirse el RipsComplex con una distancia maxima de 1
    y una dimension maxima de 42. Ademas de que es necesario dividir el dataset en datos de entrenamiento y prueba.
    Para una distancia maxima de 12 como se aconseja, la dimension sugerida es 1, dando lugar solamente a lineas,
    la maxima dimension permitida es de 4. dando posibilidad a 3-simplices.
3. no se puede aprovechar un simplice de una distancia i para construir otro con distancia i+1.
4. no se pueden insertar puntos dinamicamente siendo necesario recalcularlo todo de nuevo.
5. Tiempo de ejecucion exponencial para la construccion del RipsComplex.
6. Gudhi no permite implementar la variante incremental
"""


RUN_TDABC_IRIS, RUN_TDABC_SWISS_ROLL, RUN_TDABC_DSA, RUN_TDABC_LIGHT_CURVES, \
RUN_IPYNB_EXAMPLE, RUN_HYPER_PARAMS_TUNING, RUN_HYPER_PARAMS_PLOTTER, RUN_VISUALIZATIONS = range(8)


class ProgramChoiceSelector:
    def __init__(self, choice=None):
        self.default_opt = choice if not choice is None else RUN_IPYNB_EXAMPLE
        # self.default_opt = RUN_TDABC_IRIS
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-o", "--option", help="RUN_TDABC_IRIS es un entero para guiar la ejecucion del software.",
                                 type=int,
                                 choices= [0, 1, 2, 3, 4, 5, 6])

    def execute(self):
        args = self.parser.parse_args()

        # Aquí procesamos lo que se tiene que hacer con cada argumento
        if args.option:
            print("la opcion seleccionada es: ", args.option)
            self.exec_option(args.option)
        else:
            self.exec_option(self.default_opt)

    def exec_option(self, opt):
        print("option=", opt)
        if RUN_TDABC_IRIS == opt:               #
            IrisChoice().execute()
        elif RUN_TDABC_SWISS_ROLL == opt:       #
            SwissRollChoice().execute()
        elif RUN_TDABC_DSA == opt:              #
            DailyActivitiesChoice().execute()
        elif RUN_TDABC_LIGHT_CURVES == opt:
            LightCurvesChoice().execute()
        elif RUN_HYPER_PARAMS_TUNING == opt:
            HyperparamsTunningChoice().execute()
        elif RUN_HYPER_PARAMS_PLOTTER == opt:
            HyperparamPlotterChoice().execute()
        elif RUN_IPYNB_EXAMPLE == opt:
            JupyterIPYNBChoice().execute()
        elif RUN_VISUALIZATIONS == opt:
            SimplexTreeViewer().execute()
        else:
            pass


if __name__ == '__main__':
    try:
        # ProgramChoiceSelector(RUN_TDABC_IRIS).execute()
        ProgramChoiceSelector(RUN_VISUALIZATIONS).execute()
    except BaseException as e:
        print("ERROR global: {0}".format(e))
