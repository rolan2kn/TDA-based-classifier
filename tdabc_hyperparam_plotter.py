#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random

import time
import os
import pandas as pd
import pickle as pickle
import gudhi as gd
from pylab import *
import seaborn as sns

from tdabc_hyperparam_file_parser import TDABCHyperParamFileParser


class TDABCHyperParamPlotter:
    def __init__(self):
        self.hpParser = TDABCHyperParamFileParser()

    def execute(self):
        all_values = self.hpParser.merge_all_data()
        fig = plt.figure()
        fig.set_size_inches(15, 10)
        ax = fig.add_subplot(111)

        ks = ["Time", "Storage"]

        coeff0 = range(151)

        coeff = [c for c in coeff0]
        memory = np.exp2(coeff)
        time = np.exp2(coeff0)
        simplex_size = np.log2(150)

        mem_size = []
        tem_cplx = []
        ms = 0
        for id, i in enumerate(memory):
            ms += (memory[id]*simplex_size) #* (10**(-9))

            ts = (time[id]*simplex_size)
            mem_size.append(ms)
            tem_cplx.append(ts)
            print("dimension: {0}, storage: {1}, time: {2}".format(id, ms, ts))
        area = (15) ** 2
        for idx, c in enumerate(['b', 'g', 'r', 'c', 'm', 'y']):
            value = mem_size
            if ks[idx].find("Time") != -1:
                value = tem_cplx
            ax.plot(value[0], value[1], s=area, c=c, marker='o', label=ks[idx].strip())

        t = range(151)
        t2 = [c - 33 for c in t]

        y1 = (2 ** (t))*simplex_size
        y2 = ((2 ** (t2))*simplex_size)

        ax.plot(t[0:10], mem_size[0:10], c="r", marker='o', label="Storage Bits")

        ax.plot(t[0:10], tem_cplx[0:10], c="g", marker='o', label="Time Seconds")

        # Texto en la gráfica en coordenadas (x,y)
        texto1 = ax.text(7, 60, r'$2^{(N - 33)}*log(N)$', fontsize=20)

        texto2 = ax.text(5, 500, r'$2^N*log(N)$', fontsize=20)

        # Añado una malla al gráfico
        ax.grid()

        ax.set(xlabel="q Dimension", ylabel="Complexity", size=30)
        ax.set_title(label="Storage and Time Complexity", fontsize=30)
        plt.text.set_size(20)

        legend(fontsize=20)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(18)

        show()

        ax.set_xlabel('q Dimension', fontdict={'size': 30})
        ax.set_ylabel('Result value', fontdict={'size': 30})
        # ax.
        # path = "./docs/HYPER_PARAM"
        # file_name = time.strftime("{0}/hyper-param_%Y%M%d__%H%M%S.png".format(path))
        #
        # if not os.path.exists(file_name):
        #     os.makedirs(file_name)
        #
        # plt.savefig(file_name)




