#!/usr/bin/python
# -*- coding: utf-8 -*-

class MetricsFileParser:
    def __init__(self, filename):
        self.filename = filename
        self.precision = 0.0
        self.recall = 0.0
        self.agree_rate = 0.0
        self.fp_rate = 0.0
        self.f1_measure = 0.0
        self.mse = 0.0

        self.precision_by_class = {}
        self.recall_by_class = {}
        self.agree_rate_by_class = {}
        self.fp_rate_by_class = {}
        self.f1_measure_by_class = {}
        self.mse_by_class = {}

    def parse(self):
        file = open(self.filename, "r")
        cc = ""
        no_clases = {}
        count_general = 0
        read_general = False
        read_class = False

        for line in file.readlines():
            if len(line) < 2:
                continue
            if line.find("**") != -1:
                #estoy por leer un bloque de datos
                pos = line.find("--")
                if pos == 2: #bloque de datos generales
                    read_general = True
                    read_class = False
                    count_general += 1
                else:        #bloque de datos de clase
                    read_class = True
                    read_general = False
                    split = line.split(":")
                    cc = split[0][2:].lower().strip()
                    if not cc in no_clases:
                        no_clases.update({cc: 0})
                    if not cc in self.precision_by_class:
                        self.precision_by_class[cc] = 0
                    if not cc in self.recall_by_class:
                        self.recall_by_class[cc] = 0
                    if not cc in self.agree_rate_by_class:
                        self.agree_rate_by_class[cc] = 0
                    if not cc in self.fp_rate_by_class:
                        self.fp_rate_by_class[cc] = 0
                    if not cc in self.f1_measure_by_class:
                        self.f1_measure_by_class[cc] = 0
                    if not cc in self.mse_by_class:
                        self.mse_by_class[cc] = 0

                    no_clases[cc] += 1
            else:
                split = line.split(":")
                attr_name = split[0].lower().strip()
                attr_value = float(split[1].lower().strip())

                if read_general:
                    if attr_name.find("precision") != -1:
                        self.precision += attr_value
                    elif attr_name.find("recall") != -1:
                        self.recall += attr_value
                    elif attr_name.find("agree_rate") != -1:
                        self.agree_rate += attr_value
                    elif attr_name.find("fp_rate") != -1:
                        self.fp_rate += attr_value
                    elif attr_name.find("f1_measure") != -1:
                        self.f1_measure += attr_value
                    elif attr_name.find("mse") != -1:
                        self.mse += attr_value
                elif read_class:

                    if attr_name.find("precision") != -1:
                        self.precision_by_class[cc] += attr_value
                    elif attr_name.find("recall") != -1:
                        self.recall_by_class[cc] += attr_value
                    elif attr_name.find("agree_rate") != -1:
                        self.agree_rate_by_class[cc] += attr_value
                    elif attr_name.find("fp_rate") != -1:
                        self.fp_rate_by_class[cc] += attr_value
                    elif attr_name.find("f1_measure") != -1:
                        self.f1_measure_by_class[cc] += attr_value
                    elif attr_name.find("mse") != -1:
                        self.mse_by_class[cc] += attr_value

        count_general = 1 if not count_general else count_general
        self.precision /= count_general
        self.recall /= count_general
        self.agree_rate /= count_general
        self.fp_rate /= count_general
        self.f1_measure /= count_general
        self.mse /= count_general

        for cc in no_clases:
            no = 1 if not no_clases[cc] else no_clases[cc]
            self.precision_by_class[cc] /= no
            self.recall_by_class[cc] /= no
            self.agree_rate_by_class[cc] /= no
            self.fp_rate_by_class[cc] /= no
            self.f1_measure_by_class[cc] /= no
            self.mse_by_class[cc] /= no

        file.close()

        file = open(self.filename, "a")
        file.write("\n\n\n##################################################################")
        file.write("\n\n\n##################################################################")
        file.write("\n\n\n##################################################################")
        file.write("\n\n\n##################################################################")
        file.write("\n\n\n##################################################################")
        file.write(
            "\n\n**--*-*-*-*-*-*-*-*-*-*-*\n agree_rate: {0}\nprecision: {1}\nrecall: {2}\nfp_rate: {3}\nmse: {4}\nf1_measure: {5}".format(
                self.agree_rate, self.precision, self.recall, self.fp_rate, self.mse, self.f1_measure))
        file.write(
            "\n\n**SETOSA: --*-*-*-*-*-*-*-*-*-*-*\n agree_rate: {0}\nprecision: {1}\nrecall: {2}\nfp_rate: {3}\nmse: {4}\nf1_measure: {5}".format(
                self.agree_rate_by_class['setosa'],
                self.precision_by_class['setosa'],
                self.recall_by_class['setosa'],
                self.fp_rate_by_class['setosa'],
                self.mse_by_class['setosa'],
                self.f1_measure_by_class['setosa']))
        file.write(
            "\n\n**VIRGINICA: --*-*-*-*-*-*-*-*-*-*-*\n agree_rate: {0}\nprecision: {1}\nrecall: {2}\nfp_rate: {3}\nmse: {4}\nf1_measure: {5}".format(
                self.agree_rate_by_class['virginica'],
                self.precision_by_class['virginica'],
                self.recall_by_class['virginica'],
                self.fp_rate_by_class['virginica'],
                self.mse_by_class['virginica'],
                self.f1_measure_by_class['virginica']))
        file.write(
            "\n\n**VERSICOLOR: --*-*-*-*-*-*-*-*-*-*-*\n agree_rate: {0}\nprecision: {1}\nrecall: {2}\nfp_rate: {3}\nmse: {4}\nf1_measure: {5}".format(
                self.agree_rate_by_class['versicolor'],
                self.precision_by_class['versicolor'],
                self.recall_by_class['versicolor'],
                self.fp_rate_by_class['versicolor'],
                self.mse_by_class['versicolor'],
                self.f1_measure_by_class['versicolor']))
        file.close()
