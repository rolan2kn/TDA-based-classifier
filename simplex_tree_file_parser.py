from filtration_predicates import FiltrationPredicate
import gudhi
import time
import os
import utils


class SimplexTreeFileParser:
    def __init__(self, source=None, point_list = None, filtration_predicate = None):
        assert isinstance(filtration_predicate, FiltrationPredicate)
        assert source is not None

        self.predicate = filtration_predicate
        self.filename = None
        self.simplex_tree = None
        self.points = []

        if isinstance(source, str):
            self.filename = source
        elif isinstance(source, gudhi.simplex_tree.SimplexTree):
            self.simplex_tree = source
            assert point_list is not None
            self.points = point_list

        self.simplices = dict()

    @staticmethod
    def get_filtration_values(simplex_tree_file):
        assert simplex_tree_file is not None

        st_file = open(simplex_tree_file, "r")
        filt_value_list = []
        filt_dict = {}

        for line in st_file.readlines():
            splitted_line = line.split(":")
            filt = splitted_line[-1]
            filt_dict.update({filt.strip(): ""})
            del splitted_line

        st_file.close()
        for filt in filt_dict:
            filt_value_list.append(float(filt))
        filt_dict.clear()
        del filt_dict

        return filt_value_list

    def execute(self):
        if self.simplex_tree is not None:
            self.parse_simplex_tree()
        elif self.filename is not None:
            self.parse_file()
        return

    def parse_file(self):
        st_file = open(self.filename, "r")

        """
        line format in RegEx = [idx(, idx)*]: one or more indexes separed by commas
        """
        dict_points = {}
        del self.points
        self.points = []

        for line in st_file.readlines():
            str_qsimplex, str_point, filt = self.process_line(line)

            if filt == 0 or self.predicate.eval(filt):  # if filtration predicate is True
                sdim = self.parse_qsimplex(str_qsimplex)  # the we get the q-simplex

                if sdim == 0:  # a 0-simplex represent a point
                    point = self.parse_point(str_point)
                    point_idx = self.simplices[0][-1][0]
                    dict_points.update({point_idx: point})

        st_file.close()
        keys = dict_points.keys()

        for idx in sorted(keys):
            self.points.append(dict_points[idx])
        dict_points.clear()
        del keys
        del dict_points

    def parse_simplex_tree(self):
        filtrations = self.simplex_tree.get_filtration()

        for filtered_value in filtrations:
            qsimplex = filtered_value[0]
            filt = filtered_value[1]

            if filt == 0.0 or self.predicate.eval(filt):
                simplex_dimension = len(filtered_value[0]) - 1
                if not simplex_dimension in self.simplices:
                    self.simplices.update({simplex_dimension: []})

                self.simplices[simplex_dimension].append([int(s) for s in qsimplex])

        del filtrations

    def process_line(self, line):
        splitted_line = line.split(":")
        point = ""

        str_qsimplex = splitted_line[0]

        if len(splitted_line) == 2:
            filt = splitted_line[1]
        elif len(splitted_line) == 3:
            point = splitted_line[1]
            filt = splitted_line[2]

        return str_qsimplex, point, float(filt)

    def parse_qsimplex(self, line):
        pos_init = line.find("[")  # left bracket index
        pos_end = line.find("]")  # right bracket index

        if pos_init == -1 or pos_end == -1:  # if any bracket was not found we jump to next line
            return

        inner_line = line[pos_init + 1: pos_end]
        qsimplex = inner_line.split(",")

        simplex_dimension = len(qsimplex) - 1
        if not simplex_dimension in self.simplices:
            self.simplices.update({simplex_dimension: []})

        self.simplices[simplex_dimension].append([int(s) for s in qsimplex])

        return simplex_dimension

    def parse_point(self, str_point):
        if str_point is None or str_point == "":
            return

        str_point = str_point[1:-1]     # remove left and right parenthesis
        if len(str_point) == 0:
            return

        str_point = str_point[1:-1]     # remove left and right square brackets
        elems = str_point.split(",")    # we process the elements

        point = [float(p) for p in elems]

        del str_point
        del elems

        return point

    def points_count(self):
        if self.points is None:
            return 0
        return len(self.points)

    def simplices_count_by_dimension(self, face_dim=0):
        if self.simplices is None or len(self.simplices) == 0:
            return 0
        if not face_dim in self.simplices:
            return 0

        return len(self.simplices[face_dim])
