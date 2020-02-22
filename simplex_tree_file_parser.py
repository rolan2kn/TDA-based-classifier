from filtration_predicates import FiltrationPredicate


class SimplexTreeFileParser:
    def __init__(self, filename, filtration_predicate = None):
        assert isinstance(filtration_predicate, FiltrationPredicate)

        self.predicate = filtration_predicate
        self.filename = filename
        self.points = []
        self.simplices = dict()

    def execute(self):
        st_file = open(self.filename, "r")

        """
        line format in RegEx = [idx(, idx)*]: one or more indexes separed by commas
        """
        dict_points = {}
        for line in st_file.readlines():
            str_qsimplex, str_point, filt = self.process_line(line)

            if self.predicate.eval(filt):                   # if filtration predicate is True
                sdim = self.parse_qsimplex(str_qsimplex)           # the we get the q-simplex

                if sdim == 0:                                   # a 0-simplex represent a point
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