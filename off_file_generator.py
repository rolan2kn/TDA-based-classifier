from simplex_tree_file_parser import SimplexTreeFileParser
from scipy import special
import itertools


class OffFileGenerator:
        def __init__(self, filename):
            self.stf_parser = None
            self.off_filename = filename

        def init_from_simplex_tree_file(self, st_filename, filt_predicate):
            assert st_filename is not None
            self.stf_parser = SimplexTreeFileParser(st_filename, filt_predicate)

        def init_from_simplex_tree_parser(self, parser):
            assert parser is not None and isinstance(parser, SimplexTreeFileParser)
            self.stf_parser = parser

        def execute(self):
            off_file = open(self.off_filename, "w")

            self.write_header(off_file)
            self.write_points(off_file)
            self.write_faces(off_file)

            off_file.close()

        def write_header(self, off):
            off.write("OFF\n")

            N = self.stf_parser.points_count()
            F = self.faces_count()
            E = 0
            off.write("{0} {1} {2}\n".format(int(N), int(F), int(E)))

        def faces_count(self):
            total_faces = 0
            for dimension in self.stf_parser.simplices:
                total_2simplices = special.binom(dimension+1, 3)
                if total_2simplices == 0:
                    total_2simplices = 1
                total_faces += self.stf_parser.simplices_count_by_dimension(dimension)*total_2simplices

            return total_faces

        def face_points_by_dimension(self, dimension):
            if dimension < 3:
                return dimension
            return 3

        def write_points(self, off):
            for point in self.stf_parser.points:
                p = point[:-1]
                str_point = [str(c) for c in p]
                line = "\t{0}\n".format(" ".join(str_point))
                off.write(line)

        def write_faces(self, off):
            for dimension in self.stf_parser.simplices:
                qsimplices = self.stf_parser.simplices[dimension]

                for qsimplex in qsimplices:
                    self.write_qsimplex(qsimplex, off)

        def write_qsimplex(self, qsimplex, off):
            face_points = self.face_points_by_dimension(len(qsimplex))
            face_lists = list(itertools.combinations(qsimplex, 3))
            face_color = self.color_by_dimension(qsimplex)

            if len(face_lists) == 0:        # write simplex from 0 and 1 dimensions
                str_simplex = [str(s) for s in qsimplex]
                line = "{0}\t{1}\t{2}\n".format(face_points, " ".join(str_simplex), face_color)
                off.write(line)
            else:                       # write a q-simplex with dimension q > 1
                del qsimplex
                for qsimplex in face_lists:
                    str_simplex = [str(s) for s in qsimplex]
                    line = "{0}\t{1}\t{2}\n".format(face_points, " ".join(str_simplex), face_color)
                    off.write(line)

        def color_by_dimension(self, simplex):
            if len(simplex) == 1:
                return "1.0 0.0 0.0"
            if len(simplex) == 2:
                return "1.0 0.5 0.5"
            if len(simplex) == 3:
                return "0.0 1.0 1.0"
            if len(simplex) == 4:
                return "0.0 1.0 0.0"

            return "0.0 0.0 1.0"


