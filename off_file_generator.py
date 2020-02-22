from simplex_tree_file_parser import SimplexTreeFileParser


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
            self.write_points_and_colors(off_file)
            self.write_faces(off_file)

            off_file.close()

        def write_header(self, off):
            off.write("COFF")

            N = self.stf_parser.points_count()
            F = self.faces_count()

        def faces_count(self):
            return 0

        def write_points_and_colors(self, off):
            pass

        def write_faces(self, off):

            pass

