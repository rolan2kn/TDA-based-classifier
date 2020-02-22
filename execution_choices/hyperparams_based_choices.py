from tdabc_hyperparam_file_parser import TDABCHyperParamFileParser
from tdabc_hyperparam_plotter import TDABCHyperParamPlotter
from tdabc_hyperparam_estimator import TDABCHyperParamEstimator


class HyperparamsTunningChoice:
    def __init__(self):
        pass

    def execute(self):
        tdabc_hyperp_estimator = TDABCHyperParamEstimator()

        tdabc_hyperp_estimator.execute()

        """
        Metodo manual
        """

        # tdabc_hyperp_estimator.execute_with_params(1, 86)

        tdabc_hp_parser = TDABCHyperParamFileParser()

        # valuesdict = tdabc_hp_parser.get_hyper_params()
        all_values = tdabc_hp_parser.merge_all_data()

        for k in all_values:
            print("{0}\n".format(all_values[k]))


class HyperparamPlotterChoice:
    def __init__(self):
        pass

    def execute(self):
        tdabc_hp_plotter = TDABCHyperParamPlotter()
        tdabc_hp_plotter.execute()
