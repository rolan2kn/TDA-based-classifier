from tda_based_classifier import TDABasedClassifier
from dataset_handler import IRIS, SWISSROLL, DAILY_AND_SPORTS, LIGHT_CURVES

class IrisChoice:
    def __init__(self):
        pass

    def execute(self):
            tdabClassifier = TDABasedClassifier(dataset_type=IRIS)
            tdabClassifier.execute()


class SwissRollChoice:
    def __init__(self):
        pass

    def execute(self):
            tdabClassifier = TDABasedClassifier(dataset_type=SWISSROLL)
            tdabClassifier.execute()


class DailyActivitiesChoice:
    def __init__(self):
        pass

    def execute(self):
        tdabClassifier = TDABasedClassifier(dataset_type=DAILY_AND_SPORTS)
        tdabClassifier.execute()


class LightCurvesChoice:
    def __init__(self):
        pass

    def execute(self):
        tdabClassifier = TDABasedClassifier(dataset_type=LIGHT_CURVES)
        tdabClassifier.execute()