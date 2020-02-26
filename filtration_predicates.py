

class FiltrationPredicate:
    def __init__(self, filtration=None):
        filtration = 0 if filtration is None else filtration
        self.filtration_value = filtration

    def eval(self, new_value):
        return True

    def __str__(self):
        return "FiltrationPredicate"

    def class_name(self):
        return "FiltrationPredicate"


class FiltrationEqualPredicate(FiltrationPredicate):
    def __init__(self, filtration=None):
        super(FiltrationEqualPredicate, self).__init__(filtration)

    def eval(self, new_value):
        return self.filtration_value == new_value

    def __str__(self):
        return "EqualTo{0}".format(self.filtration_value)

    def class_name(self):
        return "FiltrationEqualPredicate"


class FiltrationLowerThanPredicate(FiltrationPredicate):
    def __init__(self, filtration=None):
        super(FiltrationLowerThanPredicate, self).__init__(filtration)

    def eval(self, new_value):
        return self.filtration_value > new_value

    def __str__(self):
        return "LowerThan{0}".format(self.filtration_value)

    def class_name(self):
        return "FiltrationLowerThanPredicate"


class FiltrationGreaterThanPredicate(FiltrationPredicate):
    def __init__(self, filtration=None):
        super(FiltrationGreaterThanPredicate, self).__init__(filtration)

    def eval(self, new_value):
        return self.filtration_value < new_value

    def __str__(self):
        return "GreaterThan{0}".format(self.filtration_value)

    def class_name(self):
        return "FiltrationGreaterThanPredicate"


class FiltrationOnOpenIntervalPredicate(FiltrationPredicate):
    def __init__(self, filtrationInit=None, filtrationEnd=None,):
        super(FiltrationOnOpenIntervalPredicate, self).__init__(filtrationInit)
        self.filtrationEnd = filtrationEnd

    def eval(self, new_value):
        return self.filtration_value < new_value < self.filtrationEnd

    def __str__(self):
        return "Between{0}And{1}".format(self.filtration_value, self.filtrationEnd)

    def class_name(self):
        return "FiltrationOnOpenIntervalPredicate"