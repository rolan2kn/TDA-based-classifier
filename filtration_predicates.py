

class FiltrationPredicate:
    def __init__(self, filtration=None):
        filtration = 0 if filtration is None else filtration
        self.filtration_value = filtration

    def eval(self, new_value):
        return True


class FiltrationEqualPredicate(FiltrationPredicate):
    def __init__(self, filtration=None):
        super(FiltrationEqualPredicate, self).__init__(filtration)

    def eval(self, new_value):
        return self.filtration_value == new_value


class FiltrationLowerThanPredicate(FiltrationPredicate):
    def __init__(self, filtration=None):
        super(FiltrationLowerThanPredicate, self).__init__(filtration)

    def eval(self, new_value):
        return self.filtration_value > new_value


class FiltrationGreaterThanPredicate(FiltrationPredicate):
    def __init__(self, filtration=None):
        super(FiltrationGreaterThanPredicate, self).__init__(filtration)

    def eval(self, new_value):
        return self.filtration_value < new_value


class FiltrationLogicalPredicate(FiltrationPredicate):
    def __init__(self, filtration=None, function=None):
        super(FiltrationLogicalPredicate, self).__init__(filtration)
        self.logical_function = function

    def eval(self, new_value):
        return self.logical_function(self.filtration, new_value)