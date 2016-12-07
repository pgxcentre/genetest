"""
Utilities to build statistical models.
"""

# Dynamically import all the statistical tests so that they are available for
# use in the ModelSpec creation.

import uuid

from .statistics import available_models


class SNPs(object):
    pass


class DependencyManager(object):
    """Class that remembers which items are accessed in an internal set."""
    dependencies = {}

    def __init__(self, source):
        self.source = source

    def __getitem__(self, key):
        deps = self.__class__.dependencies

        if (self.source, key) in deps:
            return deps[(self.source, key)]

        id = str(uuid.uuid4())
        deps[(self.source, key)] = id
        return id

    __getattr__ = __getitem__


class TransformationManager(object):
    """Class that remembers which data manipulation will be necessary."""
    transformations = []

    def __init__(self, action):
        self.action = action

    def __call__(self, target, *params):
        TransformationManager.transformations.append(
            (self.action, target, params)
        )


class ModelSpec(object):
    def __init__(self, outcome, predictors, test):
        """Statistical model specification.

        Args:
            outcome (str): The outcome variable of the model.

        """
        self.outcome = self._clean_outcome(outcome)
        self.predictors = self._clean_predictors(predictors)
        self.test = self._clean_test(test)

    def _clean_test(self, test):
        if test not in available_models:
            raise ValueError(
                "{} is not a valid statistical model.".format(test)
            )
        return test

    def _clean_outcome(self, outcome):
        return outcome

    def _clean_predictors(self, predictors):
        iter_failed = True
        try:
            iter(predictors)
            iter_failed = False
        except TypeError:
            pass
        if iter_failed:
            raise TypeError("'predictors' argument needs to be an iterable.")

        return predictors

    @property
    def dependencies(self):
        return DependencyManager.dependencies

    @property
    def transformations(self):
        return TransformationManager.transformations


phenotypes = DependencyManager("PHENOTYPES")
genotypes = DependencyManager("GENOTYPES")
factor = TransformationManager("ENCODE_FACTOR")
log10 = TransformationManager("LOG10")
pow = TransformationManager("POW")
interaction = TransformationManager("INTERACTION")
grs = TransformationManager("GENETIC_RISK_SCORE")
