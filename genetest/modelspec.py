"""
Utilities to build statistical models.
"""

# Dynamically import all the statistical tests so that they are available for
# use in the ModelSpec creation.

import uuid
import operator

import numpy as np

from .statistics import available_models


class transformation_handler(object):
    handlers = {}

    def __init__(self, action):
        self.action = action

    def __call__(self, f):
        self.__class__.handlers[self.action] = f
        return f


class SNPs(object):
    pass


class Expression(object):
    def __init__(self, expression):
        self.expression = expression

    def __mul__(self, other):
        return Expression((operator.mul, self.expression, other))

    def __div__(self, other):
        return Expression((operator.div, self.expression, other))

    def __add__(self, other):
        return Expression((operator.add, self.expression, other))

    def __sub__(self, other):
        return Expression((operator.sub, self.expression, other))

    __rmul__ = __mul__
    __rdiv__ = __div__
    __radd__ = __add__
    __rsub__ = __sub__

    def eval(self):
        return self._eval_expression(self.expression)

    @staticmethod
    def _eval_expression(expression):
        op, left, right = expression

        if isinstance(left, tuple):
            left = Expression._eval_expression(left)

        if isinstance(left, Expression):
            left = Expression._eval_expression(left.expression)

        if isinstance(right, tuple):
            right = Expression._eval_expression(right)

        if isinstance(right, Expression):
            right = Expression._eval_expression(right.expression)

        if hasattr(left, "values"):
            left = left.values

        if hasattr(right, "values"):
            right = right.values

        return op(left, right)


class EntityIdentifier(object):
    def __init__(self, id=None):
        if id:
            self.id = id
        else:
            self.id = str(uuid.uuid4())
        self.values = None

    def __mul__(self, other):
        return Expression((operator.mul, self, other))

    def __div__(self, other):
        return Expression((operator.div, self, other))

    def __add__(self, other):
        return Expression((operator.add, self, other))

    def __sub__(self, other):
        return Expression((operator.sub, self, other))

    __rmul__ = __mul__
    __rdiv__ = __div__
    __radd__ = __add__
    __rsub__ = __sub__

    def __repr__(self):
        return self.id.split("-")[0]

    def bind(self, values):
        self.values = values


class DependencyManager(object):
    """Class that remembers which items are accessed in an internal set."""
    dependencies = {}

    def __init__(self, source):
        self.source = source

    def __getitem__(self, key):
        deps = self.__class__.dependencies

        if (self.source, key) in deps:
            return deps[(self.source, key)]

        id = EntityIdentifier()
        deps[(self.source, key)] = id
        return id

    __getattr__ = __getitem__


class TransformationManager(object):
    """Class that remembers which data manipulation will be necessary."""
    transformations = []

    def __init__(self, action):
        self.action = action

    def __call__(self, source, *params, name=None):
        # source is either an EntityIdentifier or an Expression
        # Generate an identifier for the result of the transformation.
        if name is None:
            # Generate an interpretable name.
            if isinstance(source, Expression):
                raise ValueError(
                    "The name parameter is mandatory for expression-based "
                    "transformations."
                )
            else:
                name = "TRANSFORM:{}:{}".format(self.action, source.id)

        target = EntityIdentifier(name)
        TransformationManager.transformations.append(
            (self.action, source, target, params)
        )
        return target


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

    def get_translations(self):
        """Returns a dict mapping IDs to regular variable names."""
        return {id.id: tu[1] for tu, id in self.dependencies.items()}

    def create_data_matrix(self, phenotypes, genotypes):
        """Create the data matrix given data containers.

        Args:
            phenotypes (PhenotypeContainer): The PhenotypeContainer instance.
            genotypes (GenotypeContainer): The GenotypeContainer instance.

        """
        # Extract the phenotype dependencies.
        PHENOTYPES = "PHENOTYPES"

        phen_keys = [
            k[1] for k, v in self.dependencies.items() if k[0] == PHENOTYPES
        ]

        df = phenotypes.get_phenotypes(phen_keys)
        df.columns = [
            self.dependencies[(PHENOTYPES, k)].id for k in phen_keys
        ]

        # Extract the genotype dependencies.
        GENOTYPES = "GENOTYPES"

        geno_keys = [
            k[1] for k, v in self.dependencies.items() if k[0] == GENOTYPES
        ]

        for marker in geno_keys:
            entity_id = self.dependencies[(GENOTYPES, marker)]

            g = genotypes.get_genotypes(marker).genotypes
            df[entity_id.id] = g

            # Also bind the EntityIdentifier.
            entity_id.bind(g)

        # Apply transformations.
        for action, source, target, params in self.transformations:
            f = transformation_handler.handlers[action]
            res = f(df, source, *params)

            # Some tranformations return multiple columns. We create all the
            # relevant columns in the dataframe.
            if isinstance(res, tuple):
                for i, col in enumerate(res):
                    df["{}:{}".format(target.id, i + 1)] = col

            elif isinstance(res, dict):
                for key, col in res.items():
                    df["{}:{}".format(target.id, key)] = col

            # In most cases, transformations return a single array. We set it
            # under the target ID.
            else:
                df[target.id] = res

        # Only keep predictors and outcomes.
        # TODO

        return df


@transformation_handler("LOG10")
def _log10(data, entity):
    return np.log10(data[entity.id])


@transformation_handler("ENCODE_FACTOR")
def _encode_factor(data, entity):
    raise NotImplementedError()
    return {
        "level1": data[entity.id],
        "level2": data[entity.id]
    }


@transformation_handler("POW")
def _pow(data, entity, power):
    return np.pow(data[entity.id], power)


@transformation_handler("INTERACTION")
def _interaction(data, entity, interaction_target):
    # It is possible that there are multiple levels in the interaction_target.
    # This is TODO
    return {
        interaction_target.id: data[entity.id] * data[interaction_target.id]
    }


@transformation_handler("GENETIC_RISK_SCORE")
def _grs(data, entity):
    if not isinstance(entity, Expression):
        raise ValueError("grs function requires an expression.")
    return entity.eval()

phenotypes = DependencyManager("PHENOTYPES")
genotypes = DependencyManager("GENOTYPES")

factor = TransformationManager("ENCODE_FACTOR")
log10 = TransformationManager("LOG10")
pow = TransformationManager("POW")
interaction = TransformationManager("INTERACTION")
grs = TransformationManager("GENETIC_RISK_SCORE")
