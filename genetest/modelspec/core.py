"""
Utilities to build statistical models.
"""

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import re
import uuid
import operator
import itertools
import functools

import numpy as np
import pandas as pd

from ..statistics import model_map
from ..statistics.descriptive import get_maf


SNPs = "SNPs"
model = "MODEL"


class PheWAS(object):
    """A collection of EntityIdentifiers used to describe pheWAS analyses."""
    def __init__(self, li=None):
        self.li = li


class transformation_handler(object):
    handlers = {}

    def __init__(self, action):
        self.action = action

    def __call__(self, f):
        self.__class__.handlers[self.action] = f
        return f


class Result(object):
    def __init__(self, entity):
        self.path = [entity]

    def __getitem__(self, key):
        self.path.append(key)
        return self

    def get(self, results):
        cur = results

        for field in self.path:
            if isinstance(field, EntityIdentifier):
                # Access for transformations and entities.
                if field.id in results.keys():
                    cur = cur[field.id]
                else:
                    # Check for many levels.
                    results_per_level = {}

                    for key in results:
                        split_key = key.split(":")
                        id, level = (":".join(split_key[:-1]), split_key[-1])
                        if id == field.id:
                            results_per_level[level] = cur[key]

                    cur = results_per_level

            else:
                # Access for special fields like "SNPs", "MODEL" or for
                # regular keys.
                cur = cur[field]

        return cur


class ResultMetaclass(object):
    def __getitem__(self, entity):
        return Result(entity)


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
    """Placeholder for variables in the modelspec.

    Instances get assigned an ID (a UUID4 if nothing is specified), the
    chosen identifier otherwise.

    It is possible to do basic arithmetic on instances. This will return an
    Expression object that can be evaluated. Before evaluation, data needs
    to be bound to the EntityIdentifier using the bind method.

    """
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
    """Class that remembers which items are accessed in an internal set.

    For example, "genotypes" and "phenotypes" are dependency managers.

    """
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

        else:
            name = "TRANSFORM:{}".format(name)

        target = EntityIdentifier(name)

        # Make sure there are no entities with the same name.
        existing_transformations = {
            i[2].id: i[2] for i in TransformationManager.transformations
        }
        if target.id in existing_transformations:
            # The transformation was already performed, so we'll just return
            # the already created target
            return existing_transformations[target.id]

        TransformationManager.transformations.append(
            (self.action, source, target, params)
        )
        return target


class ModelSpec(object):
    def __init__(self, outcome, predictors, test, no_intercept=False,
                 stratify_by=None):
        """Statistical model specification.

        Args:
            outcome (EntityIdentifier): The outcome variable of the model.
            predictors (list): A list of EntityIdentifier that represent
                               covariates.
            test (callable or str): Either the name of a statistical test or
                                    a callable that returns an instance of a
                                    statistical test.
            no_intercept (bool): Controls if a column of ones is to be added.
            stratify_by (list): A list of EntityIdentifier instances
                                to stratify the analysis.

        """
        self.outcome = self._clean_outcome(outcome)
        self.predictors = self._clean_predictors(predictors)
        self.test = self._clean_test(test)
        self.no_intercept = no_intercept

        if hasattr(stratify_by, "__iter__"):
            self.stratify_by = stratify_by
        elif stratify_by is not None:
            self.stratify_by = list(stratify_by)
        else:
            self.stratify_by = None

        # SNP metadata is stored in the modelspec because it is obtained as a
        # consequence of building the data matrix.
        # Hence, it is more efficient to cache it than to request it.
        self.variant_metadata = {}

    def _clean_test(self, test):
        """Returns a factory function to create instances of a statistical
        model.

        Args:
            test (str, class or callable): This can be either the name of the
                                           test or a class or callable that
                                           returns a valid instance.

        """
        try:
            return model_map[test]
        except KeyError:
            pass

        if hasattr(test, "__call__"):
            return test

        raise ValueError(
            "{} is not a valid statistical model.".format(test)
        )

    def _clean_outcome(self, outcome):
        if isinstance(outcome, EntityIdentifier):
            return {"y": outcome}

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

    def get_tested_variants(self):
        if SNPs in self.predictors:
            return SNPs
        else:
            return [
                (v, k[1]) for k, v in self.dependencies.items()
                if k[0] == "GENOTYPES"
            ]

    def get_translations(self):
        """Returns a dict mapping IDs to regular variable names."""
        return {id.id: tu[1] for tu, id in self.dependencies.items()}

    def create_data_matrix(self, phenotypes, genotypes):
        """Create the data matrix given data containers.

        Args:
            phenotypes (PhenotypeContainer): The phenotypes.
            genotypes (geneparse.core.Genotypes): The genotypes.

        """
        PHENOTYPES = "PHENOTYPES"

        # Extract the phenotype dependencies.
        if isinstance(self.outcome, PheWAS) and self.outcome.li is None:
            # Get all the phenotypes.
            df = phenotypes.get_phenotypes()

            # Create entities if needed for the outcomes.
            self.outcome.li = []

            col_names = []
            for col in df.columns:
                # Get an entity if it already exists.
                entity = self.dependencies.get((PHENOTYPES, col))

                # Create the entity if it didn't exist.
                if entity is None:
                    entity = EntityIdentifier()
                    self.dependencies[(PHENOTYPES, col)] = entity

                # If the entity is not in the predictors, we should analyze it
                # in the pheWAS.
                if entity not in self.predictors:
                    self.outcome.li.append(entity)

                col_names.append(entity.id)

            df.columns = col_names

        else:
            phen_keys = [
                k[1] for k, v in self.dependencies.items()
                if k[0] == PHENOTYPES
            ]

            df = phenotypes.get_phenotypes(phen_keys)
            df.columns = [
                self.dependencies[(PHENOTYPES, k)].id for k in phen_keys
            ]

        # Extract the genotype dependencies.
        markers = self.get_tested_variants()

        # In GWAS we handle the SNPs differently, but the modelspec adds
        # variants that are specified explicitly directly to the phenotypes
        # dataframe.
        if markers is not SNPs:
            for entity, marker in markers:
                entity_id = self.dependencies[("GENOTYPES", marker)]

                try:
                    g = genotypes.get_variant_by_name(marker)
                except (KeyError, ValueError):
                    raise ValueError(
                        "Could not find '{}' in genotypes container."
                        "".format(marker)
                    )

                # Checking we have a single value and using it
                if len(g) != 1:
                    raise ValueError("{}: invalid marker name".format(marker))
                g = g.pop()

                # Rename the genotypes column before joining.
                df = df.join(
                    pd.Series(g.genotypes, index=genotypes.get_samples(),
                              name=entity_id.id),
                    how="inner",
                )

                if df.shape[0] == 0:
                    raise ValueError(
                        "No sample left after joining. Perhaps the sample IDs "
                        "in the genotypes and phenotypes containers are "
                        "different."
                    )

                # Compute the maf.
                maf, minor, major, flip = get_maf(
                    df[entity_id.id], g.coded, g.reference,
                )

                if flip:
                    df.loc[:, entity_id.id] = 2 - df.loc[:, entity_id.id]

                # Also bind the EntityIdentifier in case we need to compute
                # a GRS.
                entity_id.bind(df[entity_id.id])

                # And save the variant metadata.
                self.variant_metadata[entity.id] = {
                    "name": marker, "chrom": g.variant.chrom, "pos":
                    g.variant.pos, "minor": minor, "major": major, "maf": maf
                }

        # Apply transformations.
        df = self._apply_transformations(df)

        # Only keep predictors and outcomes.
        keep_cols = self._filter_columns(df)

        # Adding the intercept
        if not self.no_intercept:
            keep_cols.append("intercept")
            df["intercept"] = 1

        return df[keep_cols]

    def _apply_transformations(self, df):
        for action, source, target, params in self.transformations:
            f = transformation_handler.handlers[action]
            res = f(df, source, *params)

            # Some transformations return multiple columns. We create all the
            # relevant columns in the dataframe.
            if isinstance(res, dict):
                for key, col in res.items():
                    df["{}:{}".format(target.id, key)] = col

            # In most cases, transformations return a single array. We set it
            # under the target ID.
            else:
                df[target.id] = res

        return df

    def _filter_columns(self, df):
        if isinstance(self.outcome, PheWAS):
            keep_cols = [
                v.id for v in self.outcome.li if v not in self.predictors
            ]
        else:
            keep_cols = [v.id for v in self.outcome.values()]

        for pred in self.predictors:
            if pred is SNPs:
                continue

            if not isinstance(pred, EntityIdentifier):
                raise ValueError(
                    "Predictors are expected to be entity identifiers (and "
                    "'{}' is of type {}).".format(pred, type(pred))
                )

            # Some identifiers might have multiple associated columns (e.g.
            # factor transformation).
            for col in df.columns:
                if col.startswith(pred.id):
                    keep_cols.append(col)

        if self.stratify_by is not None:
            for col in self.stratify_by:
                if not isinstance(col, EntityIdentifier):
                    raise ValueError(
                        "Statification variables are expected to be entity "
                        "identifiers (and '{}' is of type {})."
                        "".format(col, type(col))
                    )
                keep_cols.append(col.id)

        return keep_cols


@transformation_handler("LOG10")
def _log10(data, entity):
    return np.log10(data[entity.id])


@transformation_handler("LN")
def _ln(data, entity):
    return np.log(data[entity.id])


@transformation_handler("ENCODE_FACTOR")
def _encode_factor(data, entity):
    out = {}

    v = data[entity.id]
    nulls = v.isnull()

    # Pandas category.
    levels = None
    if hasattr(v, "cat"):
        levels = sorted(v.cat.categories)
    else:
        levels = sorted(v[~nulls].unique())

    for i, level in enumerate(levels):
        # First level is the reference.
        if i == 0:
            continue
        level_name = "level.{}".format(level)
        out[level_name] = (v == level).astype(float)
        out[level_name][nulls] = np.nan

    return out


@transformation_handler("POW")
def _pow(data, entity, power):
    return np.power(data[entity.id], power)


@transformation_handler("INTERACTION")
def _interaction(data, *entities):
    # Finding all the columns for all the targets
    column_names = tuple(
        tuple(name for name in data.columns if name.startswith(entity.id))
        for entity in entities
    )

    # Finding the level column names if there are factors
    factor_levels = tuple(
        tuple(name[len(entity.id)+1:] for name in names)
        for names, entity in zip(column_names, entities)
    )

    # Creating the final columns
    out = {}
    for cols, level_names in zip(itertools.product(*column_names),
                                 itertools.product(*factor_levels)):
        # Getting the key (for factors, if present)
        key = re.sub(":{2,}", "", ":".join(level_names).strip(":"))

        # Computing the multiplication of all the columns
        mult = functools.reduce(np.multiply, (data[col] for col in cols))

        # If the key is empty, it means there are no factors in the interaction
        # term, hence there is only one resulting column and we return it
        if key == "":
            return mult

        out[key] = mult

    return out


def _reset():
    TransformationManager.transformations = []
    DependencyManager.dependencies = {}


result = ResultMetaclass()

phenotypes = DependencyManager("PHENOTYPES")
genotypes = DependencyManager("GENOTYPES")

factor = TransformationManager("ENCODE_FACTOR")
log10 = TransformationManager("LOG10")
ln = TransformationManager("LN")
pow = TransformationManager("POW")
interaction = TransformationManager("INTERACTION")
