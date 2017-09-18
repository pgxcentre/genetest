"""
Utilities to build statistical models.
"""

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import itertools
import functools

import numpy as np
import pandas as pd

from ..statistics import model_map
from ..statistics.descriptive import get_maf

from geneparse.utils import genotype_to_df


__all__ = ["SNPs", "phenotypes", "genotypes", "Phenotype", "Genotype",
           "Factor", "Interaction", "Ln", "Log10", "ModelSpec", "Pow"]


SNPs = "SNPs"


class Variable(object):
    def __init__(self, name):
        self.name = name
        self.columns = (name, )

    def get_data(self, phenotypes, genotypes, cache):
        raise NotImplemented()

    def __repr__(self):
        return "{}:{}".format(self.__class__.__name__, self.name)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        is_equal = (
            self.__class__ is other.__class__ and str(self) == str(other)
        )
        if is_equal:
            other.match = self
        return is_equal


class Phenotype(Variable):
    def get_data(self, phenotypes, genotypes, cache):
        # Checking in the cache
        if self in cache:
            return cache[self]

        # Caching and returning a DataFrame
        df = phenotypes.get_phenotypes([self.name])
        cache[self] = df
        return df


class Genotype(Variable):
    def get_genotype(self, phenotypes, genotypes):
        li = genotypes.get_variant_by_name(self.name)
        if len(li) == 0:
            raise ValueError("Could not find variant: {}".format(self.name))
        elif len(li) == 1:
            g = li[0]

            # Getting the MAF, minor and major allele
            maf, minor, major, flip = get_maf(
                g.genotypes, g.coded, g.reference,
            )

            # Flipping, if required
            if flip:
                g.genotypes = 2 - g.genotypes

            # Saving the variants metadata
            self.metadata = {
                "name": self.name, "chrom": str(g.variant.chrom),
                "pos": g.variant.pos, "minor": minor, "major": major,
                "maf": maf, "coded": minor,
            }

            return g
        else:
            raise NotImplementedError(
                "Multi-allelic variants are not yet handled ({})."
                "".format(self.name)
            )

    def get_data(self, phenotypes, genotypes, cache, allele_string=False):
        # Checking the cache
        if self in cache:
            return cache[self]

        # Caching and returning a DataFrame
        df = genotype_to_df(
            self.get_genotype(phenotypes, genotypes),
            samples=genotypes.get_samples(),
            as_string=allele_string
        )
        cache[self] = df
        return df


class Transformation(object):
    def __init__(self, *args, **kwargs):
        raise NotImplemented()

    def __call__(self, phenotypes, genotypes, cache):
        raise NotImplemented()

    def __hash__(self):
        raise NotImplemented()

    def __eq__(self, other):
        raise NotImplemented()


class Factor(Transformation):
    def __init__(self, entity, name=None):
        self.entity = entity
        self.name = name

    def __hash__(self):
        return hash("{}:{}".format(self.__class__.__name__, self.entity))

    def __eq__(self, other):
        is_equal = (
            self.__class__ is other.__class__ and self.entity == other.entity
        )
        if is_equal:
            other.match = self
        return is_equal

    def __call__(self, phenotypes, genotypes, cache):
        # Checking the cache
        if self in cache:
            return cache[self]

        # Special case where we want to encode genotypes with homozygous of the
        # reference allele is the reference factor
        df = None
        if isinstance(self.entity, Genotype):
            g = self.entity.get_genotype(phenotypes, genotypes)

            df = _make_factor_from_genotype(
                g, samples=genotypes.get_samples()
            )

        else:
            df = self.entity.get_data(phenotypes, genotypes, cache)

        assert df.shape[1] == 1

        # Take the unique levels and make the dummy variables (non-null values)
        nulls = df.iloc[:, 0].isnull()
        levels = sorted(df.loc[~nulls, self.entity.name].unique())

        # Computing the levels
        results = {}
        self.columns = []
        for i, level in enumerate(levels):
            if i == 0:
                continue

            # The name of the column
            col_name = "{}:{}".format(self.name, level)
            if self.name is None:
                col_name = "factor({}):{}".format(self.entity.name, level)

            r = (df.iloc[:, 0] == level).astype(float)
            r[nulls] = np.nan
            results[col_name] = r
            self.columns.append(col_name)

        # Changing the columns to a tuple
        self.columns = tuple(self.columns)

        # Caching and returning a DataFrame
        df = pd.DataFrame(results)
        cache[self] = df
        return df


class Pow(Transformation):
    def __init__(self, entity, power, name=None):
        self.entity = entity
        self.power = power
        self.name = name

    def __hash__(self):
        return hash("{}:{}:{}".format(
            self.__class__.__name__, self.entity, self.power
        ))

    def __eq__(self, other):
        is_equal = (
            self.__class__ is other.__class__ and self.entity == other.entity
            and self.power == other.power
        )
        if is_equal:
            other.match = self
        return is_equal

    def __call__(self, phenotypes, genotypes, cache):
        # Checking the cache
        if self in cache:
            return cache[self]

        # Getting the data
        df = self.entity.get_data(phenotypes, genotypes, cache).pow(self.power)

        assert df.shape[1] == 1
        d = df.iloc[:, 0]

        # The name of the column
        col_name = self.name
        if col_name is None:
            col_name = "pow({},{})".format(self.entity.name, self.power)

        # Caching and returning a DataFrame
        df = pd.DataFrame({col_name: d})
        cache[self] = df
        self.columns = (col_name, )
        return df


class Ln(Transformation):
    def __init__(self, entity, name=None):
        self.entity = entity
        self.name = name

    def __hash__(self):
        return hash("{}:{}".format(self.__class__.__name__, self.entity))

    def __eq__(self, other):
        is_equal = (
            self.__class__ is other.__class__ and self.entity == other.entity
        )
        if is_equal:
            other.match = self
        return is_equal

    def __call__(self, phenotypes, genotypes, cache):
        # Checking the cache
        if self in cache:
            return cache[self]

        # Getting the data
        df = self.entity.get_data(phenotypes, genotypes, cache)

        assert df.shape[1] == 1

        # The name of the column
        col_name = self.name
        if col_name is None:
            col_name = "ln({})".format(self.entity.name)

        # Creating the new DataFrame
        df = pd.DataFrame({col_name: np.log(df.iloc[:, 0].values)},
                          index=df.index)

        # Caching and returning a DataFrame
        cache[self] = df
        self.columns = (col_name, )
        return df


class Log10(Transformation):
    def __init__(self, entity, name=None):
        self.entity = entity
        self.name = name

    def __hash__(self):
        return hash("{}:{}".format(self.__class__.__name__, self.entity))

    def __eq__(self, other):
        is_equal = (
            self.__class__ is other.__class__ and self.entity == other.entity
        )
        if is_equal:
            other.match = self
        return is_equal

    def __call__(self, phenotypes, genotypes, cache):
        # Checking the cache
        if self in cache:
            return cache[self]

        # Getting the data
        df = self.entity.get_data(phenotypes, genotypes, cache)

        assert df.shape[1] == 1

        # The name of the column
        col_name = self.name
        if col_name is None:
            col_name = "log10({})".format(self.entity.name)

        # Creating the new DataFrame
        df = pd.DataFrame({col_name: np.log10(df.iloc[:, 0].values)},
                          index=df.index)

        # Caching and returning a DataFrame
        cache[self] = df
        self.columns = (col_name, )
        return df


class Interaction(Transformation):
    def __init__(self, *entities, name=None):
        self.entities = entities
        self.name = name

    def __hash__(self):
        entity_names = [
            v.name if isinstance(v, Variable) else v.entity.name
            for v in self.entities
        ]
        return hash("{}:{}".format(
            self.__class__.__name__, ":".join(entity_names)
        ))

    def __eq__(self, other):
        is_equal = (
            self.__class__ is other.__class__
            and self.entities == other.entities
        )
        if is_equal:
            other.match = self
        return is_equal

    def __call__(self, phenotypes, genotypes, cache):
        # Finding all the combinations of 2, 3, ..., n items
        combinations = []
        for nb_term in range(2, len(self.entities) + 1):
            combinations.extend(itertools.combinations(self.entities, nb_term))

        # Computing each of the columns
        results = {}
        self.columns = []
        for combination in combinations:
            available_columns = []
            dfs = []
            for entity in combination:
                # Getting the original entity in the cache
                if entity in cache:
                    # If the entity doesn't have the attribute 'match', it's
                    # because we have the real entity
                    if hasattr(entity, "match"):
                        entity = entity.match

                df = None
                if isinstance(entity, Transformation):
                    df = entity(phenotypes, genotypes, cache)
                else:
                    df = entity.get_data(phenotypes, genotypes, cache)

                    if isinstance(entity, Genotype):
                        # Fixing for repeated measurements
                        df = _fix_repeated_measurements(df, phenotypes)

                # Adding the columns and the DataFrame to the list
                available_columns.append(entity.columns)
                dfs.append(df.sort_index())

            # Creating the final df
            df = pd.concat(dfs, axis=1, join="outer")

            # Creating the product
            for cols in itertools.product(*available_columns):
                # The column name
                col_name = None
                if self.name is None:
                    col_name = "inter({})".format(",".join(cols))
                else:
                    col_name = "{}({})".format(self.name, ",".join(cols))

                results[col_name] = functools.reduce(
                    np.multiply, (df[col] for col in cols)
                )
                self.columns.append(col_name)

        # Changing the columns to a tuple
        self.columns = tuple(self.columns)

        # Caching and returning a DataFrame
        df = pd.DataFrame(results)
        cache[self] = df
        return df


class GWASInteraction(Transformation):
    def __init__(self, *entities, name=None):
        # Making sure SNPs isn't in the entity list
        self.entities = [entity for entity in entities if entity != SNPs]
        self.name = name

    def __hash__(self):
        entity_names = [
            v.name if isinstance(v, Variable) else v.entity.name
            for v in self.entities
        ]
        return hash("{}:{}".format(
            self.__class__.__name__, ":".join(entity_names)
        ))

    def __eq__(self, other):
        is_equal = (
            self.__class__ is other.__class__
            and self.entities == other.entities
        )
        if is_equal:
            other.match = self
        return is_equal

    def __call__(self, phenotypes, genotypes, cache):
        # Creating the entities
        entities = [SNPs] + self.entities

        # Finding all the combinations of 2, 3, ..., n items (adding the SNPs
        # entity
        combinations = []
        for nb_term in range(2, len(entities) + 1):
            combinations.extend(itertools.combinations(entities, nb_term))

        # TODO: Needs optimization, since interaction terms without SNPs will
        # be computed each time during the analysis...

        # Computing each of the columns
        self.columns = []
        self.interaction_cols = {}
        for combination in combinations:
            available_columns = []
            for entity in combination:
                # Getting the original entity in the cache
                if entity in cache:
                    # If the entity doesn't have the attribute 'match', it's
                    # because we have the real entity
                    if hasattr(entity, "match"):
                        entity = entity.match

                if entity == SNPs:
                    available_columns.append([SNPs])
                else:
                    available_columns.append(entity.columns)

            # Generating the columns for the product (in analysis)
            for cols in itertools.product(*available_columns):
                # The column name
                col_name = None
                if self.name is None:
                    col_name = "gwas_inter({})".format(",".join(cols))
                else:
                    col_name = "{}({})".format(self.name, ",".join(cols))

                self.interaction_cols[col_name] = cols
                self.columns.append(col_name)

        # Changing the list of columns to a tuple
        self.columns = tuple(self.columns)

    def get_gwas_interaction(self):
        return self.interaction_cols


def _make_factor_from_genotype(g, samples):
    """Encode genotypes as factors by using reference allele as reference."""
    name = g.variant.name if g.variant.name else "genotypes"
    df = pd.DataFrame(g.genotypes, index=samples, columns=[name])

    df["alleles"] = None

    hard_calls = df[name].round()
    df.loc[hard_calls == 0, "alleles"] = "0-{0}/{0}".format(g.reference)
    df.loc[hard_calls == 1, "alleles"] = "1-{0}/{1}".format(g.reference,
                                                            g.coded)
    df.loc[hard_calls == 2, "alleles"] = "2-{0}/{0}".format(g.coded)

    df = df[["alleles"]]
    df.columns = [name]

    return df


class ModelSpec(object):
    def __init__(self, outcome, predictors, test, no_intercept=False,
                 stratify_by=None, cache=None):

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

        # The dependencies
        self.dependencies = list(self.outcome.values()) + self.predictors

        if stratify_by is not None:
            self.dependencies.extend(stratify_by)

        # The cache
        self.cache = cache

        # SNP metadata is stored in the modelspec because it is obtained as a
        # consequence of building the data matrix.
        # Hence, it is more efficient to cache it than to request it.
        self.variant_metadata = {}

        # Is this a GWAS?
        self.is_gwas = False

        # Is there GWAS interaction involved?
        self.has_gwas_interaction = False

    def _clean_outcome(self, outcome):
        if not isinstance(outcome, dict):
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

    def create_data_matrix(self, phenotypes, genotypes):
        if self.cache is None:
            self.cache = {}
            self._create_entities(phenotypes, genotypes)

        # Merging all DataFrames from the different entities
        dfs = []
        for entity in self.dependencies:
            # GWAS interaction
            if isinstance(entity, GWASInteraction):
                # This is a special case (GWAS interaction)
                self.has_gwas_interaction = True
                self.gwas_interaction = entity.get_gwas_interaction()

            # GWAS
            elif entity == SNPs:
                self.is_gwas = True

            else:
                # Getting the data from the cache
                df = self.cache[entity]

                if isinstance(entity, Genotype):
                    # This is a genotype, so we save the metadata
                    self.variant_metadata[entity.name] = entity.metadata
                    df = _fix_repeated_measurements(df, phenotypes)

                dfs.append(df.sort_index())
        df = pd.concat(dfs, axis=1, join="outer")

        # Adding the intercept
        if not self.no_intercept:
            df["intercept"] = 1

        # Checking of all samples have at least one missing value. If it's the
        # case, it might be because there were no intersect between samples and
        # genotypes... Since we usually drop NaN before analysis, this will
        # give an empty DataFrame...
        if df.isnull().any(axis=1).all():
            raise ValueError(
                "No sample left after joining. Perhaps the sample IDs in the "
                "genotypes and phenotypes containers are different."
            )

        return df

    def _create_entities(self, phenotypes, genotypes):
        """Filling the cache according to the dependencies of the model."""
        for entity in self.dependencies:
            # Is the entity already in the cache?
            if (entity not in self.cache) and (entity != SNPs):
                # Populating the cache
                if isinstance(entity, Transformation):
                    entity(phenotypes, genotypes, self.cache)
                else:
                    entity.get_data(phenotypes, genotypes, self.cache)

    def get_entity(self, entity):
        """Returns the original entity."""
        if entity in self.cache:
            return entity.match
        else:
            raise KeyError(entity)


class _VariableFactory(object):
    def __init__(self, variable_type):
        if variable_type == "PHENOTYPES":
            self.cls = Phenotype
        elif variable_type == "GENOTYPES":
            self.cls = Genotype
        else:
            raise ValueError("{}: invalid variable type".format(variable_type))

    def __getitem__(self, name):
        return self.cls(name)

    __getattr__ = __getitem__


phenotypes = _VariableFactory("PHENOTYPES")
genotypes = _VariableFactory("GENOTYPES")


class Result(object):
    def __init__(self, entity):
        self.path = [entity]

    def __getitem__(self, key):
        self.path.append(key)
        return self

    def get(self, results):
        cur = results

        for field in self.path:
            if (isinstance(field, Variable)
                    or isinstance(field, Transformation)):
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


result = ResultMetaclass()


def _fix_repeated_measurements(df, phenotypes):
    """Fixes genotypes for repeated measurements."""
    # TODO: Test this...
    if phenotypes is not None and phenotypes.is_repeated():
        nbs = phenotypes.get_nb_repeats()
        new_index = list(itertools.chain(*[
            [s]*nbs[s] for s in df.index
        ]))
        return df.loc[new_index, :]

    else:
        return df
