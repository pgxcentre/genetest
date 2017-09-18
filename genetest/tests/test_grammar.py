

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import re
import unittest
import functools

import numpy as np
import pandas as pd
from scipy.stats import binom

from geneparse import parsers

from .. import modelspec as spec
from ..phenotypes.dataframe import DataFrameContainer
from ..statistics.models.linear import StatsLinear


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestGrammar(unittest.TestCase):
    """Tests the grammar for the ModelSpec class."""
    @classmethod
    def setUpClass(cls):
        # Creating random data
        cls.data = pd.DataFrame(
            dict(
                pheno=np.random.randint(1, 100, 100),
                var1=np.random.randint(1, 100, 100),
                var2=np.random.rand(100),
                var3=["x{}".format(i) for i in np.random.randint(0, 3, 100)],
                var4=["y{}".format(i) for i in np.random.randint(0, 2, 100)],
                var5=np.random.randint(0, 4, 100),
                snp=binom.rvs(2, 0.3, size=100),
            ),
            index=["sample_{}".format(i+1) for i in range(100)],
        )

        # Changing one factor to categorical data
        cls.data.loc[:, "var5"] = cls.data.var5.astype("category")

        # Creating the dummy phenotype container
        phenotypes = ["pheno"] + ["var{}".format(i+1) for i in range(5)]
        cls.phenotypes = DataFrameContainer(cls.data[phenotypes].copy())

        # Creating the dummy genotype container
        map_info = pd.DataFrame(
            {"chrom": ["3"],
             "pos": [1234],
             "a1": ["T"],
             "a2": ["C"]},
            index=["snp"],
        )
        cls.genotypes = parsers["dataframe"](
            dataframe=cls.data[["snp"]].copy(),
            map_info=map_info,
        )

    def setUp(self):
        # Reordering the columns and the rows of the phenotype data frame
        self.phenotypes._phenotypes = self.phenotypes._phenotypes.iloc[
            np.random.permutation(self.phenotypes._phenotypes.shape[0]),
            np.random.permutation(self.phenotypes._phenotypes.shape[1])
        ]

    def test_simple_formula(self):
        """Tests a simple ModelSpec object."""
        # Parsing the formula and removing the conditions
        model = spec.parse_formula("pheno ~ g(snp) + var1 + var2")
        model["test"] = StatsLinear
        del model["conditions"]

        # Creating the model
        modelspec = spec.ModelSpec(**model)

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 5), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
        predictors = [spec.genotypes.snp, spec.phenotypes.var1,
                      spec.phenotypes.var2]
        for predictor in predictors:
            # Getting the name of the predictor
            name = predictor.name

            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, predictor.columns[0]].values,
                self.data[name].values,
                err_msg="The predictor '{}' is not as expected".format(name),
            )

    def test_pow(self):
        """Tests a power transformation."""
        # Parsing the formula and removing the conditions
        model = spec.parse_formula("pheno ~ g(snp) + pow(var1, 4)")
        model["test"] = StatsLinear
        del model["conditions"]

        # Creating the model
        modelspec = spec.ModelSpec(**model)

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 4), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
        predictors = [
            modelspec.get_entity(spec.genotypes.snp),
            modelspec.get_entity(spec.pow(spec.phenotypes.var1, 4)),
        ]
        for predictor, power, name in zip(predictors, (1, 4), ("snp", "var1")):
            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, predictor.columns[0]].values,
                self.data[name].values**power,
                err_msg="The predictor '{}**{}' is not as "
                        "expected".format(name, power),
            )

    def test_log(self):
        """Tests log transformations (log10 and ln)."""
        # Parsing the formula and removing the conditions
        model = spec.parse_formula(
            "pheno ~ g(snp) + var1 + log10(var1) + ln(var1)"
        )
        model["test"] = StatsLinear
        del model["conditions"]

        # Creating the model
        modelspec = spec.ModelSpec(**model)

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 6), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = modelspec.get_entity(spec.phenotypes.pheno).columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
        for predictor in (spec.genotypes.snp, spec.phenotypes.var1):
            # Getting the entity and name
            entity = modelspec.get_entity(predictor)

            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, entity.columns[0]].values,
                self.data[entity.name].values,
                err_msg="The predictor '{}' is not as expected"
                        "".format(entity.name),
            )

        # Checking the log transform
        predictors = (
            modelspec.get_entity(spec.log10(spec.phenotypes.var1)),
            modelspec.get_entity(spec.ln(spec.phenotypes.var1)),
        )
        for predictor, transform, name in zip(predictors, (np.log10, np.log),
                                              ("var1", "var1")):
            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, predictor.columns[0]].values,
                transform(self.data[name].values),
                err_msg="The predictor '{}({})' is not as "
                        "expected".format(transform.__name__, name),
            )

    def test_encode_factor(self):
        """Tests with factor."""
        # Parsing the formula and removing the conditions
        model = spec.parse_formula(
            "pheno ~ factor(var3) + factor(var4) + factor(var5)"
        )
        model["test"] = StatsLinear
        del model["conditions"]

        # Creating the model
        modelspec = spec.ModelSpec(**model)

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 8), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = modelspec.get_entity(spec.phenotypes.pheno).columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the factors
        var_names = ("var3", "var4", "var5")
        predictor_zipped = zip(
            tuple(modelspec.get_entity(spec.factor(spec.phenotypes[p]))
                  for p in var_names),
            var_names,
            (("x1", "x2"), ("y1", ), (1, 2, 3)),
        )
        for predictor, name, levels in predictor_zipped:
            for i, level in enumerate(levels):
                # Getting the name of the column containing the level data
                matrix_col = predictor.columns[i]
                self.assertTrue(matrix_col.endswith(":" + str(level)))

                # Comparing the results
                np.testing.assert_array_equal(
                    matrix.loc[self.data.index, matrix_col].values,
                    (self.data[name] == level).astype(float).values,
                    err_msg="The predictor '{}' (level '{}') is not as "
                            "expected".format(name, level),
                )

    def test_simple_interaction(self):
        """Tests simple interaction."""
        # Parsing the formula and removing the conditions
        model = spec.parse_formula(
            "pheno ~ g(snp) + var1 + g(snp)*var1"
        )
        model["test"] = StatsLinear
        del model["conditions"]

        # Creating the model
        modelspec = spec.ModelSpec(**model)

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 5), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = modelspec.get_entity(spec.phenotypes.pheno).columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
        for predictor in (spec.genotypes.snp, spec.phenotypes.var1):
            # Getting the name of the predictor
            predictor = modelspec.get_entity(predictor)
            name = predictor.name

            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, predictor.columns[0]].values,
                self.data[name].values,
                err_msg="The predictor '{}' is not as expected".format(name),
            )

        # Checking the interaction
        interaction = modelspec.get_entity(
            spec.interaction(spec.genotypes.snp, spec.phenotypes.var1),
        )
        np.testing.assert_array_equal(
            matrix.loc[self.data.index, interaction.columns[0]].values,
            (self.data.snp * self.data.var1).values,
            err_msg="The interaction 'snp*var1' is not as expected",
        )

    def test_multiple_interaction(self):
        """Tests simple interaction."""
        # Parsing the formula and removing the conditions
        model = spec.parse_formula(
            "pheno ~ g(snp) + var1 + var2 + g(snp)*var1*var2"
        )
        model["test"] = StatsLinear
        del model["conditions"]

        # Creating the model
        modelspec = spec.ModelSpec(**model)

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 9), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = modelspec.get_entity(spec.phenotypes.pheno).columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
        predictors = (
            modelspec.get_entity(spec.genotypes.snp),
            modelspec.get_entity(spec.phenotypes.var1),
            modelspec.get_entity(spec.phenotypes.var2),
        )
        for predictor in predictors:
            # Getting the name of the predictor
            name = predictor.name

            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, predictor.columns[0]].values,
                self.data[name].values,
                err_msg="The predictor '{}' is not as expected".format(name),
            )

        # Checking the interaction
        interaction = modelspec.get_entity(
            spec.interaction(spec.genotypes.snp, spec.phenotypes.var1,
                             spec.phenotypes.var2),
        )
        for column in interaction.columns:
            # Getting the variables
            variables = re.search(
                r"inter\(((\S+,)+\S+)\)", column
            ).group(1).split(",")

            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, column].values,
                functools.reduce(np.multiply,
                                 [self.data[col] for col in variables]),
                err_msg="The interaction '{}' is not as expected"
                        "".format(column),
            )

    def test_factor_interaction(self):
        """Tests interaction between a term and a factor."""
        # Parsing the formula and removing the conditions
        model = spec.parse_formula(
            "pheno ~ g(snp) + factor(var5) + g(snp)*factor(var5)"
        )
        model["test"] = StatsLinear
        del model["conditions"]

        # Creating the model
        modelspec = spec.ModelSpec(**model)

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 9), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = modelspec.get_entity(spec.phenotypes.pheno).columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictor
        predictor = modelspec.get_entity(spec.genotypes.snp)
        np.testing.assert_array_equal(
            matrix.loc[self.data.index, predictor.columns[0]].values,
            self.data.snp.values,
            err_msg="The predictor 'snp' is not as expected",
        )

        # For all level:
        factor = modelspec.get_entity(spec.factor(spec.phenotypes.var5))
        interaction = modelspec.get_entity(
            spec.interaction(spec.genotypes.snp, factor),
        )
        for i, level in enumerate((1, 2, 3)):
            # Creating the expected level
            expected_factor_values = (self.data.var5 == level).astype(float)

            # Checking the factor
            col = factor.columns[i]
            self.assertTrue(col.endswith(":" + str(level)))
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, col].values,
                expected_factor_values.values,
                err_msg="The predictor 'var5' (level '{}') is not as "
                        "expected".format(level),
            )

            # Checking the interaction
            col = interaction.columns[i]
            self.assertTrue(col.endswith(":" + str(level) + ")"))
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, col].values,
                (expected_factor_values * self.data.snp).values,
                err_msg="The interaction 'snp*var5' (var5 level '{}') is not "
                        "as expected".format(level),
            )

    def test_complex_interaction(self):
        """Tests a complex interaction between terms and factors."""
        # Parsing the formula and removing the conditions
        model = spec.parse_formula(
            "pheno ~ g(snp) + var2 + factor(var3) + factor(var4) + "
            "        factor(var5) + "
            "        g(snp)*var2*factor(var3)*factor(var4)*factor(var5)"
        )
        model["test"] = StatsLinear
        del model["conditions"]

        # Creating the model
        modelspec = spec.ModelSpec(**model)

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 97), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = modelspec.get_entity(spec.phenotypes.pheno).columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
        for predictor in (spec.genotypes.snp, spec.phenotypes.var2):
            # Getting the entity
            predictor = modelspec.get_entity(predictor)

            # Getting the name of the predictor
            name = predictor.name

            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, predictor.columns[0]].values,
                self.data[name].values,
                err_msg="The predictor '{}' is not as expected".format(name),
            )

        # Checking the factors
        var_names = ("var3", "var4", "var5")
        predictor_zipped = zip(
            tuple(spec.factor(spec.phenotypes[name]) for name in var_names),
            var_names,
            (("x1", "x2"), ("y1", ), (1, 2, 3)),
        )
        for predictor, name, levels in predictor_zipped:
            # Getting the official entity
            predictor = modelspec.get_entity(predictor)

            for col_name, level in zip(predictor.columns, levels):
                # Checking the level
                self.assertTrue(col_name.endswith(":" + str(level)))

                # Comparing the factor
                np.testing.assert_array_equal(
                    matrix.loc[self.data.index, col_name].values,
                    (self.data[name] == level).astype(float).values,
                    err_msg="The predictor '{}' (level '{}') is not as "
                            "expected".format(name, level),
                )

        # Checking the interaction
        interaction = modelspec.get_entity(spec.interaction(
            spec.genotypes.snp, spec.phenotypes.var2,
            spec.factor(spec.phenotypes.var3),
            spec.factor(spec.phenotypes.var4),
            spec.factor(spec.phenotypes.var5),
        ))
        self.assertEqual(87, len(set(interaction.columns)))
        for col_name in interaction.columns:
            # Getting the list of variables
            variables = re.search(
                r"inter\(((\S+,)+\S+)\)", col_name
            ).group(1).split(",")

            expected = np.ones(self.data.shape[0], dtype=float)
            for variable in variables:
                # Do we have a factor?
                if variable.startswith("factor"):
                    # Getting the variable name and level
                    name = re.search(r"factor\((\S+)\)", variable).group(1)
                    level = variable.split(":")[-1]

                    if name == "var5":
                        level = int(level)

                    # Multiplying
                    expected *= (self.data.loc[:, name] == level).astype(float)

                else:
                    # Normal variable
                    expected *= self.data.loc[:, variable]

            # Comparing with the original value
            np.testing.assert_array_almost_equal(
                matrix.loc[self.data.index, col_name].values,
                expected,
                err_msg="The interaction '{}' is not as expected"
                        "".format(col_name)
            )
