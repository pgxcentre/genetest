

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import re
import unittest
import itertools
from os import path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import binom

from pyplink import PyPlink
from geneparse import parsers

from .. import modelspec as spec
from ..phenotypes.dummy import _DummyPhenotypes


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestModelSpec(unittest.TestCase):
    """Tests the 'ModelSpec' class."""
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
        cls.phenotypes = _DummyPhenotypes()
        cls.phenotypes.data = cls.data[phenotypes].copy()

        # Creating a temporary directory
        cls.tmp_dir = TemporaryDirectory(prefix="genetest_")

        # The plink file prefix
        cls.plink_prefix = path.join(cls.tmp_dir.name, "input")

        # Permuting the sample to add a bit of randomness
        new_sample_order = np.random.permutation(cls.data.index)

        # Creating the BED file
        with PyPlink(cls.plink_prefix, "w") as bed:
            bed.write_genotypes(cls.data.loc[new_sample_order, "snp"])

        # Creating the BIM file
        with open(cls.plink_prefix + ".bim", "w") as bim:
            print(1, "snp", 0, 1, "B", "A", sep="\t", file=bim)

        # Creating the FAM file
        with open(cls.plink_prefix + ".fam", "w") as fam:
            for sample in new_sample_order:
                print(sample, sample, 0, 0, 0, -9, file=fam)

        # Creating the genotype parser
        cls.genotypes = parsers["plink"](cls.plink_prefix)

    @classmethod
    def tearDownClass(cls):
        cls.genotypes.close()
        cls.tmp_dir.cleanup()

    def setUp(self):
        # Reordering the columns and the rows of the phenotype data frame
        self.phenotypes.data = self.phenotypes.data.iloc[
            np.random.permutation(self.phenotypes.data.shape[0]),
            np.random.permutation(self.phenotypes.data.shape[1])
        ]

    def test_simple_modelspec(self):
        """Tests a simple ModelSpec object."""
        # Creating the model specification
        predictors = [spec.genotypes.snp, spec.phenotypes.var1,
                      spec.phenotypes.var2]
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=predictors,
            test="linear",
        )

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 5), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the index
        self.assertEqual(set(self.data.index), set(matrix.index),
                         "Samples are not the same")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
        for predictor in predictors:
            # Getting the name of the predictor
            name = predictor.name

            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, predictor.columns[0]].values,
                self.data[name].values,
                err_msg="The predictor '{}' is not as expected".format(name),
            )

    def test_no_sample_intersect(self):
        """Tests when no intersect between phenotypes and genotypes."""
        # Creating a new phenotype container
        phenotypes = _DummyPhenotypes()
        phenotypes.data = self.data[["pheno", "var1"]].copy()
        phenotypes.data.index = ["s" + s for s in phenotypes.data.index]

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=[spec.genotypes.snp, spec.phenotypes.var1],
            test="linear",
        )

        # Gathering the observed matrix
        with self.assertRaises(ValueError):
            modelspec.create_data_matrix(phenotypes, self.genotypes)

    def test_smaller_sample_intersect(self):
        """Tests when the sample intersect is smaller between containers."""
        # Choosing 10 samples to exclude from the dataset
        to_exclude = np.random.choice(self.data.index, 10, replace=False)

        # Removing 5 samples from the phenotypes
        phenotypes = _DummyPhenotypes()
        phenotypes.data = self.data.drop(to_exclude[:5], axis=0)

        # Removing the next 5 for the genotypes
        plink_prefix = self.plink_prefix + "_less"
        geno_data = self.data.drop(to_exclude[5:], axis=0)
        with PyPlink(plink_prefix, "w") as bed:
            bed.write_genotypes(geno_data.snp)

        # Creating the BIM file
        with open(plink_prefix + ".bim", "w") as bim:
            print(1, "snp", 0, 1, "B", "A", sep="\t", file=bim)

        # Creating the FAM file
        with open(plink_prefix + ".fam", "w") as fam:
            for sample in geno_data.index:
                print(sample, sample, 0, 0, 0, -9, file=fam)

        # Creating the model specification
        predictors = [spec.genotypes.snp, spec.phenotypes.var1,
                      spec.phenotypes.var2]
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=predictors,
            test="linear",
        )

        # Gathering the observed matrix
        with parsers["plink"](plink_prefix) as genotypes:
            matrix = modelspec.create_data_matrix(phenotypes, genotypes)

        # Creating the copy of the data and setting to NaN the missing samples
        data = self.data.copy()
        data.loc[to_exclude[:5], ["pheno", "var1", "var2"]] = np.nan
        data.loc[to_exclude[5:], "snp"] = np.nan

        # Checking the shape of the matrix (should be as before)
        self.assertEqual((self.data.shape[0], 5), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the index
        self.assertEqual(set(data.index), set(matrix.index),
                         "Samples are not the same")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[data.index, outcome_col]
        self.assertTrue(
            outcomes.equals(data.pheno),
            "The outcomes are not as expected",
        )

        # Checking the predictors
        for predictor in predictors:
            # Getting the name of the predictor
            name = predictor.name

            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[data.index, predictor.columns[0]].values,
                data[name].values,
                err_msg="The predictor '{}' is not as expected".format(name),
            )

    def test_pow(self):
        """Tests a power transformation."""
        # Creating the model specification
        predictors = [spec.genotypes.snp, spec.pow(spec.phenotypes.var1, 3)]
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=predictors,
            test="linear",
        )

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 4), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the index
        self.assertEqual(set(self.data.index), set(matrix.index),
                         "Samples are not the same")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
        for predictor, power, name in zip(predictors, (1, 3), ("snp", "var1")):
            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, predictor.columns[0]].values,
                self.data[name].values**power,
                err_msg="The predictor '{}**{}' is not as "
                        "expected".format(name, power),
            )

    def test_log(self):
        """Tests log transformations (log10 and ln)."""
        # Creating the model specification
        predictors = [spec.genotypes.snp, spec.log10(spec.phenotypes.var1),
                      spec.ln(spec.phenotypes.var1)]
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=predictors,
            test="linear",
        )

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 5), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the index
        self.assertEqual(set(self.data.index), set(matrix.index),
                         "Samples are not the same")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
        for predictor in predictors[:1]:
            # Getting the name of the predictor
            name = predictor.name

            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, predictor.columns[0]].values,
                self.data[name].values,
                err_msg="The predictor '{}' is not as expected".format(name),
            )

        # Checking the log transform
        for predictor, transform, name in zip(predictors[1:],
                                              (np.log10, np.log),
                                              ("var1", "var1")):
            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, predictor.columns[0]].values,
                transform(self.data[name].values),
                err_msg="The predictor '{}**{}' is not as "
                        "expected".format(name, transform.__name__),
            )

    def test_encode_factor(self):
        """Tests with factor."""
        # Creating the model specification
        predictors = [spec.factor(spec.phenotypes.var3),
                      spec.factor(spec.phenotypes.var4),
                      spec.factor(spec.phenotypes.var5)]
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=predictors,
            test="linear",
        )

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 8), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the index
        self.assertEqual(set(self.data.index), set(matrix.index),
                         "Samples are not the same")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the factors
        predictor_zipped = zip(
            predictors,
            ("var3", "var4", "var5"),
            (("x1", "x2"), ("y1", ), (1, 2, 3)),
        )
        for predictor, name, levels in predictor_zipped:
            for matrix_col, level in zip(predictor.columns, levels):
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
        # Creating the model specification
        predictors = [spec.genotypes.snp, spec.phenotypes.var1]
        interaction = spec.interaction(spec.genotypes.snp,
                                       spec.phenotypes.var1)
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=predictors + [interaction],
            test="linear",
        )

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 5), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the index
        self.assertEqual(set(self.data.index), set(matrix.index),
                         "Samples are not the same")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
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
        np.testing.assert_array_equal(
            matrix.loc[self.data.index, interaction.columns[0]].values,
            (self.data.snp * self.data.var1).values,
            err_msg="The interaction 'snp*var1' is not as expected",
        )

    def test_multiple_interaction(self):
        """Tests simple interaction."""
        # Creating the model specification
        predictors = [spec.genotypes.snp, spec.phenotypes.var1,
                      spec.phenotypes.var2]
        interaction = spec.interaction(spec.genotypes.snp,
                                       spec.phenotypes.var1,
                                       spec.phenotypes.var2)
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=predictors + [interaction],
            test="linear",
        )

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 9), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the index
        self.assertEqual(set(self.data.index), set(matrix.index),
                         "Samples are not the same")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
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
        all_variables = (("snp", "var1"), ("snp", "var2"), ("var1", "var2"),
                         ("snp", "var1", "var2"))
        for variables in all_variables:
            # Creating the name of the column
            col_name = "inter({})".format(",".join(variables))
            self.assertTrue(col_name in matrix.columns)

            # Computing the expected values
            expected = self.data.loc[:, variables[0]]
            for i in range(1, len(variables)):
                expected = expected * self.data.loc[:, variables[i]]

            # Comparing with the expected results
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, col_name],
                expected,
                err_msg="The interaction '{}' is not as expected"
                        "".format(col_name),
            )

    def test_factor_interaction(self):
        """Tests interaction between a term and a factor."""
        # Creating the model specification
        factor = spec.factor(spec.phenotypes.var5)
        predictors = [spec.genotypes.snp, factor]
        interaction = spec.interaction(spec.genotypes.snp, factor)
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=predictors + [interaction],
            test="linear",
        )

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 9), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the index
        self.assertEqual(set(self.data.index), set(matrix.index),
                         "Samples are not the same")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictor
        np.testing.assert_array_equal(
            matrix.loc[self.data.index, spec.genotypes.snp.columns[0]].values,
            self.data.snp.values,
            err_msg="The predictor 'snp' is not as expected",
        )

        # Checking the factors
        for col_name in factor.columns:
            # Getting the level
            level = int(re.search(
                r"factor\(var5\):([0-9]+)", col_name
            ).group(1))

            # Creating the expected level
            expected_factor_values = (self.data.var5 == level).astype(float)

            # Checking the value
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, col_name].values,
                expected_factor_values.values,
                err_msg="The predictor 'var5' (level '{}') is not as "
                        "expected".format(level),
            )

        # Checking the interaction
        for col_name in interaction.columns:
            # Getting the level
            level = int(re.search(
                r"inter\(snp,factor\(var5\):([0-9]+)\)", col_name
            ).group(1))

            # Creating the expected level
            expected_factor_values = (self.data.var5 == level).astype(float)

            np.testing.assert_array_equal(
                matrix.loc[self.data.index, col_name].values,
                (expected_factor_values * self.data.snp).values,
                err_msg="The interaction 'snp*var5' (var5 level '{}') is not "
                        "as expected".format(level),
            )

    def test_complex_interaction(self):
        """Tests a complex interaction between terms and factors."""
        # Creating the model specification
        var3 = spec.factor(spec.phenotypes.var3)
        var4 = spec.factor(spec.phenotypes.var4)
        var5 = spec.factor(spec.phenotypes.var5)
        predictors = [spec.genotypes.snp, spec.phenotypes.var2, var3, var4,
                      var5]
        interaction = spec.interaction(spec.genotypes.snp,
                                       spec.phenotypes.var2, var3, var4, var5)
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=predictors + [interaction],
            test="linear",
        )

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(
            self.phenotypes, self.genotypes,
        )

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 97), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the index
        self.assertEqual(set(self.data.index), set(matrix.index),
                         "Samples are not the same")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictors
        for predictor in predictors[:2]:
            # Getting the name of the predictor
            name = predictor.name

            # Comparing the values
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, predictor.columns[0]].values,
                self.data[name].values,
                err_msg="The predictor '{}' is not as expected".format(name),
            )

        # Checking the factors
        predictor_zipped = zip(
            predictors[2:],
            ("var3", "var4", "var5"),
            (("x1", "x2"), ("y1", ), (1, 2, 3)),
        )
        for predictor, name, levels in predictor_zipped:
            # Checking the number of columns for the predictors
            self.assertEqual(len(levels), len(predictor.columns))

            for col_name in predictor.columns:
                # The level
                level = re.search(
                    r"factor\({}\):([a-z0-9]+)".format(name), col_name
                ).group(1)

                # Checking the name of the column
                self.assertTrue(col_name.endswith(":" + str(level)))

                # var5 is integer
                if name == "var5":
                    level = int(level)

                # Comparing the factor
                np.testing.assert_array_equal(
                    matrix.loc[self.data.index, col_name].values,
                    (self.data[name] == level).astype(float).values,
                    err_msg="The predictor '{}' (level '{}') is not as "
                            "expected".format(name, level),
                )

        # Checking the (unique) interaction
        self.assertEqual(87, len(set(interaction.columns)))
        for col_name in interaction.columns:
            # Getting the list of variables
            variables = re.search(
                r"inter\(((\S+,)+\S+)\)", col_name
            ).group(1).split(",")

            expected = np.ones(self.data.shape[0], dtype=float)
            for variable in variables:
                # We have a factor
                if variable.startswith("factor"):
                    # Getting the variable name and level
                    name = re.search(r"factor\((\S+)\)", variable).group(1)
                    level = variable.split(":")[-1]

                    if name == "var5":
                        level = int(level)

                    # Multiplying
                    expected *= (self.data.loc[:, name] == level).astype(float)

                # We have a normal variable
                else:
                    expected *= self.data.loc[:, variable]

            # Comparing with the original value
            np.testing.assert_array_almost_equal(
                matrix.loc[self.data.index, col_name].values,
                expected,
                err_msg="The interaction '{}' is not as expected"
                        "".format(col_name)
            )

    def test_gwas_interaction(self):
        """Test a simple GWAS interaction."""
        # Creating the model specification
        inter = spec.gwas_interaction(spec.phenotypes.var1)
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=[spec.SNPs, spec.phenotypes.var1, inter],
            test="linear",
        )

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(self.phenotypes, self.genotypes)

        # Checking we caught the GWAS interaction
        self.assertTrue(modelspec.has_gwas_interaction,
                        "The modelspec did not catch the GWAS interaction")

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 3), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the index
        self.assertEqual(set(self.data.index), set(matrix.index),
                         "Samples are not the same")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictor
        var1_col = spec.phenotypes.var1.columns[0]
        np.testing.assert_array_equal(
            matrix.loc[self.data.index, var1_col].values,
            self.data.var1.values,
            err_msg="The predictor 'var1' is not as expected.",
        )

        # Checking the resulting column for the GWAS interaction
        multiplication_dict = modelspec.gwas_interaction
        self.assertEqual(1, len(inter.columns))
        self.assertEqual(set(multiplication_dict.keys()), set(inter.columns))
        self.assertEqual(1, len(multiplication_dict),
                         "Wrong number of interaction multiplication")
        self.assertTrue(inter.columns[0] in multiplication_dict,
                        "Wrong result column for the GWAS interaction")

        # Checking the columns to multiply for the GWAS interaction
        multiplication_cols = multiplication_dict[inter.columns[0]]
        self.assertEqual(2, len(multiplication_cols),
                         "Wrong number of columns to multiply for the GWAS "
                         "interaction")
        self.assertEqual(
            ("SNPs", spec.phenotypes.var1.columns[0]), multiplication_cols,
            "Wrong column to multiply for the GWAS interaction",
        )

    def test_gwas_interaction_complex_category(self):
        """Test a simple GWAS interaction with a categorical value."""
        # Creating the model specification
        var3 = spec.factor(spec.phenotypes.var3)
        var5 = spec.factor(spec.phenotypes.var5)
        inter = spec.gwas_interaction(spec.phenotypes.var1, var3, var5)
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno,
            predictors=[spec.SNPs, spec.phenotypes.var1, var3, var5, inter],
            test="linear",
        )

        # Gathering the observed matrix
        matrix = modelspec.create_data_matrix(self.phenotypes, self.genotypes)

        # Checking we caught the GWAS interaction
        self.assertTrue(modelspec.has_gwas_interaction,
                        "The modelspec did not catch the GWAS interaction")

        # Checking the shape of the matrix
        self.assertEqual((self.data.shape[0], 8), matrix.shape,
                         "The observed matrix is not of the right shape")

        # Checking the index
        self.assertEqual(set(self.data.index), set(matrix.index),
                         "Samples are not the same")

        # Checking the intercept
        self.assertEqual([1], matrix.intercept.unique().tolist(),
                         "The intercept is not as expected")

        # Checking the outcome
        outcome_col = spec.phenotypes.pheno.columns[0]
        outcomes = matrix.loc[self.data.index, outcome_col]
        self.assertTrue(outcomes.equals(self.data.pheno),
                        "The outcomes are not as expected")

        # Checking the predictor (var1)
        var1_col = spec.phenotypes.var1.columns[0]
        np.testing.assert_array_equal(
            matrix.loc[self.data.index, var1_col].values,
            self.data.var1.values,
            err_msg="The predictor 'var1' is not as expected.",
        )

        # Checking the predictor (var3, all level)
        self.assertEqual(
            ("factor(var3):x1", "factor(var3):x2"),
            var3.columns,
            "Wrong columns for 'factor(var3)'",
        )
        for matrix_col, level in zip(var3.columns, ("x1", "x2")):
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, matrix_col].values,
                (self.data.var3 == level).astype(float).values,
                err_msg="The predictor 'var3' (level '{}') is not as "
                        "expected".format(level),
            )

        # Checking the predictor (var5, all level)
        self.assertEqual(
            ("factor(var5):1", "factor(var5):2", "factor(var5):3"),
            var5.columns,
            "Wrong columns for 'factor(var5)'",
        )
        for matrix_col, level in zip(var5.columns, (1, 2, 3)):
            np.testing.assert_array_equal(
                matrix.loc[self.data.index, matrix_col].values,
                (self.data.var5 == level).astype(float).values,
                err_msg="The predictor 'var5' (level '{}') is not as "
                        "expected".format(level),
            )

        # Checking the resulting column for the GWAS interaction
        multiplication_dict = modelspec.gwas_interaction
        self.assertEqual(40, len(multiplication_dict),
                         "Wrong number of interaction multiplication")
        self.assertEqual(40, len(inter.columns))
        self.assertEqual(set(inter.columns), set(multiplication_dict.keys()))

        # Checking the two way interaction
        all_variables = (
            ("SNPs", ),
            ("var1", ),
            ("factor(var3):x1", "factor(var3):x2"),
            ("factor(var5):1", "factor(var5):2", "factor(var5):3"),
        )
        for nb_term in range(2, len(all_variables)):
            for variables in itertools.combinations(all_variables, nb_term):
                for columns in itertools.product(*variables):
                    col_name = "gwas_inter({})".format(",".join(columns))
                    self.assertTrue(col_name in multiplication_dict)
                    self.assertEqual(columns, multiplication_dict[col_name])
