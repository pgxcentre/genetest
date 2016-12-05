

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import unittest

import numpy as np
import pandas as pd
from pkg_resources import resource_filename

from ..statistics.core import StatsError
from ..statistics.models.linear import StatsLinear


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestStatsLinear(unittest.TestCase):
    """Tests the 'StatsLinear' class."""
    @classmethod
    def setUpClass(cls):
        cls.ols = StatsLinear(
            outcome="pheno1",
            predictors=["age", "var1", "C(gender)", "geno"],
            interaction=None,
            condition_value_t=1000,
        )
        cls.ols_inter = StatsLinear(
            outcome="pheno1",
            predictors=["age", "var1", "C(gender)", "geno"],
            interaction="var1",
            condition_value_t=15000,
        )
        cls.ols_inter_categorical = StatsLinear(
            outcome="pheno1",
            predictors=["age", "var1", "C(gender)", "geno"],
            interaction="C(gender)",
            condition_value_t=15000,
        )

    def setUp(self):
        # The data
        self.data = pd.read_csv(
            resource_filename(__name__, "data/statistics/linear.txt.bz2"),
            sep="\t",
            compression="bz2",
        )

        # A dummy class for the 'get_phenotypes' function
        class DummyContainer(object):
            def set_phenotypes(self, data):
                self.data = data

            def get_phenotypes(self):
                return self.data
        self.dummy = DummyContainer()

    def test_linear_snp1_full(self):
        """Tests linear regression with the first SNP (full)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno1", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(columns={"snp1": "geno"})

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.ols.create_matrices(self.dummy)
        self.assertFalse("geno" in X.columns)

        # Merging with genotype
        y, X = self.ols.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.ols.fit(y, X)

        # The number of observation and parameters that were fitted
        n = X.shape[0]
        p = X.shape[1] - 1

        # Checking the results (according to SAS)
        self.assertAlmostEqual(113.19892138658, self.ols.results.coef)
        self.assertAlmostEqual(20.8583649966504, self.ols.results.std_err)
        self.assertAlmostEqual(71.397823827102, self.ols.results.lower_ci)
        self.assertAlmostEqual(155.000018946058, self.ols.results.upper_ci)
        self.assertAlmostEqual(5.42702754529217, self.ols.results.t_value)
        self.assertAlmostEqual(
            -np.log10(0.0000013285915771), -np.log10(self.ols.results.p_value),
        )
        self.assertAlmostEqual(
            1 - (n - 1) * (1 - 0.35513322438349)/((n - 1) - p),
            self.ols.results.rsquared_adj,
        )

    def test_linear_snp1_inter_full(self):
        """Tests linear regression for first SNP (interaction, full)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno1", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(columns={"snp1": "geno"})

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.ols_inter.create_matrices(self.dummy)
        self.assertFalse("geno" in X.columns)

        # Merging with genotype
        y, X = self.ols_inter.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.ols_inter.fit(y, X)

        # The number of observation and parameters that were fitted
        n = X.shape[0]
        p = X.shape[1] - 1

        # Checking the results (according to SAS)
        self.assertAlmostEqual(28.3750067790686, self.ols_inter.results.coef)
        self.assertAlmostEqual(15.31571903952, self.ols_inter.results.std_err)
        self.assertAlmostEqual(
            -2.33116110697257, self.ols_inter.results.lower_ci,
        )
        self.assertAlmostEqual(
            59.0811746651098, self.ols_inter.results.upper_ci,
        )
        self.assertAlmostEqual(
            1.85267219291832, self.ols_inter.results.t_value,
        )
        self.assertAlmostEqual(
            -np.log10(0.06939763567524),
            -np.log10(self.ols_inter.results.p_value),
        )
        self.assertAlmostEqual(
            1 - (n - 1) * (1 - 0.39367309450771)/((n - 1) - p),
            self.ols_inter.results.rsquared_adj,
        )

    def test_linear_snp1_inter_categorical_full(self):
        """Tests linear regression for first SNP (inter, full, category)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno1", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(columns={"snp1": "geno"})

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.ols_inter_categorical.create_matrices(self.dummy)
        self.assertFalse("geno" in X.columns)

        # Merging with genotype
        y, X = self.ols_inter_categorical.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.ols_inter_categorical.fit(y, X)

        # The number of observation and parameters that were fitted
        n = X.shape[0]
        p = X.shape[1] - 1

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            -74.5163756952978, self.ols_inter_categorical.results.coef,
        )
        self.assertAlmostEqual(
            40.2210255975831, self.ols_inter_categorical.results.std_err,
        )
        self.assertAlmostEqual(
            -155.154676865573, self.ols_inter_categorical.results.lower_ci,
            places=6,
        )
        self.assertAlmostEqual(
            6.12192547497808, self.ols_inter_categorical.results.upper_ci,
            places=6,
        )
        self.assertAlmostEqual(
            -1.8526721929183, self.ols_inter_categorical.results.t_value,
        )
        self.assertAlmostEqual(
            -np.log10(0.06939763567525),
            -np.log10(self.ols_inter_categorical.results.p_value),
        )
        self.assertAlmostEqual(
            1 - (n - 1) * (1 - 0.39367309450771)/((n - 1) - p),
            self.ols_inter_categorical.results.rsquared_adj,
        )

    def test_linear_snp1_inter_too_many_category_full(self):
        """Tests linear regression first SNP (interaction, full, to many)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno1", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(columns={"snp1": "geno"})

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Changing the gender so that there are two clases
        pheno.loc[:, "gender"] = np.random.randint(1, 5, pheno.shape[0])

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # This should raise an exception
        with self.assertRaises(ValueError):
            self.ols_inter_categorical.create_matrices(self.dummy)

    def test_linear_snp1(self):
        """Tests linear regression with the first SNP."""
        # Preparing the data
        data = self.data[["pheno1", "age", "var1", "gender", "snp1"]].rename(
            columns={"snp1": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.ols.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.ols.fit(y, X)

        # The number of observation and parameters that were fitted
        n = X.shape[0]
        p = X.shape[1] - 1

        # Checking the results (according to SAS)
        self.assertAlmostEqual(113.19892138658, self.ols.results.coef)
        self.assertAlmostEqual(20.8583649966504, self.ols.results.std_err)
        self.assertAlmostEqual(71.397823827102, self.ols.results.lower_ci)
        self.assertAlmostEqual(155.000018946058, self.ols.results.upper_ci)
        self.assertAlmostEqual(5.42702754529217, self.ols.results.t_value)
        self.assertAlmostEqual(
            -np.log10(0.0000013285915771), -np.log10(self.ols.results.p_value),
        )
        self.assertAlmostEqual(
            1 - (n - 1) * (1 - 0.35513322438349)/((n - 1) - p),
            self.ols.results.rsquared_adj,
        )

    def test_linear_snp1_inter(self):
        """Tests linear regression with the first SNP (interaction)."""
        # Preparing the data
        data = self.data[["pheno1", "age", "var1", "gender", "snp1"]].rename(
            columns={"snp1": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.ols_inter.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.ols_inter.fit(y, X)

        # The number of observation and parameters that were fitted
        n = X.shape[0]
        p = X.shape[1] - 1

        # Checking the results (according to SAS)
        self.assertAlmostEqual(28.3750067790686, self.ols_inter.results.coef)
        self.assertAlmostEqual(15.31571903952, self.ols_inter.results.std_err)
        self.assertAlmostEqual(
            -2.33116110697257, self.ols_inter.results.lower_ci,
        )
        self.assertAlmostEqual(
            59.0811746651098, self.ols_inter.results.upper_ci,
        )
        self.assertAlmostEqual(
            1.85267219291832, self.ols_inter.results.t_value,
        )
        self.assertAlmostEqual(
            -np.log10(0.06939763567524),
            -np.log10(self.ols_inter.results.p_value),
        )
        self.assertAlmostEqual(
            1 - (n - 1) * (1 - 0.39367309450771)/((n - 1) - p),
            self.ols_inter.results.rsquared_adj,
        )

    def test_linear_snp2(self):
        """Tests linear regression with the second SNP."""
        # Preparing the data
        data = self.data[["pheno1", "age", "var1", "gender", "snp2"]].rename(
            columns={"snp2": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.ols.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.ols.fit(y, X)

        # The number of observation and parameters that were fitted
        n = X.shape[0]
        p = X.shape[1] - 1

        # Checking the results (according to SAS)
        self.assertAlmostEqual(25.6638410624231, self.ols.results.coef)
        self.assertAlmostEqual(7.02442421875627, self.ols.results.std_err)
        self.assertAlmostEqual(11.5865803512147, self.ols.results.lower_ci)
        self.assertAlmostEqual(39.7411017736316, self.ols.results.upper_ci)
        self.assertAlmostEqual(3.65351525807579, self.ols.results.t_value)
        self.assertAlmostEqual(
            -np.log10(0.0005783767026428), -np.log10(self.ols.results.p_value),
        )
        self.assertAlmostEqual(
            1 - (n - 1) * (1 - 0.20318728482079)/((n - 1) - p),
            self.ols.results.rsquared_adj,
        )

    def test_linear_snp2_inter(self):
        """Tests linear regression with the second SNP (interaction)."""
        # Preparing the data
        data = self.data[["pheno1", "age", "var1", "gender", "snp2"]].rename(
            columns={"snp2": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.ols_inter.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.ols_inter.fit(y, X)

        # The number of observation and parameters that were fitted
        n = X.shape[0]
        p = X.shape[1] - 1

        # Checking the results (according to SAS)
        self.assertAlmostEqual(-0.38040787976905, self.ols_inter.results.coef)
        self.assertAlmostEqual(
            0.56827855931761, self.ols_inter.results.std_err,
        )
        self.assertAlmostEqual(
            -1.5197377932663, self.ols_inter.results.lower_ci,
        )
        self.assertAlmostEqual(
            0.75892203372818, self.ols_inter.results.upper_ci,
        )
        self.assertAlmostEqual(
            -0.66940389274205, self.ols_inter.results.t_value,
        )
        self.assertAlmostEqual(
            0.50609004475028, self.ols_inter.results.p_value,
        )
        self.assertAlmostEqual(
            1 - (n - 1) * (1 - 0.20974496120713)/((n - 1) - p),
            self.ols_inter.results.rsquared_adj,
        )

    def test_linear_snp3(self):
        """Tests linear regression with the third SNP."""
        # Preparing the data
        data = self.data[["pheno1", "age", "var1", "gender", "snp3"]].rename(
            columns={"snp3": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.ols.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.ols.fit(y, X)

        # The number of observation and parameters that were fitted
        n = X.shape[0]
        p = X.shape[1] - 1

        # Checking the results (according to SAS)
        self.assertAlmostEqual(0.08097682855889, self.ols.results.coef)
        self.assertAlmostEqual(6.6803747245602, self.ols.results.std_err)
        self.assertAlmostEqual(-13.3067932886126, self.ols.results.lower_ci)
        self.assertAlmostEqual(13.4687469457304, self.ols.results.upper_ci)
        self.assertAlmostEqual(0.0121215997451737, self.ols.results.t_value)
        self.assertAlmostEqual(0.99037246258077, self.ols.results.p_value)
        self.assertAlmostEqual(
            1 - (n - 1) * (1 - 0.0098082108350667)/((n - 1) - p),
            self.ols.results.rsquared_adj,
        )

    def test_linear_snp3_inter(self):
        """Tests linear regression with the third SNP."""
        # Preparing the data
        data = self.data[["pheno1", "age", "var1", "gender", "snp3"]].rename(
            columns={"snp3": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.ols_inter.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.ols_inter.fit(y, X)

        # The number of observation and parameters that were fitted
        n = X.shape[0]
        p = X.shape[1] - 1

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            -0.0097102715733324, self.ols_inter.results.coef,
        )
        self.assertAlmostEqual(
            0.60510302626961, self.ols_inter.results.std_err,
        )
        self.assertAlmostEqual(
            -1.22286879616119, self.ols_inter.results.lower_ci,
        )
        self.assertAlmostEqual(
            1.20344825301452, self.ols_inter.results.upper_ci,
        )
        self.assertAlmostEqual(
            -0.0160473029414429, self.ols_inter.results.t_value,
        )
        self.assertAlmostEqual(
            0.98725579876123, self.ols_inter.results.p_value,
        )
        self.assertAlmostEqual(
            1 - (n - 1) * (1 - 0.0098129328525696)/((n - 1) - p),
            self.ols_inter.results.rsquared_adj,
        )

    def test_linear_snp4(self):
        """Tests linear regression with the fourth SNP."""
        # Preparing the data
        data = self.data[["pheno1", "age", "var1", "gender", "snp4"]].rename(
            columns={"snp4": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.ols.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.ols.fit(y, X)

        # The number of observation and parameters that were fitted
        n = X.shape[0]
        p = X.shape[1] - 1

        # Checking the results (according to SAS)
        self.assertAlmostEqual(-17.0933815760203, self.ols.results.coef)
        self.assertAlmostEqual(6.49570434323821, self.ols.results.std_err)
        self.assertAlmostEqual(-30.1110639788755, self.ols.results.lower_ci)
        self.assertAlmostEqual(-4.07569917316514, self.ols.results.upper_ci)
        self.assertAlmostEqual(-2.63149008526133, self.ols.results.t_value)
        self.assertAlmostEqual(0.0110092290989312, self.ols.results.p_value)
        self.assertAlmostEqual(
            1 - (n - 1) * (1 - 0.1205341542723)/((n - 1) - p),
            self.ols.results.rsquared_adj,
        )

    def test_linear_snp4_inter(self):
        """Tests linear regression with the fourth SNP (interaction)."""
        # Preparing the data
        data = self.data[["pheno1", "age", "var1", "gender", "snp4"]].rename(
            columns={"snp4": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.ols_inter.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.ols_inter.fit(y, X)

        # The number of observation and parameters that were fitted
        n = X.shape[0]
        p = X.shape[1] - 1

        # Checking the results (according to SAS)
        self.assertAlmostEqual(-0.46834820683113, self.ols_inter.results.coef)
        self.assertAlmostEqual(
            0.50898831048606, self.ols_inter.results.std_err,
        )
        self.assertAlmostEqual(
            -1.48880832845448, self.ols_inter.results.lower_ci,
        )
        self.assertAlmostEqual(
            0.5521119147922, self.ols_inter.results.upper_ci,
        )
        self.assertAlmostEqual(
            -0.92015513359016, self.ols_inter.results.t_value,
        )
        self.assertAlmostEqual(
            0.36158411106165, self.ols_inter.results.p_value,
        )
        self.assertAlmostEqual(
            1 - (n - 1) * (1 - 0.13411074411446)/((n - 1) - p),
            self.ols_inter.results.rsquared_adj,
        )

    def test_linear_snp5(self):
        """Tests linear regression with the fifth SNP (raises StatsError)."""
        # Preparing the data
        data = self.data[["pheno1", "age", "var1", "gender", "snp5"]].rename(
            columns={"snp5": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.ols.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        with self.assertRaises(StatsError) as cm:
            self.ols.fit(y, X)

        # Checking the error message
        self.assertEqual(
            "condition number is large, inf",
            str(cm.exception),
        )

        # All the value should be NaN
        self.assertTrue(np.isnan(self.ols.results.coef))
        self.assertTrue(np.isnan(self.ols.results.std_err))
        self.assertTrue(np.isnan(self.ols.results.lower_ci))
        self.assertTrue(np.isnan(self.ols.results.upper_ci))
        self.assertTrue(np.isnan(self.ols.results.t_value))
        self.assertTrue(np.isnan(self.ols.results.p_value))
        self.assertTrue(np.isnan(self.ols.results.rsquared_adj))

    def test_linear_snp5_inter(self):
        """Tests linear regression fifth SNP (inter) (raises StatsError)."""
        # Preparing the data
        data = self.data[["pheno1", "age", "var1", "gender", "snp5"]].rename(
            columns={"snp5": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.ols_inter.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        with self.assertRaises(StatsError) as cm:
            self.ols_inter.fit(y, X)

        # Checking the error message
        self.assertEqual(
            "condition number is large, inf",
            str(cm.exception),
        )

        # All the value should be NaN
        self.assertTrue(np.isnan(self.ols_inter.results.coef))
        self.assertTrue(np.isnan(self.ols_inter.results.std_err))
        self.assertTrue(np.isnan(self.ols_inter.results.lower_ci))
        self.assertTrue(np.isnan(self.ols_inter.results.upper_ci))
        self.assertTrue(np.isnan(self.ols_inter.results.t_value))
        self.assertTrue(np.isnan(self.ols_inter.results.p_value))
        self.assertTrue(np.isnan(self.ols_inter.results.rsquared_adj))
