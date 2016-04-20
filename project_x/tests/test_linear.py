

# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import unittest
from itertools import zip_longest as zip

import numpy as np
import pandas as pd
from patsy import dmatrices
from pkg_resources import resource_filename

from ..statistics.models.linear import StatsLinear


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestStatsLinear(unittest.TestCase):
    """Tests the 'StatsLinear' class."""
    @classmethod
    def setUpClass(cls):
        cls.ols = StatsLinear()

    def setUp(self):
        self.data = pd.read_csv(
            resource_filename(__name__, "data/regression_linear.txt.bz2"),
            sep="\t",
            compression="bz2",
        )

    def test_linear_snp1(self):
        """Tests linear regression with the first SNP."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno1 ~ age + var1 + C(gender) + snp1",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.ols.fit(y, X, result_col="snp1")

        # Checking the results (according to R)
        self.assertAlmostEqual(113.1989213865802668, self.ols.results.coef)
        self.assertAlmostEqual(20.8583649966504652, self.ols.results.std_err)
        self.assertAlmostEqual(71.3978238271019876, self.ols.results.lower_ci)
        self.assertAlmostEqual(155.0000189460585602, self.ols.results.upper_ci)
        self.assertAlmostEqual(5.4270275452921783, self.ols.results.t_value)
        self.assertAlmostEqual(0.3082338225205, self.ols.results.rsquared_adj)

        # The p-value is small, so we check the -log10 value instead
        self.assertAlmostEqual(
            -np.log10(1.328591577071916e-06),
            -np.log10(self.ols.results.p_value),
        )

    def test_linear_snp2(self):
        """Tests linear regression with the second SNP."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno1 ~ age + var1 + C(gender) + snp2",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.ols.fit(y, X, result_col="snp2")

        # Checking the results (according to R)
        self.assertAlmostEqual(25.6638410624231454, self.ols.results.coef)
        self.assertAlmostEqual(7.0244242187562653, self.ols.results.std_err)
        self.assertAlmostEqual(11.5865803512147210, self.ols.results.lower_ci)
        self.assertAlmostEqual(39.7411017736315699, self.ols.results.upper_ci)
        self.assertAlmostEqual(3.6535152580757928, self.ols.results.t_value)
        self.assertAlmostEqual(0.1452372691714, self.ols.results.rsquared_adj)

        # The p-value is small, so we check the -log10 value instead
        self.assertAlmostEqual(
            -np.log10(0.0005783767026428237),
            -np.log10(self.ols.results.p_value),
        )

    def test_linear_snp3(self):
        """Tests linear regression with the third SNP."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno1 ~ age + var1 + C(gender) + snp3",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.ols.fit(y, X, result_col="snp3")

        # Checking the results (according to R)
        self.assertAlmostEqual(0.08097682855889538, self.ols.results.coef)
        self.assertAlmostEqual(6.6803747245602088, self.ols.results.std_err)
        self.assertAlmostEqual(-13.3067932886126528, self.ols.results.lower_ci)
        self.assertAlmostEqual(13.4687469457304445, self.ols.results.upper_ci)
        self.assertAlmostEqual(0.01212159974517393, self.ols.results.t_value)
        self.assertAlmostEqual(0.99037246258070233, self.ols.results.p_value)
        self.assertAlmostEqual(-0.06220573746784, self.ols.results.rsquared_adj)

    def test_linear_snp4(self):
        """Tests linear regression with the fourth SNP."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno1 ~ age + var1 + C(gender) + snp4",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.ols.fit(y, X, result_col="snp4")

        # Checking the results (according to R)
        self.assertAlmostEqual(-17.09338157602033803, self.ols.results.coef)
        self.assertAlmostEqual(6.495704343238213, self.ols.results.std_err)
        self.assertAlmostEqual(-30.1110639788755243, self.ols.results.lower_ci)
        self.assertAlmostEqual(-4.0756991731651517, self.ols.results.upper_ci)
        self.assertAlmostEqual(-2.6314900852613334, self.ols.results.t_value)
        self.assertAlmostEqual(0.01100922909893113, self.ols.results.p_value)
        self.assertAlmostEqual(0.05657300185575, self.ols.results.rsquared_adj)

    def test_linear_snp5(self):
        """Tests linear regression with the fifth SNP (raises ValueError)."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno1 ~ age + var1 + C(gender) + snp5",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        with self.assertRaises(ValueError) as cm:
            self.ols.fit(y, X, result_col="snp5")

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
