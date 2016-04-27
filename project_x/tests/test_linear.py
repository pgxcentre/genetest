

# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import unittest

import numpy as np
import pandas as pd
from patsy import dmatrices
from pkg_resources import resource_filename

from ..statistics.core import StatsError
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
            resource_filename(__name__, "data/statistics/linear.txt.bz2"),
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

    def test_linear_snp5(self):
        """Tests linear regression with the fifth SNP (raises StatsError)."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno1 ~ age + var1 + C(gender) + snp5",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        with self.assertRaises(StatsError) as cm:
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
