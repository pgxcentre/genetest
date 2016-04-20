

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

from ..statistics.core import StatsError
from ..statistics.models.logistic import StatsLogistic


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestStatsLogistic(unittest.TestCase):
    """Tests the 'StatsLogistic' class."""
    @classmethod
    def setUpClass(cls):
        cls.logistic = StatsLogistic()

    def setUp(self):
        self.data = pd.read_csv(
            resource_filename(__name__, "data/regression_logistic.txt.bz2"),
            sep="\t",
            compression="bz2",
        )

    def test_logistic_snp1(self):
        """Tests logistic regression with the first SNP."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno2 ~ age + var1 + C(gender) + snp1",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.logistic.fit(y, X, result_col="snp1")

        # Checking the results (according to R)
        self.assertAlmostEqual(-2.24198759580421436,
                               self.logistic.results.coef)
        self.assertAlmostEqual(0.59758174548961274,
                               self.logistic.results.std_err, places=4)
        self.assertAlmostEqual(-3.55431314172335844 / 10,
                               self.logistic.results.lower_ci / 10, places=1)
        self.assertAlmostEqual(-1.18244470804878388 / 10,
                               self.logistic.results.upper_ci / 10, places=1)
        self.assertAlmostEqual(-3.7517672062879051,
                               self.logistic.results.z_value, places=3)

        # The p-value is small, so we check the -log10 value instead
        self.assertAlmostEqual(
            -np.log10(0.0001755924726864904),
            -np.log10(self.logistic.results.p_value),
            places=3,
        )

    def test_logistic_snp2(self):
        """Tests logistic regression with the second SNP."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno2 ~ age + var1 + C(gender) + snp2",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.logistic.fit(y, X, result_col="snp2")

        # Checking the results (according to R)
        self.assertAlmostEqual(1.12532537333269489,
                               self.logistic.results.coef)
        self.assertAlmostEqual(0.45211812806716867,
                               self.logistic.results.std_err, places=6)
        self.assertAlmostEqual(0.28806348871004650,
                               self.logistic.results.lower_ci, places=1)
        self.assertAlmostEqual(2.08869739528846932 / 10,
                               self.logistic.results.upper_ci / 10, places=1)
        self.assertAlmostEqual(2.4890074152600925,
                               self.logistic.results.z_value, places=3)

        # The p-value is small, so we check the -log10 value instead
        self.assertAlmostEqual(
            -np.log10(0.01281002939196238),
            -np.log10(self.logistic.results.p_value),
            places=5,
        )

    def test_logistic_snp3(self):
        """Tests logistic regression with the third SNP (raises StatsError)."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno2 ~ age + var1 + C(gender) + snp3",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        with self.assertRaises(StatsError):
            self.logistic.fit(y, X, result_col="snp3")

        # All the value should be NaN
        self.assertTrue(np.isnan(self.logistic.results.coef))
        self.assertTrue(np.isnan(self.logistic.results.std_err))
        self.assertTrue(np.isnan(self.logistic.results.lower_ci))
        self.assertTrue(np.isnan(self.logistic.results.upper_ci))
        self.assertTrue(np.isnan(self.logistic.results.z_value))
        self.assertTrue(np.isnan(self.logistic.results.p_value))
