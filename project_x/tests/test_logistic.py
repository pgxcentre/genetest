

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
            resource_filename(__name__, "data/statistics/logistic.txt.bz2"),
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

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            -2.24198635855498, self.logistic.results.coef, places=5,
        )
        self.assertAlmostEqual(
            0.59759558908668, self.logistic.results.std_err, places=6,
        )
        self.assertAlmostEqual(
            -3.41325219048488, self.logistic.results.lower_ci, places=5,
        )
        self.assertAlmostEqual(
            -1.07072052662507, self.logistic.results.upper_ci, places=5,
        )
        self.assertAlmostEqual(
            14.0750894991982, self.logistic.results.z_value**2, places=4,
        )
        self.assertAlmostEqual(
            -np.log10(0.0001756548178104),
            -np.log10(self.logistic.results.p_value), places=5,
        )

    def test_logistic_snp1_formula(self):
        """Tests logistic regression with the first SNP (with formula)."""
        # Fitting
        self.logistic.from_formula(
            formula="pheno2 ~ age + var1 + C(gender) + snp1",
            data=self.data,
            result_col="snp1",
        )

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            -2.24198635855498, self.logistic.results.coef, places=5,
        )
        self.assertAlmostEqual(
            0.59759558908668, self.logistic.results.std_err, places=6,
        )
        self.assertAlmostEqual(
            -3.41325219048488, self.logistic.results.lower_ci, places=5,
        )
        self.assertAlmostEqual(
            -1.07072052662507, self.logistic.results.upper_ci, places=5,
        )
        self.assertAlmostEqual(
            14.0750894991982, self.logistic.results.z_value**2, places=4,
        )
        self.assertAlmostEqual(
            -np.log10(0.0001756548178104),
            -np.log10(self.logistic.results.p_value), places=5,
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

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            1.12532308347075, self.logistic.results.coef, places=5,
        )
        self.assertAlmostEqual(
            0.45211815097241, self.logistic.results.std_err, places=6,
        )
        self.assertAlmostEqual(
            0.23918779080797, self.logistic.results.lower_ci, places=5,
        )
        self.assertAlmostEqual(
            2.01145837613353, self.logistic.results.upper_ci, places=5,
        )
        self.assertAlmostEqual(
            6.19513207316499, self.logistic.results.z_value**2, places=4,
        )
        self.assertAlmostEqual(
            -np.log10(0.0128102164253392),
            -np.log10(self.logistic.results.p_value), places=5,
        )

    def test_logistic_snp2_formula(self):
        """Tests logistic regression with the second SNP (from formula)."""
        # Fitting
        self.logistic.from_formula(
            formula="pheno2 ~ age + var1 + C(gender) + snp2",
            data=self.data,
            result_col="snp2",
        )

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            1.12532308347075, self.logistic.results.coef, places=5,
        )
        self.assertAlmostEqual(
            0.45211815097241, self.logistic.results.std_err, places=6,
        )
        self.assertAlmostEqual(
            0.2391877908079700, self.logistic.results.lower_ci, places=5,
        )
        self.assertAlmostEqual(
            2.0114583761335300, self.logistic.results.upper_ci, places=5,
        )
        self.assertAlmostEqual(
            6.19513207316499, self.logistic.results.z_value**2, places=4,
        )
        self.assertAlmostEqual(
            -np.log10(0.0128102164253392),
            -np.log10(self.logistic.results.p_value), places=5,
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

    def test_logistic_snp3_formula(self):
        """Tests logistic regression with the third SNP (with formula)."""
        # Fitting
        with self.assertRaises(StatsError):
            self.logistic.from_formula(
                formula="pheno2 ~ age + var1 + C(gender) + snp3",
                data=self.data,
                result_col="snp3",
            )

        # All the value should be NaN
        self.assertTrue(np.isnan(self.logistic.results.coef))
        self.assertTrue(np.isnan(self.logistic.results.std_err))
        self.assertTrue(np.isnan(self.logistic.results.lower_ci))
        self.assertTrue(np.isnan(self.logistic.results.upper_ci))
        self.assertTrue(np.isnan(self.logistic.results.z_value))
        self.assertTrue(np.isnan(self.logistic.results.p_value))
