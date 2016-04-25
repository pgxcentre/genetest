

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
from ..statistics.models.survival import StatsCoxPH


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestStatsCoxPH(unittest.TestCase):
    """Tests the 'StatsCoxPH' class."""
    @classmethod
    def setUpClass(cls):
        cls.coxph = StatsCoxPH()

    def setUp(self):
        self.data = pd.read_csv(
            resource_filename(__name__, "data/statistics/coxph.txt.bz2"),
            sep="\t",
            compression="bz2",
        )

    def test_coxph_snp1(self):
        """Tests coxph regression with the first SNP."""
        # Preparing the matrices
        y, X = dmatrices(
            "tte + event ~ age + var1 + C(gender) + snp1",
            self.data,
            return_type="dataframe",
        )

        # Dropping the intercept (for linefiles)
        X = X.drop("Intercept", axis=1)

        # Fitting
        self.coxph.fit(y, X, tte="tte", event="event", result_col="snp1")

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            -3.49771655666559, self.coxph.results.coef, places=2,
        )
        self.assertAlmostEqual(
            1.05740411066576, self.coxph.results.std_err, places=4,
        )
        self.assertAlmostEqual(
            0.0302664162224589, self.coxph.results.hr, places=4,
        )
        self.assertAlmostEqual(
            0.0038097544883423, self.coxph.results.hr_lower_ci, places=5,
        )
        self.assertAlmostEqual(
            0.24045012710247, self.coxph.results.hr_upper_ci, places=5,
        )
        self.assertAlmostEqual(
            10.9417613148234, self.coxph.results.z_value**2, places=2,
        )
        self.assertAlmostEqual(
            -np.log10(0.0009402074852055),
            -np.log10(self.coxph.results.p_value), places=2,
        )

    def test_coxph_snp2(self):
        """Tests coxph regression with the second SNP."""
        # Preparing the matrices
        y, X = dmatrices(
            "tte + event ~ age + var1 + C(gender) + snp2",
            self.data,
            return_type="dataframe",
        )

        # Dropping the intercept (for linefiles)
        X = X.drop("Intercept", axis=1)

        # Fitting
        self.coxph.fit(y, X, tte="tte", event="event", result_col="snp2")

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            1.13120378638922, self.coxph.results.coef, places=6,
        )
        self.assertAlmostEqual(
            0.30473046186896, self.coxph.results.std_err,
        )
        self.assertAlmostEqual(
            3.09938525314605, self.coxph.results.hr, places=6,
        )
        self.assertAlmostEqual(
            1.70564451869289, self.coxph.results.hr_lower_ci, places=3,
        )
        self.assertAlmostEqual(
            5.63199942434711, self.coxph.results.hr_upper_ci, places=2,
        )
        self.assertAlmostEqual(
            13.780023571179, self.coxph.results.z_value**2, places=5,
        )
        self.assertAlmostEqual(
            -np.log10(0.0002055098552077),
            -np.log10(self.coxph.results.p_value), places=6,
        )

    def test_coxph_snp3(self):
        """Tests coxph regression with the third SNP."""
        # Preparing the matrices
        y, X = dmatrices(
            "tte + event ~ age + var1 + C(gender) + snp3",
            self.data,
            return_type="dataframe",
        )

        # Dropping the intercept (for linefiles)
        X = X.drop("Intercept", axis=1)

        # Fitting
        self.coxph.fit(y, X, tte="tte", event="event", result_col="snp3")

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            -0.0069430199975568, self.coxph.results.coef, places=5,
        )
        self.assertAlmostEqual(
            0.39831693319749, self.coxph.results.std_err,
        )
        self.assertAlmostEqual(
            0.99308102708048, self.coxph.results.hr, places=5,
        )
        self.assertAlmostEqual(
            0.45492174515325, self.coxph.results.hr_lower_ci, places=3,
        )
        self.assertAlmostEqual(
            2.16786719222444, self.coxph.results.hr_upper_ci, places=3,
        )
        self.assertAlmostEqual(
            0.000303836044335, self.coxph.results.z_value**2, places=6,
        )
        self.assertAlmostEqual(
            0.98609286353578, self.coxph.results.p_value, places=4,
        )

    def test_coxph_snp4(self):
        """Tests coxph regression with the third SNP."""
        # Preparing the matrices
        y, X = dmatrices(
            "tte + event ~ age + var1 + C(gender) + snp4",
            self.data,
            return_type="dataframe",
        )

        # Dropping the intercept (for linefiles)
        X = X.drop("Intercept", axis=1)

        # Fitting
        with self.assertRaises(StatsError):
            self.coxph.fit(y, X, tte="tte", event="event", result_col="snp4")

        # Checking the results (according to SAS)
        self.assertTrue(np.isnan(self.coxph.results.coef))
        self.assertTrue(np.isnan(self.coxph.results.std_err))
        self.assertTrue(np.isnan(self.coxph.results.hr))
        self.assertTrue(np.isnan(self.coxph.results.hr_lower_ci))
        self.assertTrue(np.isnan(self.coxph.results.hr_upper_ci))
        self.assertTrue(np.isnan(self.coxph.results.z_value))
        self.assertTrue(np.isnan(self.coxph.results.p_value))
