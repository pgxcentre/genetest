

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
from ..statistics.models.mixed_lm import StatsMixedLM


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestStatsMixedLM(unittest.TestCase):
    """Tests the 'StatsMixedLM' class."""
    @classmethod
    def setUpClass(cls):
        cls.mixedlm = StatsMixedLM()

    def setUp(self):
        self.data = pd.read_csv(
            resource_filename(__name__, "data/statistics/mixedlm.txt.bz2"),
            sep="\t",
            compression="bz2",
        )

    def test_mixedlm_snp1_reml(self):
        """Tests mixedlm regression with the first SNP (using REML)."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno3 ~ age + var1 + C(gender) + snp1 + C(visit)",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.mixedlm.fit(y, X, groups=self.data.sampleid, result_col="snp1",
                         reml=True)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            37.45704065571509034, self.mixedlm.results.coef,
        )
        self.assertAlmostEqual(
            8.3417880825867652, self.mixedlm.results.std_err, places=4,
        )
        self.assertAlmostEqual(
            21.1074364471796017, self.mixedlm.results.lower_ci, places=4,
        )
        self.assertAlmostEqual(
            53.806644864250579, self.mixedlm.results.upper_ci, places=4,
        )
        self.assertAlmostEqual(
            4.4902891664085249, self.mixedlm.results.z_value, places=4,
        )
        self.assertAlmostEqual(
            -np.log10(7.112654716978639e-06),
            -np.log10(self.mixedlm.results.p_value), places=4,
        )

    def test_mixedlm_snp1_ml(self):
        """Tests mixedlm regression with the first SNP (using ML)."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno3 ~ age + var1 + C(gender) + snp1 + C(visit)",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.mixedlm.fit(y, X, groups=self.data.sampleid, result_col="snp1",
                         reml=False)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            37.45704065571743513, self.mixedlm.results.coef,
        )
        self.assertAlmostEqual(
            7.986654823996228, self.mixedlm.results.std_err, places=5,
        )
        self.assertAlmostEqual(
            21.8034848437317450, self.mixedlm.results.lower_ci, places=4,
        )
        self.assertAlmostEqual(
            53.1105964677031253, self.mixedlm.results.upper_ci, places=4,
        )
        self.assertAlmostEqual(
            4.6899536140182541, self.mixedlm.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            -np.log10(2.732669904137452e-06),
            -np.log10(self.mixedlm.results.p_value), places=5,
        )

    def test_mixedlm_snp2_reml(self):
        """Tests mixedlm regression with the second SNP (using REML)."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno3 ~ age + var1 + C(gender) + snp2 + C(visit)",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.mixedlm.fit(y, X, groups=self.data.sampleid, result_col="snp2",
                         reml=True)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            28.87619649886422479, self.mixedlm.results.coef,
        )
        self.assertAlmostEqual(
            7.616420586507464, self.mixedlm.results.std_err, places=4,
        )
        self.assertAlmostEqual(
            13.9482864582001636, self.mixedlm.results.lower_ci, places=4,
        )
        self.assertAlmostEqual(
            43.8041065395282843, self.mixedlm.results.upper_ci, places=4,
        )
        self.assertAlmostEqual(
            3.7913080259798924, self.mixedlm.results.z_value, places=4,
        )
        self.assertAlmostEqual(
            -np.log10(1.498559584953707e-04),
            -np.log10(self.mixedlm.results.p_value), places=4,
        )

    def test_mixedlm_snp2_ml(self):
        """Tests mixedlm regression with the second SNP (using ML)."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno3 ~ age + var1 + C(gender) + snp2 + C(visit)",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.mixedlm.fit(y, X, groups=self.data.sampleid, result_col="snp2",
                         reml=False)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            28.87619649885465023, self.mixedlm.results.coef,
        )
        self.assertAlmostEqual(
            7.2921675410708389, self.mixedlm.results.std_err, places=4,
        )
        self.assertAlmostEqual(
            14.5838107491238045, self.mixedlm.results.lower_ci, places=4,
        )
        self.assertAlmostEqual(
            43.1685822485854942, self.mixedlm.results.upper_ci, places=4,
        )
        self.assertAlmostEqual(
            3.9598920809510423, self.mixedlm.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            -np.log10(7.498364115088307e-05),
            -np.log10(self.mixedlm.results.p_value), places=4,
        )

    def test_mixedlm_snp3_reml(self):
        """Tests mixedlm regression with the third SNP (using REML)."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno3 ~ age + var1 + C(gender) + snp3 + C(visit)",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.mixedlm.fit(y, X, groups=self.data.sampleid, result_col="snp3",
                         reml=True)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            21.61350866000438486, self.mixedlm.results.coef,
        )
        self.assertAlmostEqual(
            6.4018199962254876, self.mixedlm.results.std_err, places=4,
        )
        self.assertAlmostEqual(
            9.0661720318940873, self.mixedlm.results.lower_ci, places=4,
        )
        self.assertAlmostEqual(
            34.160845288114686, self.mixedlm.results.upper_ci, places=4,
        )
        self.assertAlmostEqual(
            3.37615063727935283, self.mixedlm.results.z_value, places=4,
        )
        self.assertAlmostEqual(
            -np.log10(7.350766113720653e-04),
            -np.log10(self.mixedlm.results.p_value), places=4,
        )

    def test_mixedlm_snp3_ml(self):
        """Tests mixedlm regression with the third SNP (using ML)."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno3 ~ age + var1 + C(gender) + snp3 + C(visit)",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.mixedlm.fit(y, X, groups=self.data.sampleid, result_col="snp3",
                         reml=False)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            21.61350866001425786, self.mixedlm.results.coef,
        )
        self.assertAlmostEqual(
            6.1292761423280124, self.mixedlm.results.std_err, places=4,
        )
        self.assertAlmostEqual(
            9.6003481697507578, self.mixedlm.results.lower_ci, places=4,
        )
        self.assertAlmostEqual(
            33.6266691502777562, self.mixedlm.results.upper_ci, places=4,
        )
        self.assertAlmostEqual(
            3.52627425459820243, self.mixedlm.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            -np.log10(4.214502895376615e-04),
            -np.log10(self.mixedlm.results.p_value), places=4,
        )

    def test_mixedlm_snp4_reml(self):
        """Tests mixedlm regression with the fourth SNP (using REML)."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno3 ~ age + var1 + C(gender) + snp4 + C(visit)",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.mixedlm.fit(y, X, groups=self.data.sampleid, result_col="snp4",
                         reml=True)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            1.4891822461435404, self.mixedlm.results.coef,
        )
        self.assertAlmostEqual(
            7.9840787818167422, self.mixedlm.results.std_err, places=3,
        )
        self.assertAlmostEqual(
            -14.1593246159476980, self.mixedlm.results.lower_ci, places=3,
        )
        self.assertAlmostEqual(
            17.1376891082347775, self.mixedlm.results.upper_ci, places=3,
        )
        self.assertAlmostEqual(
            0.1865189819437983, self.mixedlm.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            8.520377946017625e-01, self.mixedlm.results.p_value, places=5,
        )

    def test_mixedlm_snp4_ml(self):
        """Tests mixedlm regression with the fourth SNP (using ML)."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno3 ~ age + var1 + C(gender) + snp4 + C(visit)",
            self.data,
            return_type="dataframe",
        )

        # Fitting
        self.mixedlm.fit(y, X, groups=self.data.sampleid, result_col="snp4",
                         reml=False)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            1.489182246136439, self.mixedlm.results.coef,
        )
        self.assertAlmostEqual(
            7.644172965779382, self.mixedlm.results.std_err, places=4,
        )
        self.assertAlmostEqual(
            -13.4931214583858772, self.mixedlm.results.lower_ci, places=3,
        )
        self.assertAlmostEqual(
            16.4714859506587565, self.mixedlm.results.upper_ci, places=3,
        )
        self.assertAlmostEqual(
            0.1948127355049462, self.mixedlm.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            8.455395518199895e-01, self.mixedlm.results.p_value, places=5,
        )

    def test_mixedlm_monomorphic(self):
        """Tests mixedlm regression on monomorphic SNP."""
        # Preparing the matrices
        y, X = dmatrices(
            "pheno3 ~ age + var1 + C(gender) + snp4 + C(visit)",
            self.data,
            return_type="dataframe",
        )
        X.snp4 = 0

        # Fitting
        with self.assertRaises(StatsError):
            self.mixedlm.fit(y, X, groups=self.data.sampleid,
                             result_col="snp4", reml=False)

        # Checking the results
        self.assertTrue(np.isnan(self.mixedlm.results.coef))
        self.assertTrue(np.isnan(self.mixedlm.results.std_err))
        self.assertTrue(np.isnan(self.mixedlm.results.lower_ci))
        self.assertTrue(np.isnan(self.mixedlm.results.upper_ci))
        self.assertTrue(np.isnan(self.mixedlm.results.z_value))
        self.assertTrue(np.isnan(self.mixedlm.results.p_value))
