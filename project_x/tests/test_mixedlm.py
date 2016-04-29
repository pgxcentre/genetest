

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
from ..statistics.models.mixedlm import StatsMixedLM


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

        # The genotypes
        genotypes = pd.DataFrame(
            [("s0", 0.0), ("s1", 1.0), ("s2", 0.0), ("s3", np.nan),
             ("s4", 0.0), ("s5", 1.0), ("s6", 2.0), ("s7", np.nan),
             ("s8", 0.0), ("s9", 0.0)],
            columns=["sample_id", "geno"],
        ).set_index("sample_id")
        self.genotypes = genotypes.iloc[np.random.permutation(len(genotypes))]

        # The "original" samples
        samples = pd.DataFrame(
            [("s0_1", "s0"), ("s0_2", "s0"), ("s0_3", "s0"), ("s1_1", "s1"),
             ("s1_2", "s1"), ("s1_3", "s1"), ("s2_1", "s2"), ("s2_2", "s2"),
             ("s2_3", "s2"), ("s3_1", "s3"), ("s3_2", "s3"), ("s3_3", "s3"),
             ("s4_1", "s4"), ("s4_2", "s4"), ("s4_3", "s4"), ("s5_1", "s5"),
             ("s5_2", "s5"), ("s5_3", "s5"), ("s6_1", "s6"), ("s6_2", "s6"),
             ("s6_3", "s6"), ("s7_1", "s7"), ("s7_2", "s7"), ("s7_3", "s7"),
             ("s8_1", "s8"), ("s8_2", "s8"), ("s8_3", "s8"), ("s9_1", "s9"),
             ("s9_2", "s9"), ("s9_3", "s9")],
            columns=["sample_id", "_ori_sample_names_"],
        ).set_index("sample_id")
        self.samples = samples.iloc[np.random.permutation(len(samples))]

        # The values for y
        y = pd.DataFrame(
            [("s0_1", 3.0), ("s0_2", 3.4), ("s0_3", 3.4), ("s1_1", 3.4),
             ("s1_2", 3.4), ("s1_3", 3.4), ("s2_1", 5.3), ("s2_2", 5.3),
             ("s2_3", 5.3), ("s3_1", 6.0), ("s3_2", 6.0), ("s3_3", 6.0),
             ("s4_1", 0.5), ("s4_2", 0.5), ("s4_3", 0.5), ("s5_1", 2.4),
             ("s5_2", 2.4), ("s5_3", 2.4), ("s6_1", 5.6), ("s6_2", 5.6),
             ("s6_3", 5.6), ("s7_1", 7.6), ("s7_2", 7.6), ("s7_3", 7.6),
             ("s8_1", 0.3), ("s8_2", 0.3), ("s8_3", 0.3), ("s9_1", 1.9),
             ("s9_2", 1.9), ("s9_3", 1.9)],
            columns=["sample_id", "pheno"],
        ).set_index("sample_id")
        self.y = y.iloc[np.random.permutation(len(y))]

        # The values for X
        X = pd.DataFrame(
            [("s0_1", 1.0, 0.0, 0.0, 12.0), ("s0_2", 1.0, 0.0, 0.0, 12.0),
             ("s0_3", 1.0, 0.0, 0.0, 12.0), ("s1_1", 1.0, 1.0, 0.0, 30.8),
             ("s1_2", 1.0, 1.0, 0.0, 30.8), ("s1_3", 1.0, 1.0, 0.0, 30.8),
             ("s2_1", 1.0, 0.0, 0.0, 50.2), ("s2_2", 1.0, 0.0, 0.0, 50.2),
             ("s2_3", 1.0, 0.0, 0.0, 50.2), ("s3_1", 1.0, 0.0, 1.0, 30.6),
             ("s3_2", 1.0, 0.0, 1.0, 30.6), ("s3_3", 1.0, 0.0, 1.0, 30.6),
             ("s4_1", 1.0, 1.0, 0.0, 40.0), ("s4_2", 1.0, 1.0, 0.0, 40.0),
             ("s4_3", 1.0, 1.0, 0.0, 40.0), ("s5_1", 1.0, 0.0, 0.0, 80.5),
             ("s5_2", 1.0, 0.0, 0.0, 80.5), ("s5_3", 1.0, 0.0, 0.0, 80.5),
             ("s6_1", 1.0, 0.0, 0.0, 70.0), ("s6_2", 1.0, 0.0, 0.0, 70.0),
             ("s6_3", 1.0, 0.0, 0.0, 70.0), ("s7_1", 1.0, 1.0, 0.0, 87.4),
             ("s7_2", 1.0, 1.0, 0.0, 87.4), ("s7_3", 1.0, 1.0, 0.0, 87.4),
             ("s8_1", 1.0, 0.0, 0.0, 63.0), ("s8_2", 1.0, 0.0, 0.0, 63.0),
             ("s8_3", 1.0, 0.0, 0.0, 63.0), ("s9_1", 1.0, 0.0, 1.0, 54.3),
             ("s9_2", 1.0, 0.0, 1.0, 54.3), ("s9_3", 1.0, 0.0, 1.0, 54.3)],
            columns=["sample_id", "Intercept", "C(var2)[T.f2]",
                     "C(var2)[T.f3]", "var1"],
        ).set_index("sample_id")
        self.X = X.iloc[np.random.permutation(len(X))]

        # The expected y
        self.expected_y = pd.DataFrame(
            [("s0_1", 3.0), ("s0_2", 3.4), ("s0_3", 3.4), ("s1_1", 3.4),
             ("s1_2", 3.4), ("s1_3", 3.4), ("s2_1", 5.3), ("s2_2", 5.3),
             ("s2_3", 5.3), ("s4_1", 0.5), ("s4_2", 0.5), ("s4_3", 0.5),
             ("s5_1", 2.4), ("s5_2", 2.4), ("s5_3", 2.4), ("s6_1", 5.6),
             ("s6_2", 5.6), ("s6_3", 5.6), ("s8_1", 0.3), ("s8_2", 0.3),
             ("s8_3", 0.3), ("s9_1", 1.9), ("s9_2", 1.9), ("s9_3", 1.9)],
            columns=["sample_id", "pheno"],
        ).set_index("sample_id")

        # The expected X
        self.expected_X = pd.DataFrame(
            [("s0_1", 1.0, 0.0, 0.0, 12.0, 0.0),
             ("s0_2", 1.0, 0.0, 0.0, 12.0, 0.0),
             ("s0_3", 1.0, 0.0, 0.0, 12.0, 0.0),
             ("s1_1", 1.0, 1.0, 0.0, 30.8, 1.0),
             ("s1_2", 1.0, 1.0, 0.0, 30.8, 1.0),
             ("s1_3", 1.0, 1.0, 0.0, 30.8, 1.0),
             ("s2_1", 1.0, 0.0, 0.0, 50.2, 0.0),
             ("s2_2", 1.0, 0.0, 0.0, 50.2, 0.0),
             ("s2_3", 1.0, 0.0, 0.0, 50.2, 0.0),
             ("s4_1", 1.0, 1.0, 0.0, 40.0, 0.0),
             ("s4_2", 1.0, 1.0, 0.0, 40.0, 0.0),
             ("s4_3", 1.0, 1.0, 0.0, 40.0, 0.0),
             ("s5_1", 1.0, 0.0, 0.0, 80.5, 1.0),
             ("s5_2", 1.0, 0.0, 0.0, 80.5, 1.0),
             ("s5_3", 1.0, 0.0, 0.0, 80.5, 1.0),
             ("s6_1", 1.0, 0.0, 0.0, 70.0, 2.0),
             ("s6_2", 1.0, 0.0, 0.0, 70.0, 2.0),
             ("s6_3", 1.0, 0.0, 0.0, 70.0, 2.0),
             ("s8_1", 1.0, 0.0, 0.0, 63.0, 0.0),
             ("s8_2", 1.0, 0.0, 0.0, 63.0, 0.0),
             ("s8_3", 1.0, 0.0, 0.0, 63.0, 0.0),
             ("s9_1", 1.0, 0.0, 1.0, 54.3, 0.0),
             ("s9_2", 1.0, 0.0, 1.0, 54.3, 0.0),
             ("s9_3", 1.0, 0.0, 1.0, 54.3, 0.0)],
            columns=["sample_id", "Intercept", "C(var2)[T.f2]",
                     "C(var2)[T.f3]", "var1", "geno"],
        ).set_index("sample_id")

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

    def test_merge_matrices_genotypes(self):
        """Tests the 'merge_matrices_genotypes' function for MixedLM."""
        new_y, new_X, groups = self.mixedlm.merge_matrices_genotypes(
            y=self.y, X=self.X, genotypes=self.genotypes,
            ori_samples=self.samples,
        )

        # Checking the results
        self.assertTrue(np.array_equal(new_y.index.values, new_X.index.values))
        self.assertTrue(self.expected_y.equals(new_y.sortlevel()))
        self.assertTrue(self.expected_X.equals(new_X.sortlevel()))
        self.assertTrue(np.array_equal(
            np.array([s[:2] for s in new_X.index.values]),
            groups,
        ))
