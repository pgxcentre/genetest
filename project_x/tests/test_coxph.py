

# This file is part of project_x.
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
from ..statistics.models.survival import StatsCoxPH


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestStatsCoxPH(unittest.TestCase):
    """Tests the 'StatsCoxPH' class."""
    @classmethod
    def setUpClass(cls):
        cls.coxph = StatsCoxPH(
            time_to_event="tte",
            event="event",
            predictors=["age", "var1", "C(gender)", "geno"],
            interaction=None,
            normalize=False,
        )
        cls.coxph_inter = StatsCoxPH(
            time_to_event="tte",
            event="event",
            predictors=["age", "var1", "C(gender)", "geno"],
            interaction="var1",
            normalize=False,
        )
        cls.coxph_inter_cat = StatsCoxPH(
            time_to_event="tte",
            event="event",
            predictors=["age", "var1", "C(gender)", "geno"],
            interaction="C(gender)",
            normalize=False,
        )

    def setUp(self):
        self.data = pd.read_csv(
            resource_filename(__name__, "data/statistics/coxph.txt.bz2"),
            sep="\t",
            compression="bz2",
        )

    def test_coxph_snp1_full(self):
        """Tests coxph regression with the first SNP (full)."""
        # Preparing the data
        pheno = self.data[["tte", "event", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(columns={"snp1": "geno"})

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Preparing the matrices
        y, X = self.coxph.create_matrices(pheno)

        # Merging with genotype
        y, X = self.coxph.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.coxph.fit(y, X)

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

    def test_coxph_snp1_inter_full(self):
        """Tests coxph regression with the first SNP (full, interaction)."""
        # Preparing the data
        pheno = self.data[["tte", "event", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(columns={"snp1": "geno"})

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Preparing the matrices
        y, X = self.coxph_inter.create_matrices(pheno)

        # Merging with genotype
        y, X = self.coxph_inter.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.coxph_inter.fit(y, X)

        # Checking the results (according to SAS)
        # SAS doesn't compute a hazard ratio (and its 95% CI) for interaction
        # (since it would be interpretable)
        self.assertAlmostEqual(
            0.047567074771854, self.coxph_inter.results.coef, places=4,
        )
        self.assertAlmostEqual(
            0.06434824185209, self.coxph_inter.results.std_err, places=5,
        )
        self.assertAlmostEqual(
            np.exp(0.047567074771854), self.coxph_inter.results.hr, places=4,
        )
        self.assertAlmostEqual(
            0.54643626988504, self.coxph_inter.results.z_value**2, places=3,
        )
        self.assertAlmostEqual(
            0.45977749951036, self.coxph_inter.results.p_value, places=3,
        )

    def test_coxph_snp1_inter_category_full(self):
        """Tests coxph first SNP (full, interaction, category)."""
        # Preparing the data
        pheno = self.data[["tte", "event", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(columns={"snp1": "geno"})

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Preparing the matrices
        y, X = self.coxph_inter_cat.create_matrices(pheno)

        # Merging with genotype
        y, X = self.coxph_inter_cat.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.coxph_inter_cat.fit(y, X)

        # Checking the results (according to SAS)
        # SAS doesn't compute a hazard ratio (and its 95% CI) for interaction
        # (since it would be interpretable)
        self.assertAlmostEqual(
            -1.34428789715448, self.coxph_inter_cat.results.coef, places=3,
        )
        self.assertAlmostEqual(
            1.7731990844019, self.coxph_inter_cat.results.std_err, places=5,
        )
        self.assertAlmostEqual(
            np.exp(-1.34428789715448), self.coxph_inter_cat.results.hr,
            places=4,
        )
        self.assertAlmostEqual(
            0.57473756080668, self.coxph_inter_cat.results.z_value**2,
            places=3,
        )
        self.assertAlmostEqual(
            0.44838245349613, self.coxph_inter_cat.results.p_value, places=4,
        )

    def test_coxph_snp1(self):
        """Tests coxph regression with the first SNP."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp1"]]
        data = data.rename(columns={"snp1": "geno"})

        # Preparing the matrices
        y, X = self.coxph.create_matrices(data, create_dummy=False)

        # Fitting
        self.coxph.fit(y, X)

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

    def test_coxph_snp1_inter(self):
        """Tests coxph regression with the first SNP (interaction)."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp1"]]
        data = data.rename(columns={"snp1": "geno"})

        # Preparing the matrices
        y, X = self.coxph_inter.create_matrices(data, create_dummy=False)

        # Fitting
        self.coxph_inter.fit(y, X)

        # Checking the results (according to SAS)
        # SAS doesn't compute a hazard ratio (and its 95% CI) for interaction
        # (since it would be interpretable)
        self.assertAlmostEqual(
            0.047567074771854, self.coxph_inter.results.coef, places=4,
        )
        self.assertAlmostEqual(
            0.06434824185209, self.coxph_inter.results.std_err, places=5,
        )
        self.assertAlmostEqual(
            np.exp(0.047567074771854), self.coxph_inter.results.hr, places=4,
        )
        self.assertAlmostEqual(
            0.54643626988504, self.coxph_inter.results.z_value**2, places=3,
        )
        self.assertAlmostEqual(
            0.45977749951036, self.coxph_inter.results.p_value, places=3,
        )

    def test_coxph_snp2(self):
        """Tests coxph regression with the second SNP."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp2"]]
        data = data.rename(columns={"snp2": "geno"})

        # Preparing the matrices
        y, X = self.coxph.create_matrices(data, create_dummy=False)

        # Fitting
        self.coxph.fit(y, X)

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

    def test_coxph_snp2_inter(self):
        """Tests coxph regression with the second SNP (interaction)."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp2"]]
        data = data.rename(columns={"snp2": "geno"})

        # Preparing the matrices
        y, X = self.coxph_inter.create_matrices(data, create_dummy=False)

        # Fitting
        self.coxph_inter.fit(y, X)

        # Checking the results (according to SAS)
        # SAS doesn't compute a hazard ratio (and its 95% CI) for interaction
        # (since it would be interpretable)
        self.assertAlmostEqual(
            0.0174236370729153, self.coxph_inter.results.coef, places=5,
        )
        self.assertAlmostEqual(
            0.0221904396707763, self.coxph_inter.results.std_err,
        )
        self.assertAlmostEqual(
            np.exp(0.0174236370729153), self.coxph_inter.results.hr, places=5,
        )
        self.assertAlmostEqual(
            0.6165180814095, self.coxph_inter.results.z_value**2, places=4,
        )
        self.assertAlmostEqual(
            0.4323441392819500, self.coxph_inter.results.p_value, places=4,
        )

    def test_coxph_snp3(self):
        """Tests coxph regression with the third SNP."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp3"]]
        data = data.rename(columns={"snp3": "geno"})

        # Preparing the matrices
        y, X = self.coxph.create_matrices(data, create_dummy=False)

        # Fitting
        self.coxph.fit(y, X)

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

    def test_coxph_snp3_inter(self):
        """Tests coxph regression with the third SNP (interaction)."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp3"]]
        data = data.rename(columns={"snp3": "geno"})

        # Preparing the matrices
        y, X = self.coxph_inter.create_matrices(data, create_dummy=False)

        # Fitting
        self.coxph_inter.fit(y, X)

        # Checking the results (according to SAS)
        # SAS doesn't compute a hazard ratio (and its 95% CI) for interaction
        # (since it would be interpretable)
        self.assertAlmostEqual(
            0.0058742328153452, self.coxph_inter.results.coef, places=5,
        )
        self.assertAlmostEqual(
            0.0335085402386551, self.coxph_inter.results.std_err,
        )
        self.assertAlmostEqual(
            np.exp(0.0058742328153452), self.coxph_inter.results.hr, places=5,
        )
        self.assertAlmostEqual(
            0.0307320331310296, self.coxph_inter.results.z_value**2, places=4,
        )
        self.assertAlmostEqual(
            0.86083955715083, self.coxph_inter.results.p_value, places=4,
        )

    def test_coxph_snp4(self):
        """Tests coxph regression with the third SNP."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp4"]]
        data = data.rename(columns={"snp4": "geno"})

        # Preparing the matrices
        y, X = self.coxph.create_matrices(data, create_dummy=False)

        # Fitting
        with self.assertRaises(StatsError):
            self.coxph.fit(y, X)

        # Checking the results (according to SAS)
        self.assertTrue(np.isnan(self.coxph.results.coef))
        self.assertTrue(np.isnan(self.coxph.results.std_err))
        self.assertTrue(np.isnan(self.coxph.results.hr))
        self.assertTrue(np.isnan(self.coxph.results.hr_lower_ci))
        self.assertTrue(np.isnan(self.coxph.results.hr_upper_ci))
        self.assertTrue(np.isnan(self.coxph.results.z_value))
        self.assertTrue(np.isnan(self.coxph.results.p_value))

    def test_coxph_snp4_inter(self):
        """Tests coxph regression with the third SNP (interaction)."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp4"]]
        data = data.rename(columns={"snp4": "geno"})

        # Preparing the matrices
        y, X = self.coxph_inter.create_matrices(data, create_dummy=False)

        # Fitting
        with self.assertRaises(StatsError):
            self.coxph_inter.fit(y, X)

        # Checking the results (according to SAS)
        self.assertTrue(np.isnan(self.coxph_inter.results.coef))
        self.assertTrue(np.isnan(self.coxph_inter.results.std_err))
        self.assertTrue(np.isnan(self.coxph_inter.results.hr))
        self.assertTrue(np.isnan(self.coxph_inter.results.hr_lower_ci))
        self.assertTrue(np.isnan(self.coxph_inter.results.hr_upper_ci))
        self.assertTrue(np.isnan(self.coxph_inter.results.z_value))
        self.assertTrue(np.isnan(self.coxph_inter.results.p_value))
