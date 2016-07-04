

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
from ..statistics.models.logistic import StatsLogistic


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestStatsLogistic(unittest.TestCase):
    """Tests the 'StatsLogistic' class."""
    @classmethod
    def setUpClass(cls):
        cls.logistic = StatsLogistic(
            outcome="pheno2",
            predictors=["age", "var1", "C(gender)", "geno"],
            interaction=None,
        )
        cls.logistic_inter = StatsLogistic(
            outcome="pheno2",
            predictors=["age", "var1", "C(gender)", "geno"],
            interaction="var1",
        )
        cls.logistic_inter_cat = StatsLogistic(
            outcome="pheno2",
            predictors=["age", "var1", "C(gender)", "geno"],
            interaction="C(gender)",
        )

    def setUp(self):
        # The data
        self.data = pd.read_csv(
            resource_filename(__name__, "data/statistics/logistic.txt.bz2"),
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

    def test_logistic_snp1_full(self):
        """Tests logistic regression with the first SNP (full)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno2", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(
            columns={"snp1": "geno"},
        )

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.logistic.create_matrices(self.dummy)
        self.assertFalse("geno" in X.columns)

        # Merging with genotype
        y, X = self.logistic.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.logistic.fit(y, X)

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

    def test_logistic_inter_snp1_inter_full(self):
        """Tests logistic regression with the first SNP (interaction, full)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno2", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(
            columns={"snp1": "geno"},
        )

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.logistic_inter.create_matrices(self.dummy)
        self.assertFalse("geno" in X.columns)

        # Merging with genotype
        y, X = self.logistic_inter.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.logistic_inter.fit(y, X)

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            0.001531148296819, self.logistic_inter.results.coef, places=6,
        )
        self.assertAlmostEqual(
            0.0423504567007674, self.logistic_inter.results.std_err,
        )
        self.assertAlmostEqual(
            -0.0814742215655, self.logistic_inter.results.lower_ci, places=6,
        )
        self.assertAlmostEqual(
            0.08453651815914, self.logistic_inter.results.upper_ci,
        )
        self.assertAlmostEqual(
            0.0013071285938734, self.logistic_inter.results.z_value**2,
            places=6,
        )
        self.assertAlmostEqual(
            0.97115937855364, self.logistic_inter.results.p_value, places=5,
        )

    def test_logistic_inter_snp1_inter_category_full(self):
        """Tests logistic regression first SNP (inter, full, category)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno2", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(
            columns={"snp1": "geno"},
        )

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.logistic_inter_cat.create_matrices(self.dummy)
        self.assertFalse("geno" in X.columns)

        # Merging with genotype
        y, X = self.logistic_inter_cat.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.logistic_inter_cat.fit(y, X)

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            -0.0089544870073693, self.logistic_inter_cat.results.coef,
            places=4,
        )
        self.assertAlmostEqual(
            1.19878325025509, self.logistic_inter_cat.results.std_err,
            places=5,
        )
        self.assertAlmostEqual(
            -2.35852648277723, self.logistic_inter_cat.results.lower_ci,
            places=4,
        )
        self.assertAlmostEqual(
            2.34061750876249, self.logistic_inter_cat.results.upper_ci,
            places=5,
        )
        self.assertAlmostEqual(
            0.0000557956175619, self.logistic_inter_cat.results.z_value**2,
            places=6,
        )
        self.assertAlmostEqual(
            0.99404013987338, self.logistic_inter_cat.results.p_value,
            places=5,
        )

    def test_logistic_snp1(self):
        """Tests logistic regression with the first SNP."""
        # Preparing the data
        data = self.data[["pheno2", "age", "var1", "gender", "snp1"]].rename(
            columns={"snp1": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.logistic.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.logistic.fit(y, X)

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

    def test_logistic_snp1_inter(self):
        """Tests logistic regression with the first SNP (interaction)."""
        # Preparing the data
        data = self.data[["pheno2", "age", "var1", "gender", "snp1"]].rename(
            columns={"snp1": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.logistic_inter.create_matrices(
            self.dummy, create_dummy=False,
        )

        # Fitting
        self.logistic_inter.fit(y, X)

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            0.0015311482968189, self.logistic_inter.results.coef, places=6,
        )
        self.assertAlmostEqual(
            0.0423504567007674, self.logistic_inter.results.std_err,
        )
        self.assertAlmostEqual(
            -0.0814742215655, self.logistic_inter.results.lower_ci, places=6,
        )
        self.assertAlmostEqual(
            0.08453651815914, self.logistic_inter.results.upper_ci,
        )
        self.assertAlmostEqual(
            0.0013071285938733, self.logistic_inter.results.z_value**2,
            places=6,
        )
        self.assertAlmostEqual(
            0.9711593785536400, self.logistic_inter.results.p_value, places=5,
        )

    def test_logistic_snp2(self):
        """Tests logistic regression with the second SNP."""
        # Preparing the data
        data = self.data[["pheno2", "age", "var1", "gender", "snp2"]].rename(
            columns={"snp2": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.logistic.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.logistic.fit(y, X)

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

    def test_logistic_snp2_inter(self):
        """Tests logistic regression with the second SNP (interaction)."""
        # Preparing the data
        data = self.data[["pheno2", "age", "var1", "gender", "snp2"]].rename(
            columns={"snp2": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.logistic_inter.create_matrices(
            self.dummy, create_dummy=False,
        )

        # Fitting
        self.logistic_inter.fit(y, X)

        # Checking the results (according to SAS)
        self.assertAlmostEqual(
            0.0239292800721822, self.logistic_inter.results.coef, places=5,
        )
        self.assertAlmostEqual(
            0.0342619407842591, self.logistic_inter.results.std_err, places=6,
        )
        self.assertAlmostEqual(
            -0.0432228899054096, self.logistic_inter.results.lower_ci,
            places=5,
        )
        self.assertAlmostEqual(
            0.09108145004977, self.logistic_inter.results.upper_ci, places=5,
        )
        self.assertAlmostEqual(
            0.48779275460699, self.logistic_inter.results.z_value**2, places=3,
        )
        self.assertAlmostEqual(
            0.48491356159603, self.logistic_inter.results.p_value, places=4,
        )

    def test_logistic_snp3(self):
        """Tests logistic regression with the third SNP (raises StatsError)."""
        # Preparing the data
        data = self.data[["pheno2", "age", "var1", "gender", "snp3"]].rename(
            columns={"snp3": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.logistic.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        with self.assertRaises(StatsError):
            self.logistic.fit(y, X)

        # All the value should be NaN
        self.assertTrue(np.isnan(self.logistic.results.coef))
        self.assertTrue(np.isnan(self.logistic.results.std_err))
        self.assertTrue(np.isnan(self.logistic.results.lower_ci))
        self.assertTrue(np.isnan(self.logistic.results.upper_ci))
        self.assertTrue(np.isnan(self.logistic.results.z_value))
        self.assertTrue(np.isnan(self.logistic.results.p_value))

    def test_logistic_snp3_inter(self):
        """Tests logistic regression third SNP (raises StatsError, inter)."""
        # Preparing the data
        data = self.data[["pheno2", "age", "var1", "gender", "snp3"]].rename(
            columns={"snp3": "geno"},
        )

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.logistic_inter.create_matrices(
            self.dummy, create_dummy=False,
        )

        # Fitting
        with self.assertRaises(StatsError):
            self.logistic_inter.fit(y, X)

        # All the value should be NaN
        self.assertTrue(np.isnan(self.logistic_inter.results.coef))
        self.assertTrue(np.isnan(self.logistic_inter.results.std_err))
        self.assertTrue(np.isnan(self.logistic_inter.results.lower_ci))
        self.assertTrue(np.isnan(self.logistic_inter.results.upper_ci))
        self.assertTrue(np.isnan(self.logistic_inter.results.z_value))
        self.assertTrue(np.isnan(self.logistic_inter.results.p_value))
