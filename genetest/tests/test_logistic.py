

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

from .. import analysis
from .. import subscribers
from .. import modelspec as spec
from ..phenotypes.dummy import _DummyPhenotypes
from ..genotypes.dummy import _DummyGenotypes


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestStatsLogistic(unittest.TestCase):
    """Tests the 'StatsLogistic' class."""
    @classmethod
    def setUpClass(cls):
        # Loading the data
        cls.data = pd.read_csv(
            resource_filename(__name__, "data/statistics/logistic.txt.bz2"),
            sep="\t",
            compression="bz2",
        )

        # Creating the index
        cls.data["sample"] = [
            "s{}".format(i+1) for i in range(cls.data.shape[0])
        ]
        cls.data = cls.data.set_index("sample")

        # Creating the dummy phenotype container
        cls.phenotypes = _DummyPhenotypes()
        cls.phenotypes.data = cls.data.drop(
            ["snp{}".format(i+1) for i in range(3)],
            axis=1,
        )

        # Creating the dummy genotype container
        cls.genotypes = _DummyGenotypes()
        cls.genotypes.data = cls.data.drop(
            ["pheno2", "age", "var1", "gender"],
            axis=1,
        )
        cls.genotypes.snp_info = {
            "snp1": {"chrom": "3", "pos": 1234, "major": "C", "minor": "T"},
            "snp2": {"chrom": "3", "pos": 9618, "major": "A", "minor": "C"},
            "snp3": {"chrom": "2", "pos": 1519, "major": "T", "minor": "G"},
        }

    def setUp(self):
        # Resetting the model specification
        spec._reset()

        # Reordering the columns and the rows of the genotype data frame
        self.genotypes.data = self.genotypes.data.iloc[
            np.random.permutation(self.genotypes.data.shape[0]),
            np.random.permutation(self.genotypes.data.shape[1])
        ]

        # Reordering the columns and the rows of the phenotype data frame
        self.phenotypes.data = self.phenotypes.data.iloc[
            np.random.permutation(self.phenotypes.data.shape[0]),
            np.random.permutation(self.phenotypes.data.shape[1])
        ]

    def test_logistic_gwas(self):
        """Tests logistic regression with the first SNP (full)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno2,
            predictors=[spec.SNPs, spec.phenotypes.age,
                        spec.phenotypes.var1, gender],
            test="logistic",
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        gwas_results = subscriber._get_gwas_results()

        # Checking the number of results (should be 2)
        self.assertEqual(2, len(gwas_results.keys()))

        # Checking the first marker (snp1)
        results = gwas_results["snp1"]
        self.assertEqual("snp1", results["SNPs"]["name"])
        self.assertEqual("3", results["SNPs"]["chrom"])
        self.assertEqual(1234, results["SNPs"]["pos"])
        self.assertEqual("T", results["SNPs"]["minor"])
        self.assertEqual("C", results["SNPs"]["major"])

        # Checking the marker statistics (according to SAS)
        self.assertAlmostEqual(-2.24198635855498, results["SNPs"]["coef"],
                               places=5)
        self.assertAlmostEqual(0.59759558908668, results["SNPs"]["std_err"],
                               places=6)
        self.assertAlmostEqual(-3.41325219048488, results["SNPs"]["lower_ci"],
                               places=5)
        self.assertAlmostEqual(-1.07072052662507, results["SNPs"]["upper_ci"],
                               places=5)
        self.assertAlmostEqual(14.0750894991982, results["SNPs"]["t_value"]**2,
                               places=4)
        self.assertAlmostEqual(-np.log10(0.0001756548178104),
                               -np.log10(results["SNPs"]["p_value"]),
                               places=5)

        # Checking the second marker (snp2)
        results = gwas_results["snp2"]
        self.assertEqual("snp2", results["SNPs"]["name"])
        self.assertEqual("3", results["SNPs"]["chrom"])
        self.assertEqual(9618, results["SNPs"]["pos"])
        self.assertEqual("C", results["SNPs"]["minor"])
        self.assertEqual("A", results["SNPs"]["major"])

        # Checking the marker statistics (according to SAS)
        self.assertAlmostEqual(1.12532308347075, results["SNPs"]["coef"],
                               places=5)
        self.assertAlmostEqual(0.45211815097241, results["SNPs"]["std_err"],
                               places=6)
        self.assertAlmostEqual(0.23918779080797, results["SNPs"]["lower_ci"],
                               places=5)
        self.assertAlmostEqual(2.01145837613353, results["SNPs"]["upper_ci"],
                               places=5)
        self.assertAlmostEqual(6.19513207316499, results["SNPs"]["t_value"]**2,
                               places=4)
        self.assertAlmostEqual(-np.log10(0.0128102164253392),
                               -np.log10(results["SNPs"]["p_value"]),
                               places=4)

    @unittest.skip("Not implemented")
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

    @unittest.skip("Not implemented")
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
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno2,
            predictors=[spec.genotypes.snp1, spec.phenotypes.age,
                        spec.phenotypes.var1, gender],
            test="logistic",
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information
        self.assertEqual("snp1", results["snp1"]["name"])
        self.assertEqual("3", results["snp1"]["chrom"])
        self.assertEqual(1234, results["snp1"]["pos"])
        self.assertEqual("T", results["snp1"]["minor"])
        self.assertEqual("C", results["snp1"]["major"])

        # Checking the marker statistics (according to SAS)
        self.assertAlmostEqual(-2.24198635855498, results["snp1"]["coef"],
                               places=5)
        self.assertAlmostEqual(0.59759558908668, results["snp1"]["std_err"],
                               places=6)
        self.assertAlmostEqual(-3.41325219048488, results["snp1"]["lower_ci"],
                               places=5)
        self.assertAlmostEqual(-1.07072052662507, results["snp1"]["upper_ci"],
                               places=5)
        self.assertAlmostEqual(14.0750894991982, results["snp1"]["t_value"]**2,
                               places=4)
        self.assertAlmostEqual(-np.log10(0.0001756548178104),
                               -np.log10(results["snp1"]["p_value"]),
                               places=5)

    @unittest.skip("Not implemented")
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
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno2,
            predictors=[spec.genotypes.snp2, spec.phenotypes.age,
                        spec.phenotypes.var1, gender],
            test="logistic",
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information
        self.assertEqual("snp2", results["snp2"]["name"])
        self.assertEqual("3", results["snp2"]["chrom"])
        self.assertEqual(9618, results["snp2"]["pos"])
        self.assertEqual("C", results["snp2"]["minor"])
        self.assertEqual("A", results["snp2"]["major"])

        # Checking the marker statistics (according to SAS)
        self.assertAlmostEqual(1.12532308347075, results["snp2"]["coef"],
                               places=5)
        self.assertAlmostEqual(0.45211815097241, results["snp2"]["std_err"],
                               places=6)
        self.assertAlmostEqual(0.23918779080797, results["snp2"]["lower_ci"],
                               places=5)
        self.assertAlmostEqual(2.01145837613353, results["snp2"]["upper_ci"],
                               places=5)
        self.assertAlmostEqual(6.19513207316499, results["snp2"]["t_value"]**2,
                               places=4)
        self.assertAlmostEqual(-np.log10(0.0128102164253392),
                               -np.log10(results["snp2"]["p_value"]),
                               places=4)

    @unittest.skip("Not implemented")
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
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno2,
            predictors=[spec.genotypes.snp3, spec.phenotypes.age,
                        spec.phenotypes.var1, gender],
            test="logistic",
        )

        # Performing the analysis and retrieving the results
        with self.assertRaises(StatsError):
            analysis.execute(
                self.phenotypes, self.genotypes, modelspec,
                subscribers=[subscribers.ResultsMemory()],
            )

    @unittest.skip("Not implemented")
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
