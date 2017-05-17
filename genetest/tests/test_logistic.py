

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pkg_resources import resource_filename

from geneparse.dataframe import DataFrameReader

from ..statistics.core import StatsError

from .. import analysis
from .. import subscribers
from .. import modelspec as spec
from ..phenotypes.dummy import _DummyPhenotypes


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

        # Permuting the sample to add a bit of randomness
        new_sample_order = np.random.permutation(cls.data.index)

        # Creating the genotypes data frame
        genotypes = cls.data.loc[
            new_sample_order,
            [_ for _ in cls.data.columns if _.startswith("snp")],
        ].copy()

        # Creating the mapping information
        map_info = pd.DataFrame(
            {"chrom": ["3", "3", "2"],
             "pos": [1234, 9618, 1519],
             "a1": ["T", "C", "G"],
             "a2": ["C", "A", "T"]},
            index=["snp1", "snp2", "snp3"],
        )

        # Creating the genotype parser
        cls.genotypes = DataFrameReader(
            dataframe=genotypes,
            map_info=map_info,
        )

        # Creating a temporary directory
        cls.tmp_dir = TemporaryDirectory(prefix="genetest_test_logistic_")

    def setUp(self):
        # Resetting the model specification
        spec._reset()

        # Reordering the columns and the rows of the phenotype data frame
        self.phenotypes.data = self.phenotypes.data.iloc[
            np.random.permutation(self.phenotypes.data.shape[0]),
            np.random.permutation(self.phenotypes.data.shape[1])
        ]

    @classmethod
    def tearDownClass(cls):
        # Cleaning the temporary directory
        cls.tmp_dir.cleanup()

    def test_logistic_gwas(self):
        """Tests logistic regression for GWAS."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno2,
            predictors=[spec.SNPs, spec.phenotypes.age,
                        spec.phenotypes.var1, gender],
            test="logistic",
        )

        # The output prefix
        out_prefix = os.path.join(self.tmp_dir.name, "results")

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber], output_prefix=out_prefix,
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
        self.assertAlmostEqual(0.3749333333333334, results["SNPs"]["maf"])

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
        self.assertAlmostEqual(0.41590833333333332, results["SNPs"]["maf"])

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

        # There should be a file for the failed snp3
        self.assertTrue(os.path.isfile(out_prefix + "_failed_snps.txt"))
        with open(out_prefix + "_failed_snps.txt") as f:
            self.assertEqual(
                [["snp3", "Perfect separation detected, results not "
                          "available"]],
                [line.split("\t") for line in f.read().splitlines()],
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
        self.assertAlmostEqual(0.3749333333333334, results["snp1"]["maf"])

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

        # TODO: Check the other predictors

    def test_logistic_snp1_inter(self):
        """Tests logistic regression with the first SNP (interaction)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # The interaction term
        inter = spec.interaction(spec.phenotypes.var1, spec.genotypes.snp1)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno2,
            predictors=[spec.genotypes.snp1, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, inter],
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
        self.assertAlmostEqual(0.3749333333333334, results["snp1"]["maf"])

        # Checking the marker statistics (according to SAS)
        self.assertAlmostEqual(0.0015311482968189, results[inter.id]["coef"],
                               places=6)
        self.assertAlmostEqual(0.0423504567007674,
                               results[inter.id]["std_err"])
        self.assertAlmostEqual(-0.0814742215655, results[inter.id]["lower_ci"],
                               places=6)
        self.assertAlmostEqual(0.08453651815914, results[inter.id]["upper_ci"])
        self.assertAlmostEqual(0.0013071285938733,
                               results[inter.id]["t_value"]**2, places=6)
        self.assertAlmostEqual(0.9711593785536400,
                               results[inter.id]["p_value"], places=5)

        # TODO: Check the other predictors

    def test_logistic_snp1_inter_categorical(self):
        """Tests logistic regression first SNP (inter, category)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # The interaction term
        inter = spec.interaction(gender, spec.genotypes.snp1)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno2,
            predictors=[spec.genotypes.snp1, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, inter],
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
        self.assertAlmostEqual(0.3749333333333334, results["snp1"]["maf"])

        # Checking the marker statistics (according to SAS)
        col = inter.id + ":level.2"
        self.assertAlmostEqual(-0.0089544870073693, results[col]["coef"],
                               places=4)
        self.assertAlmostEqual(1.19878325025509, results[col]["std_err"],
                               places=5)
        self.assertAlmostEqual(-2.35852648277723, results[col]["lower_ci"],
                               places=4)
        self.assertAlmostEqual(2.34061750876249, results[col]["upper_ci"],
                               places=5)
        self.assertAlmostEqual(np.log10(0.0000557956175619),
                               np.log10(results[col]["t_value"]**2), places=2)
        self.assertAlmostEqual(0.99404013987338, results[col]["p_value"],
                               places=5)

        # TODO: Check the other predictors

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
        self.assertAlmostEqual(0.41590833333333332, results["snp2"]["maf"])

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

        # TODO: Check the other predictors

    def test_logistic_snp2_inter(self):
        """Tests logistic regression with the second SNP (interaction)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # The interaction term
        inter = spec.interaction(spec.phenotypes.var1, spec.genotypes.snp2)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno2,
            predictors=[spec.genotypes.snp2, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, inter],
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
        self.assertAlmostEqual(0.41590833333333332, results["snp2"]["maf"])

        # Checking the marker statistics (according to SAS)
        self.assertAlmostEqual(0.0239292800721822, results[inter.id]["coef"],
                               places=5)
        self.assertAlmostEqual(0.0342619407842591,
                               results[inter.id]["std_err"], places=6)
        self.assertAlmostEqual(-0.0432228899054096,
                               results[inter.id]["lower_ci"], places=5)
        self.assertAlmostEqual(0.09108145004977, results[inter.id]["upper_ci"],
                               places=5)
        self.assertAlmostEqual(0.48779275460699,
                               results[inter.id]["t_value"]**2, places=3)
        self.assertAlmostEqual(0.48491356159603, results[inter.id]["p_value"],
                               places=4)

        # TODO: Check the other predictors

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
        with self.assertRaises(StatsError) as cm:
            analysis.execute(
                self.phenotypes, self.genotypes, modelspec,
                subscribers=[subscribers.ResultsMemory()],
            )
        self.assertEqual("Perfect separation detected, results not available",
                         str(cm.exception))

    def test_logistic_snp3_inter(self):
        """Tests logistic regression third SNP (raises StatsError, inter)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # The interaction term
        inter = spec.interaction(spec.phenotypes.var1, spec.genotypes.snp2)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=spec.phenotypes.pheno2,
            predictors=[spec.genotypes.snp3, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, inter],
            test="logistic",
        )

        # Performing the analysis and retrieving the results
        with self.assertRaises(StatsError) as cm:
            analysis.execute(
                self.phenotypes, self.genotypes, modelspec,
                subscribers=[subscribers.ResultsMemory()],
            )
        self.assertEqual("Perfect separation detected, results not available",
                         str(cm.exception))
