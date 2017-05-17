

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

from geneparse.dataframe import DataFrameReader

from ..statistics.models.mixedlm import StatsMixedLM

from .. import analysis
from .. import subscribers
from .. import modelspec as spec
from ..phenotypes.dummy import _DummyPhenotypes


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestStatsMixedLM(unittest.TestCase):
    """Tests the 'StatsMixedLM' class."""
    @classmethod
    def setUpClass(cls):
        # Loading the data
        cls.data = pd.read_csv(
            resource_filename(__name__, "data/statistics/mixedlm.txt.bz2"),
            sep="\t",
            compression="bz2",
        )

        # Creating the index
        cls.data = cls.data.set_index("sampleid", drop=False)

        # Creating the dummy phenotype container
        cls.phenotypes = _DummyPhenotypes()
        cls.phenotypes.data = cls.data.drop(
            [col for col in cls.data.columns if col.startswith("snp")],
            axis=1,
        )

        # Permuting the sample to add a bit of randomness
        new_sample_order = np.random.permutation(cls.data.index)

        # Creating the genotypes data frame
        genotypes = cls.data.loc[
            new_sample_order,
            [col for col in cls.data.columns if col.startswith("snp")],
        ].copy()

        # Keeping only one copy of each data
        genotypes = genotypes[~genotypes.index.duplicated()]

        # Creating the mapping information
        map_info = pd.DataFrame(
            {"chrom": ["3", "3", "2", "22"],
             "pos": [1234, 2345, 3456, 4567],
             "a1": ["T", "C", "G", "A"],
             "a2": ["G", "A", "T", "C"]},
            index=["snp1", "snp2", "snp3", "snp4"],
        )

        # Creating the genotype parser
        cls.genotypes = DataFrameReader(
            dataframe=genotypes,
            map_info=map_info,
        )

    def setUp(self):
        # Resetting the model specification
        spec._reset()

        # Reordering the columns and the rows of the phenotypes data frame
        self.phenotypes.data = self.phenotypes.data.iloc[
            np.random.permutation(self.phenotypes.data.shape[0]),
            np.random.permutation(self.phenotypes.data.shape[1])
        ]

    def test_mixedlm_gwas(self):
        """Tests mixedlm regression for GWAS."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.SNPs, spec.phenotypes.age, spec.phenotypes.var1,
                        gender, visit],
            test="mixedlm",
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        gwas_results = subscriber._get_gwas_results()

        # Checking the number of results (should be 4)
        self.assertEqual(4, len(gwas_results.keys()))

        # Checking the first marker (snp1)
        results = gwas_results["snp1"]
        self.assertEqual("snp1", results["SNPs"]["name"])
        self.assertEqual("3", results["SNPs"]["chrom"])
        self.assertEqual(1234, results["SNPs"]["pos"])
        self.assertEqual("T", results["SNPs"]["minor"])
        self.assertEqual("G", results["SNPs"]["major"])
        self.assertAlmostEqual(0.17499166666666666, results["SNPs"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(37.45704065571509034, results["SNPs"]["coef"])
        self.assertAlmostEqual(8.3417880825867652, results["SNPs"]["std_err"],
                               places=4)
        self.assertAlmostEqual(21.1074364471796017,
                               results["SNPs"]["lower_ci"], places=4)
        self.assertAlmostEqual(53.806644864250579,
                               results["SNPs"]["upper_ci"], places=4)
        self.assertAlmostEqual(4.4902891664085249,
                               results["SNPs"]["z_value"], places=4)
        self.assertAlmostEqual(-np.log10(7.112654716978639e-06),
                               -np.log10(results["SNPs"]["p_value"]), places=4)

        # Checking the second marker (snp2)
        results = gwas_results["snp2"]
        self.assertEqual("snp2", results["SNPs"]["name"])
        self.assertEqual("3", results["SNPs"]["chrom"])
        self.assertEqual(2345, results["SNPs"]["pos"])
        self.assertEqual("C", results["SNPs"]["minor"])
        self.assertEqual("A", results["SNPs"]["major"])
        self.assertAlmostEqual(0.36666666666666664, results["SNPs"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(28.87619649886422479, results["SNPs"]["coef"])
        self.assertAlmostEqual(7.616420586507464, results["SNPs"]["std_err"],
                               places=4)
        self.assertAlmostEqual(13.9482864582001636,
                               results["SNPs"]["lower_ci"], places=4)
        self.assertAlmostEqual(43.8041065395282843,
                               results["SNPs"]["upper_ci"], places=4)
        self.assertAlmostEqual(3.7913080259798924,
                               results["SNPs"]["z_value"], places=4)
        self.assertAlmostEqual(-np.log10(1.498559584953707e-04),
                               -np.log10(results["SNPs"]["p_value"]), places=4)

        # Checking the third marker (snp3)
        results = gwas_results["snp3"]
        self.assertEqual("snp3", results["SNPs"]["name"])
        self.assertEqual("2", results["SNPs"]["chrom"])
        self.assertEqual(3456, results["SNPs"]["pos"])
        self.assertEqual("G", results["SNPs"]["minor"])
        self.assertEqual("T", results["SNPs"]["major"])
        self.assertAlmostEqual(0.40833333333333333, results["SNPs"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(21.61350866000438486, results["SNPs"]["coef"])
        self.assertAlmostEqual(6.4018199962254876, results["SNPs"]["std_err"],
                               places=4)
        self.assertAlmostEqual(9.0661720318940873,
                               results["SNPs"]["lower_ci"], places=4)
        self.assertAlmostEqual(34.160845288114686,
                               results["SNPs"]["upper_ci"], places=4)
        self.assertAlmostEqual(3.37615063727935283,
                               results["SNPs"]["z_value"], places=4)
        self.assertAlmostEqual(-np.log10(7.350766113720653e-04),
                               -np.log10(results["SNPs"]["p_value"]), places=4)

        # Checking the fourth marker (snp4)
        results = gwas_results["snp4"]
        self.assertEqual("snp4", results["SNPs"]["name"])
        self.assertEqual("22", results["SNPs"]["chrom"])
        self.assertEqual(4567, results["SNPs"]["pos"])
        self.assertEqual("A", results["SNPs"]["minor"])
        self.assertEqual("C", results["SNPs"]["major"])
        self.assertAlmostEqual(0.44166666666666665, results["SNPs"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(1.4891822461435404, results["SNPs"]["coef"])
        self.assertAlmostEqual(7.9840787818167422, results["SNPs"]["std_err"],
                               places=3)
        self.assertAlmostEqual(-14.1593246159476980,
                               results["SNPs"]["lower_ci"], places=3)
        self.assertAlmostEqual(17.1376891082347775,
                               results["SNPs"]["upper_ci"], places=3)
        self.assertAlmostEqual(0.1865189819437983,
                               results["SNPs"]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(8.520377946017625e-01),
                               -np.log10(results["SNPs"]["p_value"]), places=5)

    def test_mixedlm_snp1_reml(self):
        """Tests mixedlm regression with the first SNP (using REML)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp1, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit],
            test="mixedlm",
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
        self.assertEqual("G", results["snp1"]["major"])
        self.assertAlmostEqual(0.17499166666666666, results["snp1"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(37.45704065571509034, results["snp1"]["coef"])
        self.assertAlmostEqual(8.3417880825867652, results["snp1"]["std_err"],
                               places=4)
        self.assertAlmostEqual(21.1074364471796017,
                               results["snp1"]["lower_ci"], places=4)
        self.assertAlmostEqual(53.806644864250579,
                               results["snp1"]["upper_ci"], places=4)
        self.assertAlmostEqual(4.4902891664085249,
                               results["snp1"]["z_value"], places=4)
        self.assertAlmostEqual(-np.log10(7.112654716978639e-06),
                               -np.log10(results["snp1"]["p_value"]), places=4)

        # TODO: Check the other predictors

    def test_mixedlm_snp1_ml(self):
        """Tests mixedlm regression with the first SNP (using ML)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp1, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit],
            test=lambda: StatsMixedLM(reml=False),
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
        self.assertEqual("G", results["snp1"]["major"])
        self.assertAlmostEqual(0.17499166666666666, results["snp1"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(37.45704065571743513, results["snp1"]["coef"])
        self.assertAlmostEqual(7.986654823996228, results["snp1"]["std_err"],
                               places=5)
        self.assertAlmostEqual(21.8034848437317450,
                               results["snp1"]["lower_ci"], places=4)
        self.assertAlmostEqual(53.1105964677031253,
                               results["snp1"]["upper_ci"], places=4)
        self.assertAlmostEqual(4.6899536140182541,
                               results["snp1"]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(2.732669904137452e-06),
                               -np.log10(results["snp1"]["p_value"]), places=5)

        # TODO: Check the other predictors

    def test_mixedlm_snp1_inter_reml(self):
        """Tests mixedlm regression with the first SNP (using REML, inter)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # The interaction term
        inter = spec.interaction(spec.genotypes.snp1, spec.phenotypes.var1)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp1, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit, inter],
            test="mixedlm",
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
        self.assertEqual("G", results["snp1"]["major"])
        self.assertAlmostEqual(0.17499166666666666, results["snp1"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(1.357502192968099, results[inter.id]["coef"])
        self.assertAlmostEqual(0.6710947924603057,
                               results[inter.id]["std_err"], places=5)
        self.assertAlmostEqual(0.04218056953351801,
                               results[inter.id]["lower_ci"], places=5)
        self.assertAlmostEqual(2.6728238164026799,
                               results[inter.id]["upper_ci"], places=5)
        self.assertAlmostEqual(2.0228173548946042,
                               results[inter.id]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(4.309198170818540e-02),
                               -np.log10(results[inter.id]["p_value"]),
                               places=5)

        # TODO: Check the other predictors

    def test_mixedlm_snp1_inter_ml(self):
        """Tests mixedlm regression with the first SNP (using ML, inter)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # The interaction term
        inter = spec.interaction(spec.genotypes.snp1, spec.phenotypes.var1)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp1, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit, inter],
            test=lambda: StatsMixedLM(reml=False),
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
        self.assertEqual("G", results["snp1"]["major"])
        self.assertAlmostEqual(0.17499166666666666, results["snp1"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(1.3575021929662827, results[inter.id]["coef"])
        self.assertAlmostEqual(0.6366563415656629,
                               results[inter.id]["std_err"], places=6)
        self.assertAlmostEqual(0.1096786929685527,
                               results[inter.id]["lower_ci"], places=6)
        self.assertAlmostEqual(2.6053256929640130,
                               results[inter.id]["upper_ci"], places=6)
        self.assertAlmostEqual(2.1322369767462277,
                               results[inter.id]["z_value"], places=6)
        self.assertAlmostEqual(-np.log10(3.298737010780739e-02),
                               -np.log10(results[inter.id]["p_value"]),
                               places=5)

        # TODO: Check the other predictors

    def test_mixedlm_snp2_reml(self):
        """Tests mixedlm regression with the second SNP (using REML)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp2, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit],
            test="mixedlm",
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
        self.assertEqual(2345, results["snp2"]["pos"])
        self.assertEqual("C", results["snp2"]["minor"])
        self.assertEqual("A", results["snp2"]["major"])
        self.assertAlmostEqual(0.36666666666666664, results["snp2"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(28.87619649886422479, results["snp2"]["coef"])
        self.assertAlmostEqual(7.616420586507464, results["snp2"]["std_err"],
                               places=4)
        self.assertAlmostEqual(13.9482864582001636,
                               results["snp2"]["lower_ci"], places=4)
        self.assertAlmostEqual(43.8041065395282843,
                               results["snp2"]["upper_ci"], places=4)
        self.assertAlmostEqual(3.7913080259798924,
                               results["snp2"]["z_value"], places=4)
        self.assertAlmostEqual(-np.log10(1.498559584953707e-04),
                               -np.log10(results["snp2"]["p_value"]), places=4)

        # TODO: Check the other predictors

    def test_mixedlm_snp2_ml(self):
        """Tests mixedlm regression with the second SNP (using ML)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp2, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit],
            test=lambda: StatsMixedLM(reml=False),
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(28.87619649885465023, results["snp2"]["coef"])
        self.assertAlmostEqual(7.2921675410708389, results["snp2"]["std_err"],
                               places=4)
        self.assertAlmostEqual(14.5838107491238045,
                               results["snp2"]["lower_ci"], places=4)
        self.assertAlmostEqual(43.1685822485854942,
                               results["snp2"]["upper_ci"], places=4)
        self.assertAlmostEqual(3.9598920809510423,
                               results["snp2"]["z_value"], places=4)
        self.assertAlmostEqual(-np.log10(7.498364115088307e-05),
                               -np.log10(results["snp2"]["p_value"]), places=4)

        # TODO: Check the other predictors

    def test_mixedlm_snp2_inter_reml(self):
        """Tests mixedlm regression with the second SNP (using REML, inter)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # The interaction term
        inter = spec.interaction(spec.genotypes.snp2, spec.phenotypes.var1)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp2, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit, inter],
            test="mixedlm",
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
        self.assertEqual(2345, results["snp2"]["pos"])
        self.assertEqual("C", results["snp2"]["minor"])
        self.assertEqual("A", results["snp2"]["major"])
        self.assertAlmostEqual(0.36666666666666664, results["snp2"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(0.8952450482655012, results[inter.id]["coef"])
        self.assertAlmostEqual(0.5976670951727925,
                               results[inter.id]["std_err"], places=5)
        self.assertAlmostEqual(-0.2761609330178445,
                               results[inter.id]["lower_ci"], places=5)
        self.assertAlmostEqual(2.0666510295488472,
                               results[inter.id]["upper_ci"], places=5)
        self.assertAlmostEqual(1.497899174132508504,
                               results[inter.id]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(1.341594483012889e-01),
                               -np.log10(results[inter.id]["p_value"]),
                               places=5)

        # TODO: Check the other predictors

    def test_mixedlm_snp2_inter_ml(self):
        """Tests mixedlm regression with the second SNP (using ML, inter)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # The interaction term
        inter = spec.interaction(spec.genotypes.snp2, spec.phenotypes.var1)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp2, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit, inter],
            test=lambda: StatsMixedLM(reml=False),
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
        self.assertEqual(2345, results["snp2"]["pos"])
        self.assertEqual("C", results["snp2"]["minor"])
        self.assertEqual("A", results["snp2"]["major"])
        self.assertAlmostEqual(0.36666666666666664, results["snp2"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(0.8952450482657273, results[inter.id]["coef"])
        self.assertAlmostEqual(0.5669968515485548,
                               results[inter.id]["std_err"], places=6)
        self.assertAlmostEqual(-0.2160483601170435,
                               results[inter.id]["lower_ci"], places=5)
        self.assertAlmostEqual(2.006538456648498,
                               results[inter.id]["upper_ci"], places=5)
        self.assertAlmostEqual(1.578924196528916468,
                               results[inter.id]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(1.143534452552892e-01),
                               -np.log10(results[inter.id]["p_value"]),
                               places=5)

        # TODO: Check the other predictors

    def test_mixedlm_snp3_reml(self):
        """Tests mixedlm regression with the third SNP (using REML)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp3, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit],
            test="mixedlm",
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information
        self.assertEqual("snp3", results["snp3"]["name"])
        self.assertEqual("2", results["snp3"]["chrom"])
        self.assertEqual(3456, results["snp3"]["pos"])
        self.assertEqual("G", results["snp3"]["minor"])
        self.assertEqual("T", results["snp3"]["major"])
        self.assertAlmostEqual(0.40833333333333333, results["snp3"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(21.61350866000438486, results["snp3"]["coef"])
        self.assertAlmostEqual(6.4018199962254876, results["snp3"]["std_err"],
                               places=4)
        self.assertAlmostEqual(9.0661720318940873,
                               results["snp3"]["lower_ci"], places=4)
        self.assertAlmostEqual(34.160845288114686,
                               results["snp3"]["upper_ci"], places=4)
        self.assertAlmostEqual(3.37615063727935283,
                               results["snp3"]["z_value"], places=4)
        self.assertAlmostEqual(-np.log10(7.350766113720653e-04),
                               -np.log10(results["snp3"]["p_value"]), places=4)

        # TODO: Check the other predictors

    def test_mixedlm_snp3_ml(self):
        """Tests mixedlm regression with the third SNP (using ML)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp3, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit],
            test=lambda: StatsMixedLM(reml=False),
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information
        self.assertEqual("snp3", results["snp3"]["name"])
        self.assertEqual("2", results["snp3"]["chrom"])
        self.assertEqual(3456, results["snp3"]["pos"])
        self.assertEqual("G", results["snp3"]["minor"])
        self.assertEqual("T", results["snp3"]["major"])
        self.assertAlmostEqual(0.40833333333333333, results["snp3"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(21.61350866001425786, results["snp3"]["coef"])
        self.assertAlmostEqual(6.1292761423280124, results["snp3"]["std_err"],
                               places=4)
        self.assertAlmostEqual(9.6003481697507578,
                               results["snp3"]["lower_ci"], places=4)
        self.assertAlmostEqual(33.6266691502777562,
                               results["snp3"]["upper_ci"], places=4)
        self.assertAlmostEqual(3.52627425459820243,
                               results["snp3"]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(4.214502895376615e-04),
                               -np.log10(results["snp3"]["p_value"]), places=4)

        # TODO: Check the other predictors

    def test_mixedlm_snp3_inter_reml(self):
        """Tests mixedlm regression with the third SNP (using REML, inter)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # The interaction term
        inter = spec.interaction(spec.genotypes.snp3, spec.phenotypes.var1)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp3, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit, inter],
            test="mixedlm",
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information
        self.assertEqual("snp3", results["snp3"]["name"])
        self.assertEqual("2", results["snp3"]["chrom"])
        self.assertEqual(3456, results["snp3"]["pos"])
        self.assertEqual("G", results["snp3"]["minor"])
        self.assertEqual("T", results["snp3"]["major"])
        self.assertAlmostEqual(0.40833333333333333, results["snp3"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(0.9199422369515684, results[inter.id]["coef"])
        self.assertAlmostEqual(0.4498593164720226,
                               results[inter.id]["std_err"], places=5)
        self.assertAlmostEqual(0.03823417855659805,
                               results[inter.id]["lower_ci"], places=5)
        self.assertAlmostEqual(1.8016502953465388,
                               results[inter.id]["upper_ci"], places=5)
        self.assertAlmostEqual(2.0449553966473895,
                               results[inter.id]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(4.085925566922222e-02),
                               -np.log10(results[inter.id]["p_value"]),
                               places=4)

        # TODO: Check the other predictors

    def test_mixedlm_snp3_inter_ml(self):
        """Tests mixedlm regression with the third SNP (using ML, inter)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # The interaction term
        inter = spec.interaction(spec.genotypes.snp3, spec.phenotypes.var1)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp3, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit, inter],
            test=lambda: StatsMixedLM(reml=False),
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information
        self.assertEqual("snp3", results["snp3"]["name"])
        self.assertEqual("2", results["snp3"]["chrom"])
        self.assertEqual(3456, results["snp3"]["pos"])
        self.assertEqual("G", results["snp3"]["minor"])
        self.assertEqual("T", results["snp3"]["major"])
        self.assertAlmostEqual(0.40833333333333333, results["snp3"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(0.9199422369518273, results[inter.id]["coef"])
        self.assertAlmostEqual(0.4267740150554054,
                               results[inter.id]["std_err"], places=6)
        self.assertAlmostEqual(0.08348053790567811,
                               results[inter.id]["lower_ci"], places=5)
        self.assertAlmostEqual(1.7564039359979766,
                               results[inter.id]["upper_ci"], places=5)
        self.assertAlmostEqual(2.1555722806422435,
                               results[inter.id]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(3.111707895646454e-02),
                               -np.log10(results[inter.id]["p_value"]),
                               places=5)

        # TODO: Check the other predictors

    def test_mixedlm_snp4_reml(self):
        """Tests mixedlm regression with the fourth SNP (using REML)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp4, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit],
            test="mixedlm",
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information
        self.assertEqual("snp4", results["snp4"]["name"])
        self.assertEqual("22", results["snp4"]["chrom"])
        self.assertEqual(4567, results["snp4"]["pos"])
        self.assertEqual("A", results["snp4"]["minor"])
        self.assertEqual("C", results["snp4"]["major"])
        self.assertAlmostEqual(0.44166666666666665, results["snp4"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(1.4891822461435404, results["snp4"]["coef"])
        self.assertAlmostEqual(7.9840787818167422, results["snp4"]["std_err"],
                               places=3)
        self.assertAlmostEqual(-14.1593246159476980,
                               results["snp4"]["lower_ci"], places=3)
        self.assertAlmostEqual(17.1376891082347775,
                               results["snp4"]["upper_ci"], places=3)
        self.assertAlmostEqual(0.1865189819437983,
                               results["snp4"]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(8.520377946017625e-01),
                               -np.log10(results["snp4"]["p_value"]), places=5)

        # TODO: Check the other predictors

    def test_mixedlm_snp4_ml(self):
        """Tests mixedlm regression with the fourth SNP (using ML)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp4, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit],
            test=lambda: StatsMixedLM(reml=False),
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information
        self.assertEqual("snp4", results["snp4"]["name"])
        self.assertEqual("22", results["snp4"]["chrom"])
        self.assertEqual(4567, results["snp4"]["pos"])
        self.assertEqual("A", results["snp4"]["minor"])
        self.assertEqual("C", results["snp4"]["major"])
        self.assertAlmostEqual(0.44166666666666665, results["snp4"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(1.4891822461364390, results["snp4"]["coef"])
        self.assertAlmostEqual(7.6441729657793820, results["snp4"]["std_err"],
                               places=4)
        self.assertAlmostEqual(-13.4931214583858772,
                               results["snp4"]["lower_ci"], places=3)
        self.assertAlmostEqual(16.4714859506587565,
                               results["snp4"]["upper_ci"], places=3)
        self.assertAlmostEqual(0.1948127355049462,
                               results["snp4"]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(8.455395518199895e-01),
                               -np.log10(results["snp4"]["p_value"]), places=6)

        # TODO: Check the other predictors

    def test_mixedlm_snp4_inter_reml(self):
        """Tests mixedlm regression with the fourth SNP (using REML, inter)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # The interaction term
        inter = spec.interaction(spec.genotypes.snp4, spec.phenotypes.var1)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp4, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit, inter],
            test="mixedlm",
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information
        self.assertEqual("snp4", results["snp4"]["name"])
        self.assertEqual("22", results["snp4"]["chrom"])
        self.assertEqual(4567, results["snp4"]["pos"])
        self.assertEqual("A", results["snp4"]["minor"])
        self.assertEqual("C", results["snp4"]["major"])
        self.assertAlmostEqual(0.44166666666666665, results["snp4"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(0.05832811192587545, results[inter.id]["coef"])
        self.assertAlmostEqual(0.7381058671562147,
                               results[inter.id]["std_err"], places=4)
        self.assertAlmostEqual(-1.388332804478011,
                               results[inter.id]["lower_ci"], places=4)
        self.assertAlmostEqual(1.504989028329762,
                               results[inter.id]["upper_ci"], places=4)
        self.assertAlmostEqual(0.07902404590089884,
                               results[inter.id]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(9.370134970059800e-01),
                               -np.log10(results[inter.id]["p_value"]),
                               places=6)

        # TODO: Check the other predictors

    def test_mixedlm_snp4_inter_ml(self):
        """Tests mixedlm regression with the fourth SNP (using ML, inter)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)
        visit = spec.factor(spec.phenotypes.visit)

        # The interaction term
        inter = spec.interaction(spec.genotypes.snp4, spec.phenotypes.var1)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome={"outcome": spec.phenotypes.pheno3,
                     "groups": spec.phenotypes.sampleid},
            predictors=[spec.genotypes.snp4, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, visit, inter],
            test=lambda: StatsMixedLM(reml=False),
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information
        self.assertEqual("snp4", results["snp4"]["name"])
        self.assertEqual("22", results["snp4"]["chrom"])
        self.assertEqual(4567, results["snp4"]["pos"])
        self.assertEqual("A", results["snp4"]["minor"])
        self.assertEqual("C", results["snp4"]["major"])
        self.assertAlmostEqual(0.44166666666666665, results["snp4"]["maf"])

        # Checking the marker statistics (according to R lme4)
        self.assertAlmostEqual(0.05832811192492895, results[inter.id]["coef"])
        self.assertAlmostEqual(0.7002288518048638,
                               results[inter.id]["std_err"], places=5)
        self.assertAlmostEqual(-1.314095218548438,
                               results[inter.id]["lower_ci"], places=4)
        self.assertAlmostEqual(1.430751442398297,
                               results[inter.id]["upper_ci"], places=4)
        self.assertAlmostEqual(0.08329864125790624,
                               results[inter.id]["z_value"], places=6)
        self.assertAlmostEqual(-np.log10(9.336140806606035e-01),
                               -np.log10(results[inter.id]["p_value"]),
                               places=6)

        # TODO: Check the other predictors
