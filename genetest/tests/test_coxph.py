

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


class TestStatsCoxPH(unittest.TestCase):
    """Tests the 'StatsCoxPH' class."""
    @classmethod
    def setUpClass(cls):
        # Loading the data
        cls.data = pd.read_csv(
            resource_filename(__name__, "data/statistics/coxph.txt.bz2"),
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
            ["snp{}".format(i+1) for i in range(4)],
            axis=1,
        )

        # Creating the dummy genotype container
        cls.genotypes = _DummyGenotypes()
        cls.genotypes.data = cls.data.drop(
            ["tte", "event", "age", "var1", "gender"],
            axis=1,
        )
        cls.genotypes.snp_info = {
            "snp1": {"chrom": "3", "pos": 1234, "major": "C", "minor": "T"},
            "snp2": {"chrom": "3", "pos": 9618, "major": "A", "minor": "C"},
            "snp3": {"chrom": "2", "pos": 1519, "major": "T", "minor": "G"},
            "snp4": {"chrom": "1", "pos": 5871, "major": "A", "minor": "G"},
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

    def test_coxph_gwas(self):
        """Tests coxph regression with the first SNP (full)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=dict(tte=spec.phenotypes.tte, event=spec.phenotypes.event),
            predictors=[spec.SNPs, spec.phenotypes.age,
                        spec.phenotypes.var1, gender],
            test="coxph",
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        gwas_results = subscriber._get_gwas_results()

        # Checking the number of results (should be 3)
        self.assertEqual(3, len(gwas_results.keys()))

        # Checking the first marker (snp1)
        results = gwas_results["snp1"]
        self.assertEqual("snp1", results["SNPs"]["name"])
        self.assertEqual("3", results["SNPs"]["chrom"])
        self.assertEqual(1234, results["SNPs"]["pos"])
        self.assertEqual("C", results["SNPs"]["minor"])
        self.assertEqual("T", results["SNPs"]["major"])
        self.assertAlmostEqual(0.48162903225806442, results["SNPs"]["maf"])

        # Checking the results (according to R)
        self.assertAlmostEqual(3.49824157323489571, results["SNPs"]["coef"])
        self.assertAlmostEqual(1.05744964734605418, results["SNPs"]["std_err"])
        self.assertAlmostEqual(33.0572720291935767, results["SNPs"]["hr"])
        self.assertAlmostEqual(4.1606792782002913,
                               results["SNPs"]["hr_lower_ci"], places=2)
        self.assertAlmostEqual(262.645390558627525,
                               results["SNPs"]["hr_upper_ci"], places=0)
        self.assertAlmostEqual(3.3081873751763462, results["SNPs"]["z_value"])
        self.assertAlmostEqual(-np.log10(0.0009390195967453607),
                               -np.log10(results["SNPs"]["p_value"]))

        # Checking the second marker (snp2)
        results = gwas_results["snp2"]
        self.assertEqual("snp2", results["SNPs"]["name"])
        self.assertEqual("3", results["SNPs"]["chrom"])
        self.assertEqual(9618, results["SNPs"]["pos"])
        self.assertEqual("C", results["SNPs"]["minor"])
        self.assertEqual("A", results["SNPs"]["major"])
        self.assertAlmostEqual(0.40833333333333333, results["SNPs"]["maf"])

        # Checking the results (according to R)
        self.assertAlmostEqual(1.13120383793364998, results["SNPs"]["coef"])
        self.assertAlmostEqual(0.30473046375826845, results["SNPs"]["std_err"])
        self.assertAlmostEqual(3.0993854129021017, results["SNPs"]["hr"])
        self.assertAlmostEqual(1.7056446002934191,
                               results["SNPs"]["hr_lower_ci"], places=3)
        self.assertAlmostEqual(5.631999735500465,
                               results["SNPs"]["hr_upper_ci"], places=2)
        self.assertAlmostEqual(3.7121455596608572, results["SNPs"]["z_value"])
        self.assertAlmostEqual(-np.log10(0.000205509736523446),
                               -np.log10(results["SNPs"]["p_value"]))

        # Checking the third marker (snp3)
        results = gwas_results["snp3"]
        self.assertEqual("snp3", results["SNPs"]["name"])
        self.assertEqual("2", results["SNPs"]["chrom"])
        self.assertEqual(1519, results["SNPs"]["pos"])
        self.assertEqual("G", results["SNPs"]["minor"])
        self.assertEqual("T", results["SNPs"]["major"])
        self.assertAlmostEqual(0.20833333333333334, results["SNPs"]["maf"])

        # Checking the results (according to R)
        self.assertAlmostEqual(-0.0069461317578822582, results["SNPs"]["coef"])
        self.assertAlmostEqual(0.39831755948730113, results["SNPs"]["std_err"],
                               places=5)
        self.assertAlmostEqual(0.9930779368551547, results["SNPs"]["hr"])
        self.assertAlmostEqual(0.4549197711311750,
                               results["SNPs"]["hr_lower_ci"], places=3)
        self.assertAlmostEqual(2.167863107413991,
                               results["SNPs"]["hr_upper_ci"], places=3)
        self.assertAlmostEqual(-0.01743867824160966,
                               results["SNPs"]["z_value"])
        self.assertAlmostEqual(0.9860866530659741, results["SNPs"]["p_value"])

    def test_coxph_snp1(self):
        """Tests coxph regression with the first SNP."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=dict(tte=spec.phenotypes.tte, event=spec.phenotypes.event),
            predictors=[spec.genotypes.snp1, spec.phenotypes.age,
                        spec.phenotypes.var1, gender],
            test="coxph",
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information (it's a flip)
        self.assertEqual("snp1", results["snp1"]["name"])
        self.assertEqual("3", results["snp1"]["chrom"])
        self.assertEqual(1234, results["snp1"]["pos"])
        self.assertEqual("C", results["snp1"]["minor"])
        self.assertEqual("T", results["snp1"]["major"])
        self.assertAlmostEqual(0.48162903225806442, results["snp1"]["maf"])

        # Checking the results (according to R)
        self.assertAlmostEqual(3.49824157323489571, results["snp1"]["coef"])
        self.assertAlmostEqual(1.05744964734605418, results["snp1"]["std_err"])
        self.assertAlmostEqual(33.0572720291935767, results["snp1"]["hr"])
        self.assertAlmostEqual(4.1606792782002913,
                               results["snp1"]["hr_lower_ci"], places=2)
        self.assertAlmostEqual(262.645390558627525,
                               results["snp1"]["hr_upper_ci"], places=0)
        self.assertAlmostEqual(3.3081873751763462, results["snp1"]["z_value"])
        self.assertAlmostEqual(-np.log10(0.0009390195967453607),
                               -np.log10(results["snp1"]["p_value"]))

        # TODO: Check the other predictors

    def test_coxph_snp1_inter(self):
        """Tests coxph regression with the first SNP (interaction)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # The interaction term
        inter = spec.interaction(spec.phenotypes.var1, spec.genotypes.snp1)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=dict(tte=spec.phenotypes.tte, event=spec.phenotypes.event),
            predictors=[spec.genotypes.snp1, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, inter],
            test="coxph",
        )

        # Performing the analysis and retrieving the results
        subscriber = subscribers.ResultsMemory()
        analysis.execute(
            self.phenotypes, self.genotypes, modelspec,
            subscribers=[subscriber],
        )
        results = subscriber.results[0]

        # Checking the marker information (it's a flip)
        self.assertEqual("snp1", results["snp1"]["name"])
        self.assertEqual("3", results["snp1"]["chrom"])
        self.assertEqual(1234, results["snp1"]["pos"])
        self.assertEqual("C", results["snp1"]["minor"])
        self.assertEqual("T", results["snp1"]["major"])
        self.assertAlmostEqual(0.48162903225806442, results["snp1"]["maf"])

        # Checking the results (according to R)
        self.assertAlmostEqual(-0.04756704444017357, results[inter.id]["coef"],
                               places=4)
        self.assertAlmostEqual(0.06434818489300186,
                               results[inter.id]["std_err"], places=5)
        self.assertAlmostEqual(0.9535465409954826, results[inter.id]["hr"],
                               places=4)
        self.assertAlmostEqual(0.8405598094456039,
                               results[inter.id]["hr_lower_ci"], places=4)
        self.assertAlmostEqual(1.081720771831991,
                               results[inter.id]["hr_upper_ci"], places=4)
        self.assertAlmostEqual(-0.7392134606324645,
                               results[inter.id]["z_value"], places=3)
        self.assertAlmostEqual(0.45977738842863736,
                               results[inter.id]["p_value"], places=3)

        # TODO: Check the other predictors

    @unittest.skip("Not implemented")
    def test_coxph_snp1_inter_category(self):
        """Tests coxph first SNP (interaction, category)."""
        # Preparing the data
        pheno = self.data.loc[:, ["tte", "event", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(columns={"snp1": "geno"})

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.coxph_inter_cat.create_matrices(self.dummy)

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

    def test_coxph_snp2(self):
        """Tests coxph regression with the second SNP."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=dict(tte=spec.phenotypes.tte, event=spec.phenotypes.event),
            predictors=[spec.genotypes.snp2, spec.phenotypes.age,
                        spec.phenotypes.var1, gender],
            test="coxph",
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
        self.assertAlmostEqual(0.40833333333333333, results["snp2"]["maf"])

        # Checking the results (according to R)
        self.assertAlmostEqual(1.13120383793364998, results["snp2"]["coef"])
        self.assertAlmostEqual(0.30473046375826845, results["snp2"]["std_err"])
        self.assertAlmostEqual(3.0993854129021017, results["snp2"]["hr"])
        self.assertAlmostEqual(1.7056446002934191,
                               results["snp2"]["hr_lower_ci"], places=3)
        self.assertAlmostEqual(5.631999735500465,
                               results["snp2"]["hr_upper_ci"], places=2)
        self.assertAlmostEqual(3.7121455596608572, results["snp2"]["z_value"])
        self.assertAlmostEqual(-np.log10(0.000205509736523446),
                               -np.log10(results["snp2"]["p_value"]))

        # TODO: Check the other predictors

    def test_coxph_snp2_inter(self):
        """Tests coxph regression with the second SNP (interaction)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # The interaction term
        inter = spec.interaction(spec.phenotypes.var1, spec.genotypes.snp2)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=dict(tte=spec.phenotypes.tte, event=spec.phenotypes.event),
            predictors=[spec.genotypes.snp2, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, inter],
            test="coxph",
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
        self.assertAlmostEqual(0.40833333333333333, results["snp2"]["maf"])

        # Checking the results (according to R)
        self.assertAlmostEqual(0.01742308104178164, results[inter.id]["coef"])
        self.assertAlmostEqual(0.02219048572389944,
                               results[inter.id]["std_err"])
        self.assertAlmostEqual(1.0175757482739625, results[inter.id]["hr"])
        self.assertAlmostEqual(0.9742674031703326,
                               results[inter.id]["hr_lower_ci"], places=4)
        self.assertAlmostEqual(1.062809245291237,
                               results[inter.id]["hr_upper_ci"], places=4)
        self.assertAlmostEqual(0.7851599671392845,
                               results[inter.id]["z_value"], places=5)
        self.assertAlmostEqual(0.4323597838912484,
                               results[inter.id]["p_value"], places=5)

        # TODO: Check the other predictors

    def test_coxph_snp3(self):
        """Tests coxph regression with the third SNP."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=dict(tte=spec.phenotypes.tte, event=spec.phenotypes.event),
            predictors=[spec.genotypes.snp3, spec.phenotypes.age,
                        spec.phenotypes.var1, gender],
            test="coxph",
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
        self.assertEqual(1519, results["snp3"]["pos"])
        self.assertEqual("G", results["snp3"]["minor"])
        self.assertEqual("T", results["snp3"]["major"])
        self.assertAlmostEqual(0.20833333333333334, results["snp3"]["maf"])

        # Checking the results (according to R)
        self.assertAlmostEqual(-0.0069461317578822582, results["snp3"]["coef"])
        self.assertAlmostEqual(0.39831755948730113, results["snp3"]["std_err"],
                               places=5)
        self.assertAlmostEqual(0.9930779368551547, results["snp3"]["hr"])
        self.assertAlmostEqual(0.4549197711311750,
                               results["snp3"]["hr_lower_ci"], places=3)
        self.assertAlmostEqual(2.167863107413991,
                               results["snp3"]["hr_upper_ci"], places=3)
        self.assertAlmostEqual(-0.01743867824160966,
                               results["snp3"]["z_value"])
        self.assertAlmostEqual(0.9860866530659741, results["snp3"]["p_value"])

        # TODO: Check the other predictors

    def test_coxph_snp3_inter(self):
        """Tests coxph regression with the third SNP (interaction)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # The interaction term
        inter = spec.interaction(spec.phenotypes.var1, spec.genotypes.snp3)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=dict(tte=spec.phenotypes.tte, event=spec.phenotypes.event),
            predictors=[spec.genotypes.snp3, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, inter],
            test="coxph",
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
        self.assertEqual(1519, results["snp3"]["pos"])
        self.assertEqual("G", results["snp3"]["minor"])
        self.assertEqual("T", results["snp3"]["major"])
        self.assertAlmostEqual(0.20833333333333334, results["snp3"]["maf"])

        # Checking the results (according to R)
        self.assertAlmostEqual(0.005873556606557682, results[inter.id]["coef"])
        self.assertAlmostEqual(0.03350856673847168,
                               results[inter.id]["std_err"])
        self.assertAlmostEqual(1.0058908397614570, results[inter.id]["hr"])
        self.assertAlmostEqual(0.94195099563823814,
                               results[inter.id]["hr_lower_ci"], places=4)
        self.assertAlmostEqual(1.074170934795214,
                               results[inter.id]["hr_upper_ci"], places=4)
        self.assertAlmostEqual(0.17528522339972738,
                               results[inter.id]["z_value"], places=6)
        self.assertAlmostEqual(0.8608555220369414,
                               results[inter.id]["p_value"], places=6)

        # TODO: Check the other predictors

    def test_coxph_snp4(self):
        """Tests coxph regression with the third SNP."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=dict(tte=spec.phenotypes.tte, event=spec.phenotypes.event),
            predictors=[spec.genotypes.snp4, spec.phenotypes.age,
                        spec.phenotypes.var1, gender],
            test="coxph",
        )

        # Performing the analysis and retrieving the results
        with self.assertRaises(StatsError):
            analysis.execute(
                self.phenotypes, self.genotypes, modelspec,
                subscribers=[subscribers.ResultsMemory()],
            )

    def test_coxph_snp4_inter(self):
        """Tests coxph regression with the third SNP (interaction)."""
        # The variables which are factors
        gender = spec.factor(spec.phenotypes.gender)

        # The interaction term
        inter = spec.interaction(spec.phenotypes.var1, spec.genotypes.snp4)

        # Creating the model specification
        modelspec = spec.ModelSpec(
            outcome=dict(tte=spec.phenotypes.tte, event=spec.phenotypes.event),
            predictors=[spec.genotypes.snp4, spec.phenotypes.age,
                        spec.phenotypes.var1, gender, inter],
            test="coxph",
        )

        # Performing the analysis and retrieving the results
        with self.assertRaises(StatsError):
            analysis.execute(
                self.phenotypes, self.genotypes, modelspec,
                subscribers=[subscribers.ResultsMemory()],
            )
