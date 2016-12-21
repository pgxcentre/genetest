

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
        subscriber = analysis.ResultsMemory()
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

        # TODO: Check SAS values instead of R
        # Checking the results (according to R)
        self.assertAlmostEqual(3.49824157323490, results["SNPs"]["coef"])
        self.assertAlmostEqual(1.05744964734605, results["SNPs"]["std_err"])
        self.assertAlmostEqual(33.05727202919358, results["SNPs"]["hr"])
        self.assertAlmostEqual(4.1606792782003,
                               results["SNPs"]["hr_lower_ci"], places=2)
        self.assertAlmostEqual(262.645390558628,
                               results["SNPs"]["hr_upper_ci"], places=0)
        self.assertAlmostEqual(3.30819, results["SNPs"]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(0.00093902),
                               -np.log10(results["SNPs"]["p_value"]), places=6)

        # Checking the second marker (snp2)
        results = gwas_results["snp2"]
        self.assertEqual("snp2", results["SNPs"]["name"])
        self.assertEqual("3", results["SNPs"]["chrom"])
        self.assertEqual(9618, results["SNPs"]["pos"])
        self.assertEqual("C", results["SNPs"]["minor"])
        self.assertEqual("A", results["SNPs"]["major"])

        # Checking the results (according to SAS)
        self.assertAlmostEqual(1.13120378638922, results["SNPs"]["coef"],
                               places=6)
        self.assertAlmostEqual(0.30473046186896, results["SNPs"]["std_err"])
        self.assertAlmostEqual(3.09938525314605, results["SNPs"]["hr"],
                               places=6)
        self.assertAlmostEqual(1.70564451869289,
                               results["SNPs"]["hr_lower_ci"], places=3)
        self.assertAlmostEqual(5.63199942434711,
                               results["SNPs"]["hr_upper_ci"], places=2)
        self.assertAlmostEqual(13.780023571179, results["SNPs"]["z_value"]**2,
                               places=5)
        self.assertAlmostEqual(-np.log10(0.0002055098552077),
                               -np.log10(results["SNPs"]["p_value"]), places=6)

        # Checking the third marker (snp3)
        results = gwas_results["snp3"]
        self.assertEqual("snp3", results["SNPs"]["name"])
        self.assertEqual("2", results["SNPs"]["chrom"])
        self.assertEqual(1519, results["SNPs"]["pos"])
        self.assertEqual("G", results["SNPs"]["minor"])
        self.assertEqual("T", results["SNPs"]["major"])

        # Checking the results (according to SAS)
        self.assertAlmostEqual(-0.0069430199975568, results["SNPs"]["coef"],
                               places=5)
        self.assertAlmostEqual(0.39831693319749, results["SNPs"]["std_err"])
        self.assertAlmostEqual(0.99308102708048, results["SNPs"]["hr"],
                               places=5)
        self.assertAlmostEqual(0.45492174515325,
                               results["SNPs"]["hr_lower_ci"], places=3)
        self.assertAlmostEqual(2.16786719222444,
                               results["SNPs"]["hr_upper_ci"], places=3)
        self.assertAlmostEqual(0.000303836044335,
                               results["SNPs"]["z_value"]**2, places=6)
        self.assertAlmostEqual(0.98609286353578, results["SNPs"]["p_value"],
                               places=4)

    @unittest.skip("Not implemented")
    def test_coxph_snp1_inter_full(self):
        """Tests coxph regression with the first SNP (full, interaction)."""
        # Preparing the data
        pheno = self.data.loc[:, ["tte", "event", "age", "var1", "gender"]]
        geno = self.data[["snp1"]].rename(columns={"snp1": "geno"})

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.coxph_inter.create_matrices(self.dummy)

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

    @unittest.skip("Not implemented")
    def test_coxph_snp1_inter_category_full(self):
        """Tests coxph first SNP (full, interaction, category)."""
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
        subscriber = analysis.ResultsMemory()
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

        # TODO: Check SAS values instead of R
        # Checking the results (according to R)
        self.assertAlmostEqual(3.49824157323490, results["snp1"]["coef"])
        self.assertAlmostEqual(1.05744964734605, results["snp1"]["std_err"])
        self.assertAlmostEqual(33.05727202919358, results["snp1"]["hr"])
        self.assertAlmostEqual(4.1606792782003,
                               results["snp1"]["hr_lower_ci"], places=2)
        self.assertAlmostEqual(262.645390558628,
                               results["snp1"]["hr_upper_ci"], places=0)
        self.assertAlmostEqual(3.30819, results["snp1"]["z_value"], places=5)
        self.assertAlmostEqual(-np.log10(0.00093902),
                               -np.log10(results["snp1"]["p_value"]), places=6)

    @unittest.skip("Not implemented")
    def test_coxph_snp1_inter(self):
        """Tests coxph regression with the first SNP (interaction)."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp1"]]
        data = data.rename(columns={"snp1": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.coxph_inter.create_matrices(self.dummy, create_dummy=False)

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
        subscriber = analysis.ResultsMemory()
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

        # Checking the results (according to SAS)
        self.assertAlmostEqual(1.13120378638922, results["snp2"]["coef"],
                               places=6)
        self.assertAlmostEqual(0.30473046186896, results["snp2"]["std_err"])
        self.assertAlmostEqual(3.09938525314605, results["snp2"]["hr"],
                               places=6)
        self.assertAlmostEqual(1.70564451869289,
                               results["snp2"]["hr_lower_ci"], places=3)
        self.assertAlmostEqual(5.63199942434711,
                               results["snp2"]["hr_upper_ci"], places=2)
        self.assertAlmostEqual(13.780023571179, results["snp2"]["z_value"]**2,
                               places=5)
        self.assertAlmostEqual(-np.log10(0.0002055098552077),
                               -np.log10(results["snp2"]["p_value"]), places=6)

    @unittest.skip("Not implemented")
    def test_coxph_snp2_inter(self):
        """Tests coxph regression with the second SNP (interaction)."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp2"]]
        data = data.rename(columns={"snp2": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.coxph_inter.create_matrices(self.dummy, create_dummy=False)

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
        subscriber = analysis.ResultsMemory()
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

        # Checking the results (according to SAS)
        self.assertAlmostEqual(-0.0069430199975568, results["snp3"]["coef"],
                               places=5)
        self.assertAlmostEqual(0.39831693319749, results["snp3"]["std_err"])
        self.assertAlmostEqual(0.99308102708048, results["snp3"]["hr"],
                               places=5)
        self.assertAlmostEqual(0.45492174515325,
                               results["snp3"]["hr_lower_ci"], places=3)
        self.assertAlmostEqual(2.16786719222444,
                               results["snp3"]["hr_upper_ci"], places=3)
        self.assertAlmostEqual(0.000303836044335,
                               results["snp3"]["z_value"]**2, places=6)
        self.assertAlmostEqual(0.98609286353578, results["snp3"]["p_value"],
                               places=4)

    @unittest.skip("Not implemented")
    def test_coxph_snp3_inter(self):
        """Tests coxph regression with the third SNP (interaction)."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp3"]]
        data = data.rename(columns={"snp3": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.coxph_inter.create_matrices(self.dummy, create_dummy=False)

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
                subscribers=[analysis.ResultsMemory()],
            )

    @unittest.skip("Not implemented")
    def test_coxph_snp4_inter(self):
        """Tests coxph regression with the third SNP (interaction)."""
        # Preparing the data
        data = self.data[["tte", "event", "age", "var1", "gender", "snp4"]]
        data = data.rename(columns={"snp4": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.coxph_inter.create_matrices(self.dummy, create_dummy=False)

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
