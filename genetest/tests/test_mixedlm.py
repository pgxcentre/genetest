

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import unittest
from collections import defaultdict

import numpy as np
import pandas as pd
from pkg_resources import resource_filename

from ..statistics.core import StatsError
from ..statistics.models.mixedlm import StatsMixedLM


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


@unittest.skip("Not implemented")
class TestStatsMixedLM(unittest.TestCase):
    """Tests the 'StatsMixedLM' class."""
    @classmethod
    def setUpClass(cls):
        cls.mixedlm = StatsMixedLM(
            outcome="pheno3",
            predictors=["age", "var1", "C(gender)", "geno", "C(visit)"],
            interaction=None,
            reml=True,
            p_threshold=1e-4,
        )
        cls.mixedlm_inter = StatsMixedLM(
            outcome="pheno3",
            predictors=["age", "var1", "C(gender)", "geno", "C(visit)"],
            interaction="var1",
            reml=True,
            p_threshold=1e-4,
        )
        cls.mixedlm_inter_cat = StatsMixedLM(
            outcome="pheno3",
            predictors=["age", "var1", "C(gender)", "geno", "C(visit)"],
            interaction="C(gender)",
            reml=True,
            p_threshold=1e-4,
        )

        cls.mixedlm_ml = StatsMixedLM(
            outcome="pheno3",
            predictors=["age", "var1", "C(gender)", "C(visit)", "geno"],
            interaction=None,
            reml=False,
            p_threshold=1e-4,
        )
        cls.mixedlm_ml_inter = StatsMixedLM(
            outcome="pheno3",
            predictors=["age", "var1", "C(gender)", "geno", "C(visit)"],
            interaction="var1",
            reml=False,
            p_threshold=1e-4,
        )
        cls.mixedlm_ml_inter_cat = StatsMixedLM(
            outcome="pheno3",
            predictors=["age", "var1", "C(gender)", "geno", "C(visit)"],
            interaction="C(gender)",
            reml=False,
            p_threshold=1e-4,
        )

    def setUp(self):
        # The data
        self.data = pd.read_csv(
            resource_filename(__name__, "data/statistics/mixedlm.txt.bz2"),
            sep="\t",
            compression="bz2",
        )

        # Recoding the samples
        sample_counter = defaultdict(int)
        sample_index = [s for s in self.data["sampleid"]]
        for i in range(len(sample_index)):
            sample = sample_index[i]
            sample_index[i] = "{}_{}".format(
                sample,
                sample_counter[sample],
            )
            sample_counter[sample] += 1

        # Saving the original values
        self.data["_ori_sample_names_"] = self.data["sampleid"]

        # Changing the sample column
        self.data["sampleid"] = sample_index

        # Setting the index
        self.data = self.data.set_index("sampleid", verify_integrity=True)

        # A dummy class for the 'get_phenotypes' function
        class DummyContainer(object):
            def set_phenotypes(self, data):
                self.data = data

            def get_phenotypes(self):
                return self.data

            def set_ori(self, ori_samples):
                self.ori_samples = ori_samples

            def get_original_sample_names(self):
                return self.ori_samples
        self.dummy = DummyContainer()
        self.dummy.set_ori(self.data.loc[:, ["_ori_sample_names_"]])

    def test_mixedlm_snp1_reml_full(self):
        """Tests mixedlm regression with the first SNP (using REML, full)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno3", "age", "var1", "gender", "visit"]]
        geno = self.data[["snp1", "_ori_sample_names_"]]
        geno = geno.rename(columns={"snp1": "geno"})
        geno = geno.drop_duplicates().set_index("_ori_sample_names_")

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.mixedlm.create_matrices(self.dummy)

        # Merging with genotype
        y, X = self.mixedlm.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.mixedlm.fit(y, X)

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
        self.assertAlmostEqual(
            -np.log10(0.000023996236375808),
            -np.log10(self.mixedlm.results.ts_p_value),
        )

    def test_mixedlm_snp1_inter_reml_full(self):
        """Tests mixedlm with first SNP (using REML, interaction, full)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno3", "age", "var1", "gender", "visit"]]
        geno = self.data[["snp1", "_ori_sample_names_"]]
        geno = geno.rename(columns={"snp1": "geno"})
        geno = geno.drop_duplicates().set_index("_ori_sample_names_")

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.mixedlm_inter.create_matrices(self.dummy)

        # Merging with genotype
        y, X = self.mixedlm_inter.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.mixedlm_inter.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            1.357502192968099, self.mixedlm_inter.results.coef,
        )
        self.assertAlmostEqual(
            0.6710947924603057, self.mixedlm_inter.results.std_err, places=5,
        )
        self.assertAlmostEqual(
            0.04218056953351801, self.mixedlm_inter.results.lower_ci, places=5,
        )
        self.assertAlmostEqual(
            2.6728238164026799, self.mixedlm_inter.results.upper_ci, places=5,
        )
        self.assertAlmostEqual(
            2.0228173548946042, self.mixedlm_inter.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            -np.log10(4.309198170818540e-02),
            -np.log10(self.mixedlm_inter.results.p_value), places=5,
        )
        self.assertTrue(np.isnan(self.mixedlm_inter.results.ts_p_value))

    def test_mixedlm_snp1_inter_category_reml_full(self):
        """Tests mixedlm with first SNP (REML, interaction, category, full)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno3", "age", "var1", "gender", "visit"]]
        geno = self.data[["snp1", "_ori_sample_names_"]]
        geno = geno.rename(columns={"snp1": "geno"})
        geno = geno.drop_duplicates().set_index("_ori_sample_names_")

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.mixedlm_inter_cat.create_matrices(self.dummy)

        # Merging with genotype
        y, X = self.mixedlm_inter_cat.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.mixedlm_inter_cat.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            -7.93574947926475005, self.mixedlm_inter_cat.results.coef,
        )
        self.assertAlmostEqual(
            16.8061675796400429, self.mixedlm_inter_cat.results.std_err,
            places=4,
        )
        self.assertAlmostEqual(
            -40.8752326535039145, self.mixedlm_inter_cat.results.lower_ci,
            places=3,
        )
        self.assertAlmostEqual(
            25.0037336949744144, self.mixedlm_inter_cat.results.upper_ci,
            places=3,
        )
        self.assertAlmostEqual(
            -0.4721926900740043, self.mixedlm_inter_cat.results.z_value,
            places=5,
        )
        self.assertAlmostEqual(
            6.367892569176092e-01, self.mixedlm_inter_cat.results.p_value,
            places=5,
        )
        self.assertTrue(np.isnan(self.mixedlm_inter_cat.results.ts_p_value))

    def test_mixedlm_snp1_ml_full(self):
        """Tests mixedlm regression with the first SNP (using ML, full)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno3", "age", "var1", "gender", "visit"]]
        geno = self.data[["snp1", "_ori_sample_names_"]]
        geno = geno.rename(columns={"snp1": "geno"})
        geno = geno.drop_duplicates().set_index("_ori_sample_names_")

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.mixedlm_ml.create_matrices(self.dummy)

        # Merging with genotypes
        y, X = self.mixedlm_ml.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.mixedlm_ml.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            37.45704065571743513, self.mixedlm_ml.results.coef,
        )
        self.assertAlmostEqual(
            7.986654823996228, self.mixedlm_ml.results.std_err, places=5,
        )
        self.assertAlmostEqual(
            21.8034848437317450, self.mixedlm_ml.results.lower_ci, places=4,
        )
        self.assertAlmostEqual(
            53.1105964677031253, self.mixedlm_ml.results.upper_ci, places=4,
        )
        self.assertAlmostEqual(
            4.6899536140182541, self.mixedlm_ml.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            -np.log10(2.732669904137452e-06),
            -np.log10(self.mixedlm_ml.results.p_value), places=5,
        )
        self.assertAlmostEqual(
            -np.log10(2.399623637577515e-05),
            -np.log10(self.mixedlm_ml.results.ts_p_value),
        )

    def test_mixedlm_snp1_inter_ml_full(self):
        """Tests mixedlm with first SNP (using ML, interaction, full)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno3", "age", "var1", "gender", "visit"]]
        geno = self.data[["snp1", "_ori_sample_names_"]]
        geno = geno.rename(columns={"snp1": "geno"})
        geno = geno.drop_duplicates().set_index("_ori_sample_names_")

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.mixedlm_ml_inter.create_matrices(self.dummy)

        # Merging with genotype
        y, X = self.mixedlm_ml_inter.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.mixedlm_ml_inter.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            1.3575021929662827, self.mixedlm_ml_inter.results.coef,
        )
        self.assertAlmostEqual(
            0.6366563415656629, self.mixedlm_ml_inter.results.std_err,
            places=6,
        )
        self.assertAlmostEqual(
            0.1096786929685527, self.mixedlm_ml_inter.results.lower_ci,
            places=6,
        )
        self.assertAlmostEqual(
            2.6053256929640130, self.mixedlm_ml_inter.results.upper_ci,
            places=6,
        )
        self.assertAlmostEqual(
            2.1322369767462277, self.mixedlm_ml_inter.results.z_value,
            places=6,
        )
        self.assertAlmostEqual(
            -np.log10(3.298737010780739e-02),
            -np.log10(self.mixedlm_ml_inter.results.p_value), places=5,
        )
        self.assertTrue(np.isnan(self.mixedlm_ml_inter.results.ts_p_value))

    def test_mixedlm_snp1_inter_category_ml_full(self):
        """Tests mixedlm with first SNP (ML, interaction, category, full)."""
        # Preparing the data
        pheno = self.data.loc[:, ["pheno3", "age", "var1", "gender", "visit"]]
        geno = self.data[["snp1", "_ori_sample_names_"]]
        geno = geno.rename(columns={"snp1": "geno"})
        geno = geno.drop_duplicates().set_index("_ori_sample_names_")

        # Permuting the genotypes
        geno = geno.iloc[np.random.permutation(geno.shape[0]), :]

        # Adding the data to the object
        self.dummy.set_phenotypes(pheno)

        # Preparing the matrices
        y, X = self.mixedlm_ml_inter_cat.create_matrices(self.dummy)

        # Merging with genotype
        y, X = self.mixedlm_ml_inter_cat.merge_matrices_genotypes(y, X, geno)

        # Fitting
        self.mixedlm_ml_inter_cat.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            -7.9357494792821504, self.mixedlm_ml_inter_cat.results.coef,
        )
        self.assertAlmostEqual(
            15.9437295377973829, self.mixedlm_ml_inter_cat.results.std_err,
            places=4,
        )
        self.assertAlmostEqual(
            -39.1848851526124520, self.mixedlm_ml_inter_cat.results.lower_ci,
            places=4,
        )
        self.assertAlmostEqual(
            23.3133861940481530, self.mixedlm_ml_inter_cat.results.upper_ci,
            places=4,
        )
        self.assertAlmostEqual(
            -0.4977348280067770, self.mixedlm_ml_inter_cat.results.z_value,
            places=6,
        )
        self.assertAlmostEqual(
            6.186709566882038e-01, self.mixedlm_ml_inter_cat.results.p_value,
            places=6,
        )
        self.assertTrue(np.isnan(self.mixedlm_ml_inter_cat.results.ts_p_value))

    def test_mixedlm_snp1_reml(self):
        """Tests mixedlm regression with the first SNP (using REML)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp1", "visit"]]
        data = data.rename(columns={"snp1": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.mixedlm.fit(y, X)

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
        self.assertAlmostEqual(
            -np.log10(0.000023996236375808),
            -np.log10(self.mixedlm.results.ts_p_value),
        )

    def test_mixedlm_snp1_ml(self):
        """Tests mixedlm regression with the first SNP (using ML)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp1", "visit"]]
        data = data.rename(columns={"snp1": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_ml.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.mixedlm_ml.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            37.45704065571743513, self.mixedlm_ml.results.coef,
        )
        self.assertAlmostEqual(
            7.986654823996228, self.mixedlm_ml.results.std_err, places=5,
        )
        self.assertAlmostEqual(
            21.8034848437317450, self.mixedlm_ml.results.lower_ci, places=4,
        )
        self.assertAlmostEqual(
            53.1105964677031253, self.mixedlm_ml.results.upper_ci, places=4,
        )
        self.assertAlmostEqual(
            4.6899536140182541, self.mixedlm_ml.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            -np.log10(2.732669904137452e-06),
            -np.log10(self.mixedlm_ml.results.p_value), places=5,
        )
        self.assertAlmostEqual(
            -np.log10(2.399623637577515e-05),
            -np.log10(self.mixedlm_ml.results.ts_p_value),
        )

    def test_mixedlm_snp1_inter_reml(self):
        """Tests mixedlm regression with the first SNP (using REML, inter)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp1", "visit"]]
        data = data.rename(columns={"snp1": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_inter.create_matrices(
            self.dummy, create_dummy=False,
        )

        # Fitting
        self.mixedlm_inter.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            1.357502192968099, self.mixedlm_inter.results.coef,
        )
        self.assertAlmostEqual(
            0.6710947924603057, self.mixedlm_inter.results.std_err, places=5,
        )
        self.assertAlmostEqual(
            0.04218056953351801, self.mixedlm_inter.results.lower_ci, places=5,
        )
        self.assertAlmostEqual(
            2.6728238164026799, self.mixedlm_inter.results.upper_ci, places=5,
        )
        self.assertAlmostEqual(
            2.0228173548946042, self.mixedlm_inter.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            -np.log10(4.309198170818540e-02),
            -np.log10(self.mixedlm_inter.results.p_value), places=5,
        )
        self.assertTrue(np.isnan(self.mixedlm_inter.results.ts_p_value))

    def test_mixedlm_snp1_inter_ml(self):
        """Tests mixedlm regression with the first SNP (using ML, inter)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp1", "visit"]]
        data = data.rename(columns={"snp1": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_ml_inter.create_matrices(
            self.dummy, create_dummy=False,
        )

        # Fitting
        self.mixedlm_ml_inter.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            1.3575021929662827, self.mixedlm_ml_inter.results.coef,
        )
        self.assertAlmostEqual(
            0.6366563415656629, self.mixedlm_ml_inter.results.std_err,
            places=6,
        )
        self.assertAlmostEqual(
            0.1096786929685527, self.mixedlm_ml_inter.results.lower_ci,
            places=6,
        )
        self.assertAlmostEqual(
            2.6053256929640130, self.mixedlm_ml_inter.results.upper_ci,
            places=6,
        )
        self.assertAlmostEqual(
            2.1322369767462277, self.mixedlm_ml_inter.results.z_value,
            places=6,
        )
        self.assertAlmostEqual(
            -np.log10(3.298737010780739e-02),
            -np.log10(self.mixedlm_ml_inter.results.p_value), places=5,
        )
        self.assertTrue(np.isnan(self.mixedlm_ml_inter.results.ts_p_value))

    def test_mixedlm_snp2_reml(self):
        """Tests mixedlm regression with the second SNP (using REML)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp2", "visit"]]
        data = data.rename(columns={"snp2": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.mixedlm.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertTrue(np.isnan(self.mixedlm.results.coef))
        self.assertTrue(np.isnan(self.mixedlm.results.std_err))
        self.assertTrue(np.isnan(self.mixedlm.results.lower_ci))
        self.assertTrue(np.isnan(self.mixedlm.results.upper_ci))
        self.assertTrue(np.isnan(self.mixedlm.results.z_value))
        self.assertTrue(np.isnan(self.mixedlm.results.p_value))
        self.assertAlmostEqual(
            -np.log10(0.0003525413773446878),
            -np.log10(self.mixedlm.results.ts_p_value),
        )

    def test_mixedlm_snp2_ml(self):
        """Tests mixedlm regression with the second SNP (using ML)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp2", "visit"]]
        data = data.rename(columns={"snp2": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_ml.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.mixedlm_ml.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertTrue(np.isnan(self.mixedlm_ml.results.coef))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.std_err))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.lower_ci))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.upper_ci))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.z_value))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.p_value))
        self.assertAlmostEqual(
            -np.log10(0.000352541377346777),
            -np.log10(self.mixedlm_ml.results.ts_p_value),
        )

    def test_mixedlm_snp2_inter_reml(self):
        """Tests mixedlm regression with the second SNP (using REML, inter)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp2", "visit"]]
        data = data.rename(columns={"snp2": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_inter.create_matrices(
            self.dummy, create_dummy=False,
        )

        # Fitting
        self.mixedlm_inter.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            0.8952450482655012, self.mixedlm_inter.results.coef,
        )
        self.assertAlmostEqual(
            0.5976670951727925, self.mixedlm_inter.results.std_err, places=5,
        )
        self.assertAlmostEqual(
            -0.2761609330178445, self.mixedlm_inter.results.lower_ci, places=5,
        )
        self.assertAlmostEqual(
            2.0666510295488472, self.mixedlm_inter.results.upper_ci, places=5,
        )
        self.assertAlmostEqual(
            1.497899174132508504, self.mixedlm_inter.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            1.341594483012889e-01, self.mixedlm_inter.results.p_value,
            places=5,
        )
        self.assertTrue(np.isnan(self.mixedlm_inter.results.ts_p_value))

    def test_mixedlm_snp2_inter_ml(self):
        """Tests mixedlm regression with the second SNP (using ML, inter)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp2", "visit"]]
        data = data.rename(columns={"snp2": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_ml_inter.create_matrices(
            self.dummy, create_dummy=False,
        )

        # Fitting
        self.mixedlm_ml_inter.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            0.8952450482657273, self.mixedlm_ml_inter.results.coef,
        )
        self.assertAlmostEqual(
            0.5669968515485548, self.mixedlm_ml_inter.results.std_err,
            places=6,
        )
        self.assertAlmostEqual(
            -0.2160483601170435, self.mixedlm_ml_inter.results.lower_ci,
            places=5,
        )
        self.assertAlmostEqual(
            2.006538456648498, self.mixedlm_ml_inter.results.upper_ci,
            places=5,
        )
        self.assertAlmostEqual(
            1.578924196528916468, self.mixedlm_ml_inter.results.z_value,
            places=5,
        )
        self.assertAlmostEqual(
            1.143534452552892e-01, self.mixedlm_ml_inter.results.p_value,
            places=6,
        )
        self.assertTrue(np.isnan(self.mixedlm_ml_inter.results.ts_p_value))

    def test_mixedlm_snp3_reml(self):
        """Tests mixedlm regression with the third SNP (using REML)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp3", "visit"]]
        data = data.rename(columns={"snp3": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.mixedlm.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertTrue(np.isnan(self.mixedlm.results.coef))
        self.assertTrue(np.isnan(self.mixedlm.results.std_err))
        self.assertTrue(np.isnan(self.mixedlm.results.lower_ci))
        self.assertTrue(np.isnan(self.mixedlm.results.upper_ci))
        self.assertTrue(np.isnan(self.mixedlm.results.z_value))
        self.assertAlmostEqual(
            -np.log10(0.001256884400795483),
            -np.log10(self.mixedlm.results.ts_p_value),
        )

    def test_mixedlm_snp3_ml(self):
        """Tests mixedlm regression with the third SNP (using ML)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp3", "visit"]]
        data = data.rename(columns={"snp3": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_ml.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.mixedlm_ml.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertTrue(np.isnan(self.mixedlm_ml.results.coef))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.std_err))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.lower_ci))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.upper_ci))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.z_value))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.p_value))
        self.assertAlmostEqual(
            -np.log10(0.001256884400782397),
            -np.log10(self.mixedlm_ml.results.ts_p_value),
        )

    def test_mixedlm_snp3_inter_reml(self):
        """Tests mixedlm regression with the third SNP (using REML, inter)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp3", "visit"]]
        data = data.rename(columns={"snp3": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_inter.create_matrices(
            self.dummy, create_dummy=False,
        )

        # Fitting
        self.mixedlm_inter.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            0.9199422369515684, self.mixedlm_inter.results.coef,
        )
        self.assertAlmostEqual(
            0.4498593164720226, self.mixedlm_inter.results.std_err, places=5,
        )
        self.assertAlmostEqual(
            0.03823417855659805, self.mixedlm_inter.results.lower_ci, places=5,
        )
        self.assertAlmostEqual(
            1.8016502953465388, self.mixedlm_inter.results.upper_ci, places=5,
        )
        self.assertAlmostEqual(
            2.0449553966473895, self.mixedlm_inter.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            -np.log10(4.085925566922222e-02),
            -np.log10(self.mixedlm_inter.results.p_value), places=4,
        )
        self.assertTrue(np.isnan(self.mixedlm_inter.results.ts_p_value))

    def test_mixedlm_snp3_inter_ml(self):
        """Tests mixedlm regression with the third SNP (using ML, inter)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp3", "visit"]]
        data = data.rename(columns={"snp3": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_ml_inter.create_matrices(
            self.dummy, create_dummy=False,
        )

        # Fitting
        self.mixedlm_ml_inter.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            0.9199422369518273, self.mixedlm_ml_inter.results.coef,
        )
        self.assertAlmostEqual(
            0.4267740150554054, self.mixedlm_ml_inter.results.std_err,
            places=6,
        )
        self.assertAlmostEqual(
            0.08348053790567811, self.mixedlm_ml_inter.results.lower_ci,
            places=5,
        )
        self.assertAlmostEqual(
            1.7564039359979766, self.mixedlm_ml_inter.results.upper_ci,
            places=5,
        )
        self.assertAlmostEqual(
            2.1555722806422435, self.mixedlm_ml_inter.results.z_value,
            places=5,
        )
        self.assertAlmostEqual(
            -np.log10(3.111707895646454e-02),
            -np.log10(self.mixedlm_ml_inter.results.p_value), places=5,
        )
        self.assertTrue(np.isnan(self.mixedlm_ml_inter.results.ts_p_value))

    def test_mixedlm_snp4_reml(self):
        """Tests mixedlm regression with the fourth SNP (using REML)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp4", "visit"]]
        data = data.rename(columns={"snp4": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.mixedlm.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertTrue(np.isnan(self.mixedlm.results.coef))
        self.assertTrue(np.isnan(self.mixedlm.results.std_err))
        self.assertTrue(np.isnan(self.mixedlm.results.lower_ci))
        self.assertTrue(np.isnan(self.mixedlm.results.upper_ci))
        self.assertTrue(np.isnan(self.mixedlm.results.z_value))
        self.assertTrue(np.isnan(self.mixedlm.results.p_value))
        self.assertAlmostEqual(
            0.8488372306662677, self.mixedlm.results.ts_p_value,
        )

    def test_mixedlm_snp4_ml(self):
        """Tests mixedlm regression with the fourth SNP (using ML)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp4", "visit"]]
        data = data.rename(columns={"snp4": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_ml.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        self.mixedlm_ml.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertTrue(np.isnan(self.mixedlm_ml.results.coef))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.std_err))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.lower_ci))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.upper_ci))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.z_value))
        self.assertTrue(np.isnan(self.mixedlm_ml.results.p_value))
        self.assertAlmostEqual(
            0.8488372306664566, self.mixedlm_ml.results.ts_p_value,
        )

    def test_mixedlm_snp4_inter_reml(self):
        """Tests mixedlm regression with the fourth SNP (using REML, inter)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp4", "visit"]]
        data = data.rename(columns={"snp4": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_inter.create_matrices(
            self.dummy, create_dummy=False,
        )

        # Fitting
        self.mixedlm_inter.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            0.05832811192587545, self.mixedlm_inter.results.coef,
        )
        self.assertAlmostEqual(
            0.7381058671562147, self.mixedlm_inter.results.std_err, places=4,
        )
        self.assertAlmostEqual(
            -1.388332804478011, self.mixedlm_inter.results.lower_ci, places=4,
        )
        self.assertAlmostEqual(
            1.504989028329762, self.mixedlm_inter.results.upper_ci, places=4,
        )
        self.assertAlmostEqual(
            0.07902404590089884, self.mixedlm_inter.results.z_value, places=5,
        )
        self.assertAlmostEqual(
            9.370134970059800e-01, self.mixedlm_inter.results.p_value,
            places=5,
        )
        self.assertTrue(np.isnan(self.mixedlm_inter.results.ts_p_value))

    def test_mixedlm_snp4_inter_ml(self):
        """Tests mixedlm regression with the fourth SNP (using ML, inter)."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp4", "visit"]]
        data = data.rename(columns={"snp4": "geno"})

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm_ml_inter.create_matrices(
            self.dummy, create_dummy=False,
        )

        # Fitting
        self.mixedlm_ml_inter.fit(y, X)

        # Checking the results (according to R lme4)
        self.assertAlmostEqual(
            0.05832811192492895, self.mixedlm_ml_inter.results.coef,
        )
        self.assertAlmostEqual(
            0.7002288518048638, self.mixedlm_ml_inter.results.std_err,
            places=5,
        )
        self.assertAlmostEqual(
            -1.314095218548438, self.mixedlm_ml_inter.results.lower_ci,
            places=4,
        )
        self.assertAlmostEqual(
            1.430751442398297, self.mixedlm_ml_inter.results.upper_ci,
            places=4,
        )
        self.assertAlmostEqual(
            0.08329864125790624, self.mixedlm_ml_inter.results.z_value,
            places=6,
        )
        self.assertAlmostEqual(
            9.336140806606035e-01, self.mixedlm_ml_inter.results.p_value,
            places=6,
        )
        self.assertTrue(np.isnan(self.mixedlm_ml_inter.results.ts_p_value))

    def test_mixedlm_monomorphic(self):
        """Tests mixedlm regression on monomorphic SNP."""
        # Preparing the data
        data = self.data[["pheno3", "age", "var1", "gender", "snp4", "visit"]]
        data = data.rename(columns={"snp4": "geno"})
        data.geno = 0

        # Adding the data to the object
        self.dummy.set_phenotypes(data)

        # Preparing the matrices
        y, X = self.mixedlm.create_matrices(self.dummy, create_dummy=False)

        # Fitting
        with self.assertRaises(StatsError):
            self.mixedlm.fit(y, X)

        # Checking the results
        self.assertTrue(np.isnan(self.mixedlm.results.coef))
        self.assertTrue(np.isnan(self.mixedlm.results.std_err))
        self.assertTrue(np.isnan(self.mixedlm.results.lower_ci))
        self.assertTrue(np.isnan(self.mixedlm.results.upper_ci))
        self.assertTrue(np.isnan(self.mixedlm.results.z_value))
        self.assertTrue(np.isnan(self.mixedlm.results.p_value))
        self.assertTrue(np.isnan(self.mixedlm.results.ts_p_value))

    def test_merge_matrices_genotypes(self):
        """Tests the 'merge_matrices_genotypes' function for MixedLM."""
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
        genotypes = pd.DataFrame(
            [("s0", 0.0), ("s1", 1.0), ("s2", 0.0), ("s3", np.nan),
             ("s4", 0.0), ("s5", 1.0), ("s6", 2.0), ("s7", np.nan),
             ("s8", 0.0), ("s9", 0.0)],
            columns=["sample_id", "geno"],
        ).set_index("sample_id")
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

        # Setting the original sample names
        samples = samples.iloc[np.random.permutation(len(samples))]
        self.mixedlm._ori_samples = samples

        # Getting the observed data
        new_y, new_X = self.mixedlm.merge_matrices_genotypes(
            y=y, X=X,
            genotypes=genotypes.iloc[np.random.permutation(len(genotypes))],
        )

        # The expected data
        expected_y = pd.DataFrame(
            [("s0_1", 3.0), ("s0_2", 3.4), ("s0_3", 3.4), ("s1_1", 3.4),
             ("s1_2", 3.4), ("s1_3", 3.4), ("s2_1", 5.3), ("s2_2", 5.3),
             ("s2_3", 5.3), ("s4_1", 0.5), ("s4_2", 0.5), ("s4_3", 0.5),
             ("s5_1", 2.4), ("s5_2", 2.4), ("s5_3", 2.4), ("s6_1", 5.6),
             ("s6_2", 5.6), ("s6_3", 5.6), ("s8_1", 0.3), ("s8_2", 0.3),
             ("s8_3", 0.3), ("s9_1", 1.9), ("s9_2", 1.9), ("s9_3", 1.9)],
            columns=["sample_id", "pheno"],
        ).set_index("sample_id")
        expected_X = pd.DataFrame(
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

        # Checking the results
        self.assertTrue(np.array_equal(new_y.index.values, new_X.index.values))
        self.assertTrue(expected_y.equals(new_y.sortlevel()))
        self.assertTrue(expected_X.equals(new_X.sortlevel()))
        self.assertTrue(np.array_equal(
            np.array([s[:2] for s in new_X.index.values]),
            self.mixedlm._groups,
        ))
