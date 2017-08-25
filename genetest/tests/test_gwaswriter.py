

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

from pyplink import PyPlink
from geneparse import parsers

from .. import modelspec as spec
from ..subscribers import GWASWriter
from ..analysis import execute_formula
from ..phenotypes.dummy import _DummyPhenotypes


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestGWASWriter(unittest.TestCase):
    """Tests the 'GWASWriter' class."""
    @classmethod
    def setUpClass(cls):
        # Loading the data
        data = pd.read_csv(
            resource_filename(__name__, "data/statistics/factors.txt.bz2"),
            sep="\t",
            compression="bz2",
        ).set_index("sample_id")

        # Creating the dummy phenotype container
        cls.phenotypes = _DummyPhenotypes()
        cls.phenotypes.data = data.drop(
            [col for col in data.columns if col.startswith("snp")],
            axis=1,
        )

        # Creating a temporary directory
        cls.tmp_dir = TemporaryDirectory(prefix="genetest_test_linear_")

        # The plink file prefix
        cls.plink_prefix = os.path.join(cls.tmp_dir.name, "input")

        # Permuting the sample to add a bit of randomness
        new_sample_order = np.random.permutation(data.index)

        # Creating the BED file
        with PyPlink(cls.plink_prefix, "w") as bed:
            for i in range(3):
                snp = "snp{}".format(i + 1)
                bed.write_genotypes(data.loc[new_sample_order, snp])

        # Creating the BIM file
        with open(cls.plink_prefix + ".bim", "w") as bim:
            print(1, "snp1", 0, 1, "B", "A", sep="\t", file=bim)
            print(1, "snp2", 0, 2, "B", "A", sep="\t", file=bim)
            print(1, "snp3", 0, 3, "B", "A", sep="\t", file=bim)

        # Creating the FAM file
        with open(cls.plink_prefix + ".fam", "w") as fam:
            for sample in new_sample_order:
                print(sample, sample, 0, 0, 0, -9, file=fam)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def setUp(self):
        # Resetting the model specification
        spec._reset()

        # Creating the genotype parser
        self.genotypes = parsers["plink"](self.plink_prefix)

        # Reordering the columns and the rows of the phenotype data frame
        self.phenotypes.data = self.phenotypes.data.iloc[
            np.random.permutation(self.phenotypes.data.shape[0]),
            np.random.permutation(self.phenotypes.data.shape[1])
        ]

        # Creating a GWAS writer
        self.output_name = os.path.join(self.tmp_dir.name, "output.txt")
        self.gwas_writer = GWASWriter(self.output_name, "linear")

    def tearDown(self):
        self.genotypes.close()
        self.gwas_writer.close()

    def test_normal_gwas(self):
        """Tests a normal GWAS analysis."""
        # Executing the formula
        execute_formula(
            phenotypes=self.phenotypes,
            genotypes=self.genotypes,
            formula="pheno ~ SNPs + var1 + var2 + factor(var3) + "
                    "factor(var4) + factor(var5)",
            test="linear",
            subscribers=[self.gwas_writer],
            output_prefix="",
        )

        # Checking the file exists
        self.assertTrue(os.path.isfile(self.output_name))

        # Reading the data
        df = pd.read_csv(self.output_name, sep="\t")
        df = df.set_index("snp", drop=False).loc[["snp1", "snp2", "snp3"], :]

        # Checking all the columns are there
        self.assertEqual(
            set(df.columns),
            {"snp", "chr", "pos", "major", "minor", "maf", "n", "ll", "coef",
             "se", "lower", "upper", "t", "p", "adj_r2"},
            "Missing columns in the GWAS Writer",
        )

        # Checking the coefficients (this should be enough to make sure we get
        # the right columns)
        np.testing.assert_array_almost_equal(
            df.coef.values,
            np.array([6.48616317675085, 4.29701960321627, -1.44891512389655]),
        )

    def test_categorical_gwas(self):
        """Tests a categorical GWAS analysis."""
        # Executing the formula
        execute_formula(
            phenotypes=self.phenotypes,
            genotypes=self.genotypes,
            formula="pheno | var4 ~ SNPs + var1 + var2 + factor(var3) + "
                    "factor(var5)",
            test="linear",
            subscribers=[self.gwas_writer],
            output_prefix="",
        )

        # Checking the file exists
        self.assertTrue(os.path.isfile(self.output_name))

        # Reading the data
        df = pd.read_csv(self.output_name, sep="\t")
        df = df.set_index("snp", drop=False).loc[["snp1", "snp2", "snp3"], :]

        # Checking all the columns are there
        self.assertEqual(
            set(df.columns),
            {"snp", "chr", "pos", "major", "minor", "maf", "n", "ll", "coef",
             "se", "lower", "upper", "t", "p", "adj_r2", "subgroup"},
            "Missing columns in the GWAS Writer",
        )

        # Checking the coefficients for the fist subgroup analysis (this should
        # be enough to make sure we get the right columns)
        subgroup = df.subgroup == "var4:y0"
        np.testing.assert_array_almost_equal(
            df.loc[subgroup, "coef"].values,
            np.array([-0.6798617801022, 5.9538885434287, -15.1758958532932]),
        )

        # Checking the coefficients for the second subgroup analysis
        subgroup = df.subgroup == "var4:y1"
        np.testing.assert_array_almost_equal(
            df.loc[subgroup, "coef"].values,
            np.array([12.69000471889982, 4.2496912087917, 7.5291802846083]),
        )

    def test_simple_interaction_gwas(self):
        """Tests a simple interaction GWAS analysis."""
        # Executing the formula
        execute_formula(
            phenotypes=self.phenotypes,
            genotypes=self.genotypes,
            formula="pheno ~ SNPs + var1 + var2 + factor(var3) + "
                    "factor(var4) + factor(var5) + SNPs*var1",
            test="linear",
            subscribers=[self.gwas_writer],
            output_prefix="",
        )

        # Checking the file exists
        self.assertTrue(os.path.isfile(self.output_name))

        # Reading the data
        df = pd.read_csv(self.output_name, sep="\t")
        df = df.set_index("snp", drop=False).loc[["snp1", "snp2", "snp3"], :]

        # Checking all the columns are there
        self.assertEqual(
            set(df.columns),
            {"snp", "chr", "pos", "major", "minor", "maf", "n", "ll",
             "TRANSFORM:GWAS_INTER:coef", "TRANSFORM:GWAS_INTER:se",
             "TRANSFORM:GWAS_INTER:lower", "TRANSFORM:GWAS_INTER:upper",
             "TRANSFORM:GWAS_INTER:t", "TRANSFORM:GWAS_INTER:p", "adj_r2"},
            "Missing columns in the GWAS Writer",
        )

        # Checking the coefficients (this should be enough to make sure we get
        # the right columns)
        np.testing.assert_array_almost_equal(
            df["TRANSFORM:GWAS_INTER:coef"].values,
            np.array([0.4073975765317, -0.04253919820585, -0.05814970176832]),
        )

    def test_complex_interaction_gwas(self):
        """Tests a complex interaction GWAS analysis."""
        # Executing the formula
        execute_formula(
            phenotypes=self.phenotypes,
            genotypes=self.genotypes,
            formula="pheno ~ SNPs + var1 + var2 + factor(var3) + "
                    "factor(var4) + factor(var5) + SNPs*factor(var3)",
            test="linear",
            subscribers=[self.gwas_writer],
            output_prefix=self.output_name,
        )

        # Checking the file exists
        self.assertTrue(os.path.isfile(self.output_name))

        # Reading the data
        df = pd.read_csv(self.output_name, sep="\t")
        df = df.set_index("snp", drop=False).loc[["snp1", "snp2", "snp3"], :]

        # Checking all the columns are there
        self.assertEqual(
            set(df.columns),
            {"snp", "chr", "pos", "major", "minor", "maf", "n", "ll",
             "TRANSFORM:GWAS_INTER:level.x1:coef",
             "TRANSFORM:GWAS_INTER:level.x1:se",
             "TRANSFORM:GWAS_INTER:level.x1:lower",
             "TRANSFORM:GWAS_INTER:level.x1:upper",
             "TRANSFORM:GWAS_INTER:level.x1:t",
             "TRANSFORM:GWAS_INTER:level.x1:p",
             "TRANSFORM:GWAS_INTER:level.x2:coef",
             "TRANSFORM:GWAS_INTER:level.x2:se",
             "TRANSFORM:GWAS_INTER:level.x2:lower",
             "TRANSFORM:GWAS_INTER:level.x2:upper",
             "TRANSFORM:GWAS_INTER:level.x2:t",
             "TRANSFORM:GWAS_INTER:level.x2:p", "adj_r2"},
            "Missing columns in the GWAS Writer",
        )

        # Checking the coefficients (this should be enough to make sure we get
        # the right columns)
        #   var3=x1
        np.testing.assert_array_almost_equal(
            df["TRANSFORM:GWAS_INTER:level.x1:coef"].values,
            np.array([6.15574045825434, 2.2178331615269, -1.94057337856961]),
        )

        #   var3=x2
        np.testing.assert_array_almost_equal(
            df["TRANSFORM:GWAS_INTER:level.x2:coef"].values,
            np.array([21.47033225862596, 2.7136560668229, -0.05622018358578]),
        )

    def test_complex_interaction_categorical_gwas(self):
        """Tests a complex interaction GWAS analysis."""
        # Executing the formula
        execute_formula(
            phenotypes=self.phenotypes,
            genotypes=self.genotypes,
            formula="pheno | var4 ~ SNPs + var1 + var2 + factor(var3) + "
                    "factor(var5) + SNPs*factor(var3)",
            test="linear",
            subscribers=[self.gwas_writer],
            output_prefix=self.output_name,
        )

        # Checking the file exists
        self.assertTrue(os.path.isfile(self.output_name))

        # Reading the data
        df = pd.read_csv(self.output_name, sep="\t")
        df = df.set_index("snp", drop=False).loc[["snp1", "snp2", "snp3"], :]

        # Checking all the columns are there
        self.assertEqual(
            set(df.columns),
            {"snp", "chr", "pos", "major", "minor", "maf", "n", "ll",
             "TRANSFORM:GWAS_INTER:level.x1:coef",
             "TRANSFORM:GWAS_INTER:level.x1:se",
             "TRANSFORM:GWAS_INTER:level.x1:lower",
             "TRANSFORM:GWAS_INTER:level.x1:upper",
             "TRANSFORM:GWAS_INTER:level.x1:t",
             "TRANSFORM:GWAS_INTER:level.x1:p",
             "TRANSFORM:GWAS_INTER:level.x2:coef",
             "TRANSFORM:GWAS_INTER:level.x2:se",
             "TRANSFORM:GWAS_INTER:level.x2:lower",
             "TRANSFORM:GWAS_INTER:level.x2:upper",
             "TRANSFORM:GWAS_INTER:level.x2:t",
             "TRANSFORM:GWAS_INTER:level.x2:p", "adj_r2", "subgroup"},
            "Missing columns in the GWAS Writer",
        )

        # Checking the coefficients (this should be enough to make sure we get
        # the right columns)
        #   var3=x1, var4=y0
        subgroup = df.subgroup == "var4:y0"
        np.testing.assert_array_almost_equal(
            df.loc[subgroup, "TRANSFORM:GWAS_INTER:level.x1:coef"].values,
            np.array([7.9838584795821, 0.8787655737690, -5.4257289463845]),
        )

        #   var3=x2, var4=y0
        np.testing.assert_array_almost_equal(
            df.loc[subgroup, "TRANSFORM:GWAS_INTER:level.x2:coef"].values,
            np.array([9.5318845882662, 2.1792184593254, -3.8901765902700]),
        )

        #   var3=x1, var4=y1
        subgroup = df.subgroup == "var4:y1"
        np.testing.assert_array_almost_equal(
            df.loc[subgroup, "TRANSFORM:GWAS_INTER:level.x1:coef"].values,
            np.array([-7.33292032058131, -2.53148534004492, -14.560258817242]),
        )

        #   var3=x2, var4=y1
        np.testing.assert_array_almost_equal(
            df.loc[subgroup, "TRANSFORM:GWAS_INTER:level.x2:coef"].values,
            np.array([9.91504894770588, -1.35580107828104, -3.1717946298856]),
        )
