

# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import os
import unittest
from tempfile import TemporaryDirectory
from itertools import zip_longest as zip

import numpy as np
import pandas as pd

from ..phenotypes.core import PhenotypesContainer
from ..phenotypes.text import TextPhenotypes


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestCore(unittest.TestCase):
    def test_get_phenotypes(self):
        """Tests that 'get_phenotypes' raises a NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            PhenotypesContainer().get_phenotypes()

    def test_close(self):
        """Tests that 'close' raises a NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            PhenotypesContainer().close()


class TestText(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory(prefix="project_x_")

        # The phenotype file
        self.pheno_file = os.path.join(self.tmpdir.name, "pheno.txt")
        with open(self.pheno_file, "w") as f:
            print("sample", "V1", "V2", "V3", file=f)
            print("s1", "1", "1.2", "a", file=f)
            print("s2", "234", "34.3", "sdc", file=f)
            print("s3", "-999999", "", "csgb", file=f)
            print("s4", "5463", "16.8", "999999", file=f)
            print("s5", "2356", "574.8", "jyrhg", file=f)
            print("s6", "", "567.2", "bvnchy", file=f)
            print("s7", "57634", "3134.3", "", file=f)
            print("s8", "346", "999999", "dfgfjgsd", file=f)
            print("s9", "34626", "3421.3", "fhyrj", file=f)

        # The expected data frame
        self.expected_pheno = pd.DataFrame(
            [("s1", 1, 1.2, "a"),
             ("s2", 234, 34.3, "sdc"),
             ("s3", -999999, np.nan, "csgb"),
             ("s4", 5463, 16.8, "999999"),
             ("s5", 2356, 574.8, "jyrhg"),
             ("s6", np.nan, 567.2, "bvnchy"),
             ("s7", 57634, 3134.3, np.nan),
             ("s8", 346, 999999, "dfgfjgsd"),
             ("s9", 34626, 3421.3, "fhyrj")],
            columns=["sample", "V1", "V2", "V3"],
        ).set_index("sample")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_normal_functionality(self):
        """Tests normal functionality."""
        observed_pheno = TextPhenotypes(
            fn=self.pheno_file,
            sep=" ",
        ).get_phenotypes()

        self.assertTrue(self.expected_pheno.equals(observed_pheno))

    def test_one_other_missing_value(self):
        """Tests when there are other missing values."""
        observed_pheno = TextPhenotypes(
            fn=self.pheno_file,
            sep=" ",
            missing_values="999999",
        ).get_phenotypes()

        # Changing the expected phenotypes
        self.expected_pheno.loc["s4", "V3"] = np.nan
        self.expected_pheno.loc["s8", "V2"] = np.nan

        # Comparing
        self.assertTrue(self.expected_pheno.equals(observed_pheno))

    def test_multiple_missing_values(self):
        """Tests when there are multiple missing values."""
        observed_pheno = TextPhenotypes(
            fn=self.pheno_file,
            sep=" ",
            missing_values=["-999999", "999999"],
        ).get_phenotypes()

        # Changing the expected phenotypes
        self.expected_pheno.loc["s3", "V1"] = np.nan
        self.expected_pheno.loc["s4", "V3"] = np.nan
        self.expected_pheno.loc["s8", "V2"] = np.nan

        # Comparing
        self.assertTrue(self.expected_pheno.equals(observed_pheno))

    def test_specific_column_missing_value(self):
        """Tests when there are specific missing values for some columns."""
        observed_pheno = TextPhenotypes(
            fn=self.pheno_file,
            sep=" ",
            missing_values={"V3": ["999999"]},
        ).get_phenotypes()

        # Changing the expected phenotypes
        self.expected_pheno.loc["s4", "V3"] = np.nan

        # Comparing
        self.assertTrue(self.expected_pheno.equals(observed_pheno))

    def test_string_representation(self):
        """Checks the string representation."""
        with TextPhenotypes(fn=self.pheno_file, sep=" ") as text_pheno:
            self.assertEqual(
                "TextPhenotypes(9 samples, 3 variables)",
                str(text_pheno),
            )
