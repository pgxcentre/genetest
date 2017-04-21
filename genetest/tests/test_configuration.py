

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import os
import unittest
from tempfile import TemporaryDirectory

from ..configuration import AnalysisConfiguration


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestAnalysisConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = TemporaryDirectory(prefix="genetest_")

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    """Tests the 'AnalysisConfiguration' module."""
    def test_missing_section(self):
        """Tests when there is a missing section."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith("Missing section"))

    def test_extra_section(self):
        """Tests when there is an extra section."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[model]\ntest=linear\nformula=pheno~age", file=f)
            print("[dummy]\nformat=none", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith("Invalid section"))

    def test_missing_phenotype_format(self):
        """Tests when the phenotype format is missing."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[phenotypes]\nfilename=dummy_file", file=f)
            print("[genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[model]\ntest=linear\nformula=pheno~age", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(
            "the 'phenotypes' section should contain the "
            "'format'" in str(cm.exception)
        )

    def test_invalid_phenotype_format(self):
        """Tests when an invalid format for phenotypes is asked."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[phenotypes]\nformat=nope\nfilename=dummy_file", file=f)
            print("[genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[model]\ntest=linear\nformula=pheno~age", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith(
            "Invalid 'format' (nope) for the 'phenotypes' section"
        ))

    def test_missing_genotype_format(self):
        """Tests when the genotype format is missing."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[genotypes]\nprefix=dummy_prefix", file=f)
            print("[model]\ntest=linear\nformula=pheno~age", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(
            "the 'genotypes' section should contain the "
            "'format'" in str(cm.exception)
        )

    def test_invalid_genotype_format(self):
        """Tests when an invalid format for genotypes is asked."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[genotypes]\nformat=nope\nprefix=dummy_prefix", file=f)
            print("[model]\ntest=linear\nformula=pheno~age", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith(
            "Invalid 'format' (nope) for the 'genotypes' section"
        ))

    def test_missing_test(self):
        """Tests when the statistics test is missing."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[model]\nformula=pheno~age", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(
            "the 'model' section should contain the "
            "'test'" in str(cm.exception)
        )

    def test_invalid_test(self):
        """Tests when an invalid model for statistics is asked."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[model]\ntest=nope\nformula=pheno~age", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith(
            "Invalid 'test' for the 'model' section"
        ))

    def test_invalid_boolean(self):
        """Tests when there is an invalid boolean."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[phenotypes]\nformat=text\nfilename=dummy_file\n"
                  "options=repeated_measurements=bool:Nope", file=f)
            print("[genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[model]\ntest=linear\nformula=pheno~age", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertEqual(
            "'Nope' is not a valid boolean (True/False)",
            str(cm.exception),
        )

    def test_normal_functionality(self):
        """Tests the normal functionality."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[phenotypes]\nformat=text\nfilename=dummy_file\n"
                  "options=field_separator=str:\\t,"
                  "repeated_measurements=bool:No,"
                  "sample_column=str:sample_id,"
                  "missing_values=int:-9", file=f)
            print("[genotypes]\nformat=impute2\nfilename=fn\n"
                  "sample_filename=sample_fn\n"
                  "options=probability_threshold=float:0.95", file=f)
            print("[model]\ntest=linear\nformula=y~x+z\n"
                  "options=condition_value_t=int:3000", file=f)

        # Getting the configuration
        conf = AnalysisConfiguration(conf)

        # Checking the genotypes
        observed_geno_args = conf.get_genotypes_args()
        self.assertEqual(3, len(observed_geno_args))
        self.assertEqual("fn", observed_geno_args["filename"])
        self.assertEqual("sample_fn", observed_geno_args["sample_filename"])
        self.assertEqual(0.95, observed_geno_args["probability_threshold"])
        self.assertEqual("impute2", conf.get_genotypes_format())

        # Checking the phenotypes
        observed_pheno_args = conf.get_phenotypes_args()
        self.assertEqual(5, len(observed_pheno_args))
        self.assertEqual("dummy_file", observed_pheno_args["filename"])
        self.assertEqual("sample_id", observed_pheno_args["sample_column"])
        self.assertEqual("\t", observed_pheno_args["field_separator"])
        self.assertEqual(-9, observed_pheno_args["missing_values"])
        self.assertFalse(observed_pheno_args["repeated_measurements"])
        self.assertEqual("text", conf.get_phenotypes_format())

        # Checking the statistics
        observed_stats_args = conf.get_model_args()
        self.assertEqual(1, len(observed_stats_args))
        self.assertEqual(3000, observed_stats_args["condition_value_t"])
        self.assertEqual("y~x+z", conf.get_model_formula())
        self.assertEqual("linear", conf.get_model_test())
