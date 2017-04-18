

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import os
import unittest
from tempfile import TemporaryDirectory

from ..phenotypes.text import TextPhenotypes
from ..statistics.models.survival import StatsCoxPH

from ..configuration import AnalysisConfiguration


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


@unittest.skip("Not implemented")
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
            print("[Phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[Genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith("Missing section"))

    def test_extra_section(self):
        """Tests when there is an extra section."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[Phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[Genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[Statistics]\nmodel=linear\noutcome=pheno\npredictors=age",
                  file=f)
            print("[DummySection]\nformat=none", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith("Invalid section"))

    def test_missing_phenotype_format(self):
        """Tests when the phenotype format is missing."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[Phenotypes]\nfilename=dummy_file", file=f)
            print("[Genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[Statistics]\nmodel=linear\noutcome=pheno\npredictors=age",
                  file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(
            "the 'Phenotypes' section should contain the "
            "'format'" in str(cm.exception)
        )

    def test_invalid_phenotype_format(self):
        """Tests when an invalid format for phenotypes is asked."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[Phenotypes]\nformat=nope\nfilename=dummy_file", file=f)
            print("[Genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[Statistics]\nmodel=linear\noutcome=pheno\npredictors=age",
                  file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith(
            "Invalid 'format' for the 'Phenotypes' section"
        ))

    def test_missing_genotype_format(self):
        """Tests when the genotype format is missing."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[Phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[Genotypes]\nprefix=dummy_prefix", file=f)
            print("[Statistics]\nmodel=linear\noutcome=pheno\npredictors=age",
                  file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(
            "the 'Genotypes' section should contain the "
            "'format'" in str(cm.exception)
        )

    def test_invalid_genotype_format(self):
        """Tests when an invalid format for genotypes is asked."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[Phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[Genotypes]\nformat=nope\nprefix=dummy_prefix", file=f)
            print("[Statistics]\nmodel=linear\noutcome=pheno\npredictors=age",
                  file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith(
            "Invalid 'format' for the 'Genotypes' section"
        ))

    def test_missing_statistics_model(self):
        """Tests when the statistics model is missing."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[Phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[Genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[Statistics]\noutcome=pheno\npredictors=age", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(
            "the 'Statistics' section should contain the "
            "'model'" in str(cm.exception)
        )

    def test_invalid_statistics_model(self):
        """Tests when an invalid model for statistics is asked."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[Phenotypes]\nformat=text\nfilename=dummy_file", file=f)
            print("[Genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[Statistics]\nmodel=nope\noutcome=pheno\npredictors=age",
                  file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith(
            "Invalid 'model' for the 'Statistics' section"
        ))

    def test_invalid_options(self):
        """Tests when there is an invalid option in a section."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[Phenotypes]\nformat=text\nfilename=dummy_file\nnope=nope",
                  file=f)
            print("[Genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[Statistics]\nmodel=linear\noutcome=pheno\npredictors=age",
                  file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith(
            "Invalid options found in the 'Phenotypes' section"
        ))

    def test_missing_required_argument(self):
        """Tests when there is a missing required argument."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[Phenotypes]\nformat=text",
                  file=f)
            print("[Genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[Statistics]\nmodel=linear\noutcome=pheno\npredictors=age",
                  file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertTrue(str(cm.exception).startswith(
            "The Phenotypes component requires the 'filename' parameter"
        ))

    def test_invalid_boolean(self):
        """Tests when there is an invalid boolean."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[Phenotypes]\nformat=text\nfilename=dummy_file\n"
                  "repeated_measurements=No", file=f)
            print("[Genotypes]\nformat=plink\nprefix=dummy_prefix", file=f)
            print("[Statistics]\nmodel=coxph\ntime_to_event=tte\nevent=e\n"
                  "predictors=age,weight\nnormalize=nope", file=f)

        with self.assertRaises(ValueError) as cm:
            AnalysisConfiguration(conf)

        self.assertEqual(
            "'nope' is not a valid boolean (True/False)",
            str(cm.exception),
        )

    def test_normal_functionality(self):
        """Tests the normal functionality."""
        conf = os.path.join(self.tmpdir.name, "conf.txt")
        with open(conf, "w") as f:
            print("[Phenotypes]\nformat=text\nfilename=dummy_file\n"
                  "field_separator=\\t\nrepeated_measurements=No\n"
                  "missing_values=-9", file=f)
            print("[Genotypes]\nformat=impute2\nfilename=fn\n"
                  "sample_filename=sample_fn\n"
                  "probability_threshold=0.95", file=f)
            print("[Statistics]\nmodel=coxph\ntime_to_event=tte\nevent=e\n"
                  "predictors=Age,weight\nnormalize=yes\n"
                  "interaction=None", file=f)

        # Getting the configuration
        conf = AnalysisConfiguration(conf)

        # Checking the genotypes
        observed_geno_args = conf.get_genotypes_arguments()
        self.assertEqual(4, len(observed_geno_args))
        self.assertEqual("fn", observed_geno_args["filename"])
        self.assertEqual("sample_fn", observed_geno_args["sample_filename"])
        self.assertEqual("dosage", observed_geno_args["representation"])
        self.assertEqual(0.95, observed_geno_args["probability_threshold"])
        self.assertEqual("impute2", conf.get_genotypes_format())
        self.assertTrue(conf.get_genotypes_container() is Impute2Genotypes)

        # Checking the phenotypes
        observed_pheno_args = conf.get_phenotypes_arguments()
        self.assertEqual(5, len(observed_pheno_args))
        self.assertEqual("dummy_file", observed_pheno_args["filename"])
        self.assertEqual("sample", observed_pheno_args["sample_column"])
        self.assertEqual("\t", observed_pheno_args["field_separator"])
        self.assertEqual(["-9"], observed_pheno_args["missing_values"])
        self.assertFalse(observed_pheno_args["repeated_measurements"])
        self.assertEqual("text", conf.get_phenotypes_format())
        self.assertTrue(conf.get_phenotypes_container() is TextPhenotypes)

        # Checking the statistics
        observed_stats_args = conf.get_statistics_arguments()
        self.assertEqual(5, len(observed_stats_args))
        self.assertEqual("tte", observed_stats_args["time_to_event"])
        self.assertEqual("e", observed_stats_args["event"])
        self.assertEqual(["Age", "weight"], observed_stats_args["predictors"])
        self.assertTrue(observed_stats_args["interaction"] is None)
        self.assertTrue(observed_stats_args["normalize"])
        self.assertEqual("coxph", conf.get_statistics_model())
        self.assertTrue(conf.get_statistics_container() is StatsCoxPH)
