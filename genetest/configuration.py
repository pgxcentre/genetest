"""
"""

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import yaml

from geneparse import parsers as geno_map

from .statistics import available_models as available_tests
from .phenotypes import (
    available_formats as pheno_formats,
    format_map as pheno_map
)


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "CC BY-NC 4.0"


__all__ = ["AnalysisConfiguration"]


class AnalysisConfiguration(object):

    def __init__(self, filename):
        """Creates an instance of the class.

        Args:
            filename (str): The name of the configuration file.

        """
        # Reading the configuration file
        with open(filename, "r") as f:
            self._configuration = yaml.load(f)

        # Getting the required and available sections
        required_sections = {"genotypes", "phenotypes", "model"}
        available_sections = self._configuration.keys()

        # Checking if there are missing required sections
        missing_sections = required_sections - available_sections
        if len(missing_sections) > 0:
            raise ValueError(
                "Missing section(s) from the configuration file: {}.".format(
                    ", ".join(sorted(missing_sections)),
                )
            )

        # Checking if there are extra sections
        extra_sections = available_sections - required_sections
        if len(extra_sections) > 0:
            raise ValueError(
                "Invalid section(s) in the configuration file: {}.".format(
                    ", ".join(sorted(extra_sections)),
                )
            )

        # Configuring the Genotypes component
        self.configure_genotypes()

        # Configuring the Phenotypes component
        self.configure_phenotypes()

        # Configuring the model
        self.configure_model()

    def configure_genotypes(self):
        """Configures the genotypes component."""
        # Getting the genotype section of the configuration file
        section = self._configuration["genotypes"].copy()

        # Getting the format of the genotypes
        if "format" not in section:
            raise ValueError(
                "In the configuration file, the 'genotypes' section should "
                "contain the 'format' option specifying the format of the "
                "genotypes container."
            )
        geno_format = section.pop("format")

        # Checking if the format is valid
        if geno_format not in geno_map:
            raise ValueError(
                "Invalid 'format' ({}) for the 'genotypes' section of the "
                "configuration file.".format(geno_format)
            )

        # Getting the object to gather required and optional values
        self._geno_container = geno_map[geno_format]

        # We want to get the arguments for the genotype container
        self._geno_format = geno_format
        self._geno_arguments = self.retrieve_arguments(section)

    def get_genotypes_args(self):
        """Returns the genotypes arguments."""
        return self._geno_arguments

    def get_genotypes_format(self):
        """Returns the genotypes format."""
        return self._geno_format

    def get_genotypes(self):
        """Returns the genotypes container."""
        return self._geno_container(**self._geno_arguments)

    def configure_phenotypes(self):
        """Configures the phenotypes component."""
        # Getting the phenotype section of the configuration file
        section = self._configuration["phenotypes"].copy()

        # Getting the format of the phenotypes
        if "format" not in section:
            raise ValueError(
                "In the configuration file, the 'phenotypes' section should "
                "contain the 'format' option specifying the format of the "
                "phenotypes container."
            )
        pheno_format = section.pop("format")

        # Checking if the format is valid
        if pheno_format not in pheno_formats:
            raise ValueError(
                "Invalid 'format' ({}) for the 'phenotypes' section of the "
                "configuration file.".format(pheno_format)
            )

        # Getting the object to gather required and optional values
        self._pheno_container = pheno_map[pheno_format]

        # We want to get the arguments for the genotype container
        self._pheno_format = pheno_format
        self._pheno_arguments = self.retrieve_arguments(section)

    def get_phenotypes_args(self):
        """Returns the phenotypes arguments."""
        return self._pheno_arguments

    def get_phenotypes_format(self):
        """Returns the phenotypes format."""
        return self._pheno_format

    def get_phenotypes(self):
        """Returns the phenotypes container."""
        return self._pheno_container(**self._pheno_arguments)

    def configure_model(self):
        """Configures the statistics component."""
        # Getting the model section of the configuration file
        section = self._configuration["model"].copy()

        # Getting the required test
        if "test" not in section:
            raise ValueError(
                "In the configuration file, the 'model' section should "
                "contain the 'test' option specifying the statistical test "
                "to perform."
            )
        self._model_test = section.pop("test")

        # Checking the test is valid
        if self._model_test not in available_tests:
            raise ValueError(
                "Invalid 'test' for the 'model' section of the configuration "
                "file."
            )

        # Getting the formula
        if "formula" not in section:
            raise ValueError(
                "In the configuration file, the 'model' section should "
                "contain the 'formula' option specifying the statistical "
                "model to perform."
            )
        self._model_formula = section.pop("formula")

        # Getting the options
        self._model_arguments = self.retrieve_options(section)

    def get_model_args(self):
        """Returns the statistics arguments."""
        return self._model_arguments

    def get_model_test(self):
        """Returns the statistics model."""
        return self._model_test

    def get_model_formula(self):
        """Returns the model formula."""
        return self._model_formula

    @staticmethod
    def retrieve_arguments(config):
        """Retrieves arguments from the configuration.

        Args:
            config (configparser.SectionProxy): The section's configuration.

        Returns:
            dict: The arguments extracted from this section's configuration.

        """
        # The final arguments
        arguments = AnalysisConfiguration.retrieve_options(config)

        # Then, we gather the rest of the keys
        for key in config:
            arguments[key] = config[key]

        return arguments

    @staticmethod
    def retrieve_options(config):
        """Retrieves options from the configuration.

        Args:
            config (configparser.SectionProxy): The section's configuration.

        Returns:
            dict: The optional arguments.

        Note
        ====
            Only the restricted 'options' key is parsed, and has the following
            format: ``keyword=type:value`` (*e.g.* ``sep=str:,``). If there are
            multiple keyword/values, they are separated by a single coma.

        """
        options = {}

        if "options" in config:
            for k, v in config.pop("options").items():
                if v == r"\t":
                    v = "\t"
                options[k] = v

        return options
