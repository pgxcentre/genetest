"""
"""


# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


from configparser import ConfigParser

from .genotypes import available_formats as geno_formats, \
                       format_map as geno_map
from .phenotypes import available_formats as pheno_formats, \
                        format_map as pheno_map


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
        self._configuration = ConfigParser()
        with open(filename, "r") as f:
            self._configuration.read_file(f)

        # Getting the required and available sections
        required_sections = {"Genotypes", "Phenotypes"}
        available_sections = set(self._configuration.sections())

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

    def configure_genotypes(self):
        """Configures the genotypes component."""
        # Getting the genotype section of the configuration file
        section = self._configuration["Genotypes"]

        # Getting the format of the genotypes
        if "format" not in section:
            raise ValueError("In the configuration file, the 'Genotypes' "
                             "section should contain the 'format' option "
                             "specifying the format of the genotypes "
                             "container.")
        geno_format = section.pop("format")

        # Checking if the format is valid
        if geno_format not in geno_formats:
            raise ValueError("Invalid 'format' for the 'Genotypes' section of "
                             "the configuration file.")

        # Getting the object to gather required and optional values
        self._geno_container = geno_map[geno_format]

        # Creating the dictionary that will contain all the arguments
        self._geno_arguments = {}
        self._geno_format = geno_format

        # Gathering all the required arguments
        self.retrieve_required_arguments(
            names=self._geno_container.get_required_arguments(),
            args=self._geno_arguments,
            config=section,
            section="Genotypes",
        )

        # Getting the optional arguments
        self.retrieve_optional_arguments(
            optional_args=self._geno_container.get_optional_arguments(),
            current_args=self._geno_arguments,
            config=section,
        )

        # Checking for the invalid sections
        self.check_invalid_options(config=section, section="Genotypes")

    def get_genotypes_arguments(self):
        """Returns the genotypes arguments."""
        return self._geno_arguments

    def get_genotypes_format(self):
        """Returns the genotypes format."""
        return self._geno_format

    def get_genotypes_container(self):
        """Returns the genotypes container."""
        return self._geno_container

    def configure_phenotypes(self):
        """Configures the phenotypes component."""
        # Getting the phenotype section of the configuration file
        section = self._configuration["Phenotypes"]

        # Getting the format of the phenotypes
        if "format" not in section:
            raise ValueError("In the configuration file, the 'Phenotypes' "
                             "section should contain the 'format' option "
                             "specifying the format of the phenotypes "
                             "container.")
        pheno_format = section.pop("format")

        # Checking if the format is valid
        if pheno_format not in pheno_formats:
            raise ValueError("Invalid 'format' for the 'Phenotypes' section "
                             "of the configuration file.")

        # Getting the object to gather required and optional values
        self._pheno_container = pheno_map[pheno_format]

        # Creating the dictionary that will contain all the arguments
        self._pheno_arguments = {}
        self._pheno_format = pheno_format

        # Gathering all the required arguments
        self.retrieve_required_arguments(
            names=self._pheno_container.get_required_arguments(),
            args=self._pheno_arguments,
            config=section,
            section="Phenotypes",
        )

        # Getting the optional arguments
        self.retrieve_optional_arguments(
            optional_args=self._pheno_container.get_optional_arguments(),
            current_args=self._pheno_arguments,
            config=section,
        )

        # Checking for the invalid sections
        self.check_invalid_options(config=section, section="Phenotypes")

    def get_phenotypes_arguments(self):
        """Returns the phenotypes arguments."""
        return self._pheno_arguments

    def get_phenotypes_format(self):
        """Returns the phenotypes format."""
        return self._pheno_format

    def get_phenotypes_container(self):
        """Returns the phenotypes container."""
        return self._pheno_container

    @staticmethod
    def retrieve_required_arguments(names, args, config, section):
        """Retrieves required arguments from a configuration section.

        Args:
            names (tuple): An tuple containing the names of the required
                           arguments to fetch.
            args (dict): The current arguments.
            config (dict): The configuration section in which to retrieve the
                           required arguments.
            section (str): The name of the configuration section.

        """
        for name in names:
            if name not in config:
                raise ValueError(
                    "The '{}' {} component requires the '{}' parameter in the "
                    "configuration file.".format(config["format"], section,
                                                 name)
                )
            args[name] = config.pop(name)

    @staticmethod
    def retrieve_optional_arguments(optional_args, current_args, config):
        """Retrieves the optional arguments (with default valule).

        Args:
            optional_args (dict): The available optional arguments with their
                                  default value(s).
            current_args (dict): The current arguments.
            config (dict): The configuration in which the values are to be
                           gathered (if present).

        """
        for arg_name, arg_value in optional_args.items():
            if arg_name in config:
                current_args[arg_name] = config.pop(arg_name)
            else:
                current_args[arg_name] = arg_value

    @staticmethod
    def check_invalid_options(config, section):
        """Checks if there are invalid options left in a configuration section.

        Note
        ----
            An invalid option is an option that is left after all options were
            parsed (both required and optional).

        """
        if len(config) > 0:
            raise ValueError(
                "Invalid options found in the '{}' section of the "
                "configuration file: {}.".format(
                    section,
                    ", ".join(sorted(config.keys())),
                )
            )
