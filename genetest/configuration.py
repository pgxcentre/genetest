"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


from configparser import ConfigParser

from .phenotypes import available_formats as pheno_formats, \
                        format_map as pheno_map
from .statistics import available_models as stats_models, model_map


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "CC BY-NC 4.0"


__all__ = ["AnalysisConfiguration", "create_skeleton"]


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
        required_sections = {"Genotypes", "Phenotypes", "Statistics"}
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

        # Configuring the Statistics component
        self.configure_statistics()

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
            args_type=self._geno_container.get_arguments_type(),
            config=section,
            section="Genotypes",
        )

        # Getting the optional arguments
        self.retrieve_optional_arguments(
            optional_args=self._geno_container.get_optional_arguments(),
            args_type=self._geno_container.get_arguments_type(),
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
            args_type=self._pheno_container.get_arguments_type(),
            config=section,
            section="Phenotypes",
        )

        # Getting the optional arguments
        self.retrieve_optional_arguments(
            optional_args=self._pheno_container.get_optional_arguments(),
            args_type=self._pheno_container.get_arguments_type(),
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

    def configure_statistics(self):
        """Configures the statistics component."""
        # Getting the statistics section of the configuration file
        section = self._configuration["Statistics"]

        # Getting the statistical model
        if "model" not in section:
            raise ValueError("In the configuration file, the 'Statistics' "
                             "section should contain the 'model' option "
                             "specifying the statistical model required in "
                             "the analysis.")
        statistical_model = section.pop("model")

        # Checking if the format is valid
        if statistical_model not in stats_models:
            raise ValueError("Invalid 'model' for the 'Statistics' section "
                             "of the configuration file.")

        # Getting the object to gather required and optional values
        self._stats_container = model_map[statistical_model]

        # Creating the dictionary that will contain all the arguments
        self._stats_arguments = {}
        self._stats_model = statistical_model

        # Gathering all the required arguments
        self.retrieve_required_arguments(
            names=self._stats_container.get_required_arguments(),
            args=self._stats_arguments,
            args_type=self._stats_container.get_arguments_type(),
            config=section,
            section="Statistics",
        )

        # Getting the optional arguments
        self.retrieve_optional_arguments(
            optional_args=self._stats_container.get_optional_arguments(),
            args_type=self._stats_container.get_arguments_type(),
            current_args=self._stats_arguments,
            config=section,
        )

        # Checking for the invalid sections
        self.check_invalid_options(config=section, section="Statistics")

    def get_statistics_arguments(self):
        """Returns the statistics arguments."""
        return self._stats_arguments

    def get_statistics_model(self):
        """Returns the statistics model."""
        return self._stats_model

    def get_statistics_container(self):
        """Returns the statistics container."""
        return self._stats_container

    @staticmethod
    def retrieve_required_arguments(names, args, args_type, config, section):
        """Retrieves required arguments from a configuration section.

        Args:
            names (tuple): An tuple containing the names of the required
                           arguments to fetch.
            args (dict): The current arguments.
            args_type (dict): A dictionary containing the type of each
                              arguments.
            config (dict): The configuration section in which to retrieve the
                           required arguments.
            section (str): The name of the configuration section.

        """
        for name in names:
            if name not in config:
                raise ValueError(
                    "The {} component requires the '{}' parameter in the "
                    "configuration file.".format(section, name)
                )

            # Saving the argument
            args[name] = AnalysisConfiguration.set_type_to_argument(
                arg=config.pop(name),
                arg_type=args_type[name],
            )

    @staticmethod
    def retrieve_optional_arguments(optional_args, args_type, current_args,
                                    config):
        """Retrieves the optional arguments (with default valule).

        Args:
            optional_args (dict): The available optional arguments with their
                                  default value(s).
            args_type (dict): A dictionary containing the type of each
                              arguments.
            current_args (dict): The current arguments.
            config (dict): The configuration in which the values are to be
                           gathered (if present).

        """
        for arg_name, arg_value in optional_args.items():
            if arg_name in config:
                arg = AnalysisConfiguration.set_type_to_argument(
                    arg=config.pop(arg_name),
                    arg_type=args_type[arg_name],
                )
                current_args[arg_name] = arg
            else:
                current_args[arg_name] = arg_value

    @staticmethod
    def set_type_to_argument(arg, arg_type):
        """Sets the type to the argument.

        Args:
            arg (str): The argument.
            arg_type (type): The type of the argument.

        Returns:
            type: The argument casted to the required type.

        Note
        ----
            If the argument type is a list, the first (an only) element of that
            list is the type. The argument is split by coma and then each
            element is casted to the required type.

        """
        if isinstance(arg_type, list):
            return [arg_type[0](v) for v in arg.split(",")]

        elif arg_type is bool:
            if arg.upper() in {"F", "FALSE", "NO", "N"}:
                return False
            elif arg.upper() in {"T", "TRUE", "YES", "Y"}:
                return True
            else:
                raise ValueError(
                    "'{}' is not a valid boolean (True/False)".format(arg)
                )

        elif arg_type is str:
            # We want to catch the None
            if arg.upper() == "NONE":
                return None

            # We want to catch the \t
            if arg == r"\t":
                return "\t"

        return arg_type(arg)

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


def create_skeleton():
    """Creates the skeleton of a configuration file INI."""
    # The phenotypes
    _pretty_print_section(
        section_name="Phenotypes",
        container_map=pheno_map,
        container_type="format",
    )

    # The genotypes
    _pretty_print_section(
        section_name="Genotypes",
        container_map=geno_map,
        container_type="format",
    )

    # The statistical model
    _pretty_print_section(
        section_name="Statistics",
        container_map=model_map,
        container_type="model",
    )


def _pretty_print_section(section_name, container_map, container_type):
    """Pretty prints a section of the configuration file.

    Args:
        section_name (str): The name of the section.
        container_map (dict): A dictionary containing all containers for this
                              configuration section.
        container_type (str): The type of container (either 'format' for
                              genotypes and phenotypes, or 'model' for
                              statistics).

    """
    print("[{}]".format(section_name))
    for pheno in sorted(container_map.keys()):
        print()
        print("#" * (11 + len(pheno) + len(container_type)))
        print("# The '{}' {} #".format(pheno, container_type))
        print("#" * (11 + len(pheno) + len(container_type)))
        print("{}='{}'".format(container_type, pheno))
        container = container_map[pheno]

        # Printing the required arguments
        for arg in container.get_required_arguments():
            print("{}=".format(arg))

        # Printing the option arguments
        optional = container.get_optional_arguments()
        for arg in sorted(optional.keys()):
            print("# {}={}".format(arg, repr(optional[arg])))
    print("\n")
