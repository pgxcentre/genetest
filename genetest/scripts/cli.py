"""
"""

# This file is part of genetest
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import os
import sys
import shlex
import logging
import argparse

from ..subscribers import GWASWriter
from ..analysis import execute_formula
from .. import __version__, _LOG_FORMAT
from ..modelspec.predicates import NameFilter
from ..configuration import AnalysisConfiguration


_TESTING_MODE = False


# Configuring the log
logger = logging.getLogger("genetest")


def main():
    """The main function."""
    # Creating the option parser
    desc = ("Performs statistical analysis on genotypic data "
            "(version {}).".format(__version__))
    parser = argparse.ArgumentParser(description=desc)

    # The logging file handler
    logging_fh = None

    # Parsing the options
    args = parse_args(parser)

    try:
        # Checking the options
        check_args(args)

        if not _TESTING_MODE:
            sh = logging.StreamHandler()
            sh.setFormatter(_LOG_FORMAT)
            logger.addHandler(sh)
            logger.setLevel("INFO")

            # Adding the file handler to the logger
            logging_fh = logging.FileHandler(args.output + ".log", mode="w")
            logging_fh.setFormatter(_LOG_FORMAT)
            logger.addHandler(logging_fh)

        # Logging
        logger.info("This is {script_name} version {version}".format(
            script_name=os.path.basename(sys.argv[0]),
            version=__version__,
        ))
        logger.info("Arguments: {}".format(
            " ".join(shlex.quote(part) for part in sys.argv[1:]),
        ))

        # Getting the analysis configuration
        logger.info("Parsing configuration file")
        conf = AnalysisConfiguration(args.configuration)

        # Getting the phenotypes
        logger.info("Creating phenotypes container")
        phenotypes = conf.get_phenotypes()
        logger.info("  - {:,d} samples, {:,d} variables".format(
            phenotypes.get_nb_samples(),
            phenotypes.get_nb_variables(),
        ))
        if args.keep:
            phenotypes.keep_samples(set(args.keep.read().splitlines()))
            args.keep.close()
            logger.info(
                "  - {:,d} samples kept".format(phenotypes.get_nb_samples()),
            )

        # Getting the genotypes
        logger.info("Creating genotypes container")
        genotypes = conf.get_genotypes()
        logger.info("  - {:,d} samples, {:,d} markers".format(
            genotypes.get_number_samples(),
            genotypes.get_number_variants(),
        ))

        # Creating a list of variant predicates
        variant_predicates = []

        # The markers to extract
        if args.extract:
            to_extract = set(args.extract.read().splitlines())
            args.extract.close()

            logger.info("{:,d} variants will be extracted".format(
                len(to_extract),
            ))
            variant_predicates.append(NameFilter(extract=to_extract))

        # Checking the test information
        test = conf.get_model_test()
        formula = conf.get_model_formula()
        test_kwargs = conf.get_model_args()
        logger.info("Analysis: {}".format(test))
        logger.info("  - {}".format(formula))
        for k, v in test_kwargs.items():
            logger.info("  - {}: {}".format(k, v))

        # Creating a simple row subscriber
        subscribers = [GWASWriter(filename=args.output + ".txt", test=test)]

        # Starting the analysis
        execute_formula(
            phenotypes=phenotypes,
            genotypes=genotypes,
            formula=formula,
            test=test,
            test_kwargs=test_kwargs,
            subscribers=subscribers,
            variant_predicates=variant_predicates,
            output_prefix=args.output,
            maf_t=args.maf,
            cpus=args.nb_cpus,
        )

    # Catching the Ctrl^C
    except KeyboardInterrupt:
        logger.info("Cancelled by user")
        sys.exit(0)

    # Catching the CliError
    except CliError as e:
        logger.critical(e.message)
        parser.error(e.message)

    finally:
        # Closing the logging file
        if logging_fh:
            logging_fh.close()

        # Closing the "keep" file
        if args.keep and not args.keep.closed:
            args.keep.close()

        # Closing the "extract" file
        if args.extract and not args.extract.closed:
            args.extract.close()


def check_args(args):
    """Checks the arguments and options."""
    # Checking that configuration file exists
    if not os.path.isfile(args.configuration):
        raise CliError("{}: no such file.".format(args.configuration))

    # Checking the number of CPUs
    if args.nb_cpus < 1:
        raise CliError("{}: invalid number of CPUs".format(args.nb_cpus))

    # Checking the MAF
    if args.maf < 0 or args.maf > 0.5:
        raise CliError("{}: invalid MAF".format(args.maf))


def parse_args(parser):
    """Parses the command line options and arguments."""
    parser.add_argument("-v", "--version", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("--test", action=TestAction, nargs=0,
                        help="Execute the test suite and exit.")
    parser.add_argument(
        "--nb-cpus", type=int, metavar="NB", default=1,
        help="The number of processes to use for the analysis. [%(default)d]",
    )

    # The input options
    group = parser.add_argument_group("Input Options")
    group.add_argument(
        "--configuration", type=str, metavar="INI", required=True,
        help="The configuration file that describe the phenotypes, genotypes, "
             "and model.",
    )

    # The output options
    group = parser.add_argument_group("Output Options")
    group.add_argument(
        "--output", type=str, metavar="FILE", default="genetest_results",
        help="The output file prefix that will contain the results and other "
             "information. [%(default)s]",
    )

    # Some other options
    group = parser.add_argument_group("Other Options")
    group.add_argument(
        "--extract", type=argparse.FileType("r"), metavar="FILE",
        help="A file containing a list of marker to extract prior to the "
             "statistical analysis (one marker per line).",
    )
    group.add_argument(
        "--keep", type=argparse.FileType("r"), metavar="FILE",
        help="A file containing a list of samples to keep prior to the "
             "statistical analysis (one sample per line).",
    )
    group.add_argument(
        "--maf", type=float, default=0.01, metavar="MAF",
        help="The MAF threshold to include a marker in the analysis. "
             "[%(default).2f]",
    )

    return parser.parse_args()


class TestAction(argparse.Action):  # pragma: no cover
    def __call__(self, parser, namespace, values, option_string=None):
        import unittest
        from ..tests import test_suite
        unittest.TextTestRunner(verbosity=2).run(test_suite)
        parser.exit()


class CliError(Exception):
    """An Exception raised in case of a problem."""
    def __init__(self, msg):
        """Construction of the CliError class."""
        self.message = str(msg)

    def __str__(self):
        return self.message

    def __repr__(self):
        return "CliError: " + self.message
