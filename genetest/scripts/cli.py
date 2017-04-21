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

from .. import __version__, _LOG_FORMAT
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

    try:
        # Parsing the options
        args = parse_args(parser)
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
        conf = AnalysisConfiguration(args.configuration)

    # Catching the Ctrl^C
    except KeyboardInterrupt:
        logger.info("Cancelled by user")
        sys.exit(0)

    # Catching the CliError
    except CliError as e:
        logger.critical(e.message)
        parser.error(e.message)

    finally:
        if logging_fh:
            logging_fh.close()


def check_args(args):
    """Checks the arguments and options."""
    # Checking that configuration file exists
    if not os.path.isfile(args.configuration):
        raise CliError("{}: no such file.".format(args.configuration))


def parse_args(parser):
    """Parses the command line options and arguments."""
    parser.add_argument("-v", "--version", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("--test", action=TestAction, nargs=0,
                        help="Execute the test suite and exit.")
    parser.add_argument(
        "--nb-process",
        type=int,
        metavar="NB",
        default=1,
        help="The number of processes to use for the analysis. [%(default)d]",
    )

    # The input options
    group = parser.add_argument_group("Input Options")
    group.add_argument(
        "--configuration",
        type=str,
        metavar="INI",
        required=True,
        help="The configuration file that describe the phenotypes, genotypes, "
             "and model.",
    )

    # The output options
    group = parser.add_argument_group("Output Options")
    group.add_argument(
        "--output",
        type=str,
        metavar="FILE",
        default="results.txt",
        help="The output file that will contain the results from the "
             "statistical analysis. [%(default)s]",
    )

    # Some other options
    group = parser.add_argument_group("Other Options")
    group.add_argument(
        "--extract",
        type=argparse.FileType("r"),
        metavar="FILE",
        help="A file containing a list of marker to extract prior to the "
             "statistical analysis. One marker per line.",
    )
    group.add_argument(
        "--keep",
        type=argparse.FileType("r"),
        metavar="FILE",
        help="A file containing a list of samples to keep prior to the "
             "statistical analysis. One sample per line.",
    )
    group.add_argument(
        "--maf",
        type=float,
        default=0.01,
        metavar="MAF",
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
