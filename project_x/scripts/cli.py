"""
"""


# This file is part of project_x.
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
import traceback
from multiprocessing import Pool

import pandas as pd

from .. import __version__, _LOG_FORMAT
from ..configuration import AnalysisConfiguration


_TESTING_MODE = False


# Configuring the log
logger = logging.getLogger("project_x")


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

        # Getting the configuration
        conf = AnalysisConfiguration(args.configuration)

        # Printing the information about the Genotypes container
        geno_args = conf.get_genotypes_arguments()
        logger.info("Genotypes container: " + conf.get_genotypes_format())
        for k in sorted(geno_args.keys()):
            logger.info("  - {}={}".format(k, repr(geno_args[k])))

        # Printing the information about the phenotype container
        pheno_args = conf.get_phenotypes_arguments()
        logger.info("Phenotypes container: " + conf.get_phenotypes_format())
        for k in sorted(pheno_args.keys()):
            logger.info("  - {}={}".format(k, repr(pheno_args[k])))

        # Printing the information about the statistics container
        stats_args = conf.get_statistics_arguments()
        logger.info("Statistics container: " + conf.get_statistics_model())
        for k in sorted(stats_args.keys()):
            logger.info("  - {}={}".format(k, repr(stats_args[k])))

        # Getting the phenotypes container
        pheno = get_phenotypes(
            container=conf.get_phenotypes_container(),
            arguments=pheno_args,
        )

        # Getting the genotypes container
        geno = get_genotypes(
            container=conf.get_genotypes_container(),
            arguments=geno_args,
        )

        # Getting the statistics container
        stats = get_statistics(
            container=conf.get_statistics_container(),
            arguments=stats_args,
        )

        # Getting the list of markers to extract
        if args.extract is not None:
            args.extract = get_list_from_file(args.extract)
            logger.info("{:,d} markers will be extracted prior to "
                        "analysis".format(len(args.extract)))

        if args.keep is not None:
            args.keep = get_list_from_file(args.keep)
            logger.info("{:,d} samples will be kept prior to "
                        "analysis".format(len(args.keep)))

        # Performing the analysis
        perform_analysis(
            phenotypes=pheno,
            genotypes=geno,
            statistics=stats,
            args=args,
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
        if logging_fh:
            logging_fh.close()


def perform_analysis(phenotypes, genotypes, statistics, args):
    """Performs the analysis.

    Args:
        phenotypes (PhenotypesContainer): The phenotype container.
        genotypes (GenotypesContainer): The genotype container.
        statistics (StatsModels): The statistics container.
        args (argparse.Namespace): The options and arguments.

    """
    # Creating the matrices for the analysis from the phenotype data
    pheno = phenotypes.get_phenotypes()
    repeated_measures = phenotypes.is_repeated()

    # If there are samples to keep, we keep them
    if args.keep is not None:
        pheno = pheno.loc[args.keep, :]
        if repeated_measures:
            raise NotImplementedError()

        else:
            pheno = pheno.loc[pheno.index.isin(args.keep), :]

    # Checking there is no 'geno' column in the phenotypes
    if "geno" in pheno.columns:
        raise CliError("There should not be a column named 'geno' in the "
                       "phenotypes.")

    # Creating the y and X matrices for fitting
    y, X = statistics.create_matrices(pheno)

    # If we need to adjust for a marker, we need to merge the data to the X
    # statistics
    if args.marker_adjust:
        logger.info("Adjusting for marker {}".format(args.marker_adjust))
        marker_info = genotypes.get_genotypes(args.marker_adjust)

        # Checking the column in the phenotypes
        if marker_info.marker in X.columns:
            raise CliError("There is already a '{}' column in the "
                           "phenotypes.".format(marker_info.marker))

        # Merging the genotypes of the marker to X
        if repeated_measures:
            y, X = statistics.merge_matrices_genotypes(
                y=y, X=X, genotypes=marker_info.genotypes,
                ori_samples=phenotypes.get_original_sample_names(),
                compute_interaction=False,
            )
        else:
            y, X = statistics.merge_matrices_genotypes(
                y=y, X=X, genotypes=marker_info.genotypes,
                compute_interaction=False,
            )

        # Renaming the column 'geno' to the name of the marker
        X = X.rename(columns={"geno": marker_info.marker})

    # Just to be sure, we check there are no null values in the matrices
    assert not y.isnull().any().any()
    assert not X.isnull().any().any()

    # The multiprocessing pool, if required
    pool = None
    if args.nb_process > 1:
        pool = Pool(processes=args.nb_process)

    try:
        # Iterating over markers
        if args.extract is not None:
            raise NotImplementedError()

        else:
            raise NotImplementedError()

    except Exception:
        if pool is not None:
            logger.critical("Terminating all processes in the pool")
            pool.terminate()
        logger.critical(traceback.format_exc())
        raise CliError("Something went wrong")

    finally:
        if pool is not None:
            logger.info("Closing the process pool")
            pool.close()


def compute_fitting():
    """Computes the fitting process."""
    pass


def get_phenotypes(container, arguments):
    """Gets the phenotypes from the phenotype container.

    Args:
        container (PhenotypesContainer): The phenotype container.
        arguments (dict): The arguments to use for the creation of the instance
                          of the phenotype container.

    Returns:
        PhenotypesContainer: The phenotypes instance.

    """
    logger.info("Reading the phenotypes")
    pheno = container(**arguments)
    logger.info("  - {:,d} samples".format(pheno.get_nb_samples()))
    logger.info("  - {:,d} variables".format(pheno.get_nb_variables()))
    return pheno


def get_genotypes(container, arguments):
    """Gets the genotypes from the genotype container.

    Args:
        container (GenotypesContainer): The genotypes container.
        arguments (dict): The argument to use for the creation of the instance
                          of the genotype container.

    Returns:
        GenotypesContainer: The genotypes instance.

    """
    logger.info("Reading the genotypes")
    geno = container(**arguments)
    logger.info("  - {:,d} samples".format(geno.get_nb_samples()))
    nb_markers = geno.get_nb_markers()
    if nb_markers is not None:
        logger.info("  - {:,d} markers".format(nb_markers))
    return geno


def get_statistics(container, arguments):
    """Gets the statistics container.

    Args:
        container (StatsModels): The statistics container.
        arguments (dict): The argument to use for the creation of the instance
                          of the statistics container.

    Returns:
        StatsModels: The statistics instance.

    """
    return container(**arguments)


def get_list_from_file(f):
    """Gets a list of elements from a file.

    Args:
        f (file): The (opened) file containing the list of elemetns.

    Returns:
        tuple: The list of the elements.

    """
    elements = tuple(f.read().splitlines())
    f.close()
    return elements


def check_args(args):
    """Checks the arguments and options."""
    # Checking that configuration file exists
    if not os.path.isfile(args.configuration):
        raise CliError("{}: no such file.".format(args.configuration))


def parse_args(parser):     # pragma: no cover
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
        metavar="CONF",
        required=True,
        help="The configuration file that describe the phenotypes, genotypes, "
             "and statistical model.",
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

    # Other statistical options
    group = parser.add_argument_group("Other Statistical Options")
    group.add_argument(
        "--marker-adjust",
        type=str,
        metavar="MARKER",
        help="Add a marker from the genotypes data to the predictors list.",
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
