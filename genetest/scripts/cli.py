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
from os import path

import pandas as pd

from .. import modelspec as spec
from .. import __version__, _LOG_FORMAT
from ..modelspec.predicates import NameFilter
from ..modelspec import modelspec_from_formula
from ..analysis import execute, execute_formula
from ..configuration import AnalysisConfiguration
from ..modelspec import result as analysis_results
from ..phenotypes.dataframe import DataFrameContainer
from ..subscribers import GWASWriter, ResultsMemory, RowWriter


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

        # Checking for special test kwargs
        mixedlm_optimization = test_kwargs.get("optimize", True)
        mixedlm_p_threshold = test_kwargs.get("p_threshold", 1e-4)
        for to_del in ("optimize", "p_threshold"):
            if to_del in test_kwargs:
                del test_kwargs[to_del]

        if test == "mixedlm" and mixedlm_optimization:
            # Performing an "optimized" mixed linear model
            performed_optimized_mixedlm(
                args=args, test=test, phenotypes=phenotypes,
                genotypes=genotypes, formula=formula, test_kwargs=test_kwargs,
                variant_predicates=variant_predicates, p_t=mixedlm_p_threshold,
            )

        else:
            # Performing a "normal" analysis
            perform_normal_analysis(
                args=args, test=test, phenotypes=phenotypes,
                genotypes=genotypes, formula=formula, test_kwargs=test_kwargs,
                variant_predicates=variant_predicates,
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


def performed_optimized_mixedlm(args, test, phenotypes, genotypes, formula,
                                test_kwargs, variant_predicates, p_t):
    """Performs an "optimized" mixed linear model."""
    logger.info("Optimizing MixedLM analysis")

    # Getting the model specification and the subgroups (if any)
    modelspec, subgroups = modelspec_from_formula(formula, test, test_kwargs)

    # TODO: Find out if this is possible
    if subgroups is not None:
        raise CliError("Subgroups are not available for MixedLM optimization")

    # Removing the SNPs from the predictors
    if "SNPs" in modelspec.predictors:
        del modelspec.predictors[modelspec.predictors.index("SNPs")]

    # Executing the normal MixedLM analysis and executing the analysis
    logger.info("Computing the random effects")
    memory_subscriber = ResultsMemory()
    execute(
        phenotypes=phenotypes, genotypes=genotypes, modelspec=modelspec,
        subscribers=[memory_subscriber], output_prefix=args.output,
        subgroups=subgroups, maf_t=args.maf, cpus=args.nb_cpus,
    )

    # Getting the RE values
    random_effects = memory_subscriber.results.pop()["MODEL"]["random_effects"]

    # Getting the column for the groupings
    group_col = modelspec.get_translations()[modelspec.outcome["groups"].id]

    # Checking we don't have the random effects column in the original pheno
    assert "_random_effects" not in phenotypes.get_phenotypes().columns

    # We want to keep only the unique samples of the old phenotypes
    old_phenotypes = phenotypes.get_phenotypes()
    duplicated = old_phenotypes.index.duplicated(keep="first")

    # Merging the random effects to the phenotypes
    new_phenotypes = pd.merge(
        left=old_phenotypes.loc[~duplicated, :].copy(),
        right=pd.DataFrame(random_effects, columns=["_random_effects"]),
        left_on=group_col,
        right_index=True,
    )

    # Resetting the model spec
    spec._reset()

    # Creating the new phenotypes container
    new_phenotypes = DataFrameContainer(dataframe=new_phenotypes)
    assert not new_phenotypes.is_repeated()

    # Creating a new model spec
    optimized_modelspec = spec.ModelSpec(
        outcome=spec.phenotypes._random_effects,
        predictors=[spec.SNPs],
        test="linear",
    )

    # Creating a row subscriber
    approximation_fn = args.output + ".mixedlm_approximation.txt"
    subscribers = [RowWriter(
        filename=approximation_fn,
        columns=[
            ("snp", analysis_results["SNPs"]["name"]),
            ("chr", analysis_results["SNPs"]["chrom"]),
            ("pos", analysis_results["SNPs"]["pos"]),
            ("major", analysis_results["SNPs"]["major"]),
            ("minor", analysis_results["SNPs"]["minor"]),
            ("maf", analysis_results["SNPs"]["maf"]),
            ("n", analysis_results["MODEL"]["nobs"]),
            ("p", analysis_results["SNPs"]["p_value"]),
            ("ll", analysis_results["MODEL"]["log_likelihood"]),
            ("adj_r2", analysis_results["MODEL"]["r_squared_adj"]),
            ("approximation", "Yes")
        ],
        header=True,
        append=False,
    )]

    # Executing the optimized analysis
    logger.info("Executing the optimized MixedLM")
    execute(
        phenotypes=new_phenotypes, genotypes=genotypes,
        modelspec=optimized_modelspec, subscribers=subscribers,
        output_prefix=args.output + ".mixedlm_approximation",
        maf_t=args.maf, cpus=args.nb_cpus,
        variant_predicates=variant_predicates,
    )

    # Making sure the output file exists
    if not path.isfile(approximation_fn):
        raise CliError("{}: no such file".format(approximation_fn))

    # Reading the approximation
    approximation = pd.read_csv(approximation_fn, sep="\t")

    # Keeping only small p values (according to threshold)
    markers = set(approximation.loc[approximation.p < p_t, "snp"].values)

    if len(markers) == 0:
        logger.info("No marker had a p value < than {}".format(p_t))
        return

    # Adding the SNPs into the original ModelSpec
    modelspec.predictors.append("SNPs")

    # Creating the new variant predicates
    variant_predicates = [NameFilter(extract=markers)]

    # Creating the GWAS subscriber
    subscribers = [GWASWriter(filename=args.output + ".txt", test="mixedlm")]

    # Resetting the model spec
    spec._reset()

    # Executing the real MixdLM
    logger.info("Executing MixedLM on {:,d} markers".format(len(markers)))
    execute_formula(
        phenotypes=phenotypes, genotypes=genotypes, formula=formula,
        test="mixedlm", test_kwargs=test_kwargs, subscribers=subscribers,
        variant_predicates=variant_predicates, output_prefix=args.output,
        maf_t=args.maf, cpus=args.nb_cpus,
    )


def perform_normal_analysis(args, test, phenotypes, genotypes, formula,
                            test_kwargs, variant_predicates):
    """Performs a "normal" analysis."""
    # Creating a GWAS subscriber
    subscribers = [GWASWriter(filename=args.output + ".txt", test=test)]

    # Starting the analysis
    execute_formula(
        phenotypes=phenotypes, genotypes=genotypes, formula=formula, test=test,
        test_kwargs=test_kwargs, subscribers=subscribers,
        variant_predicates=variant_predicates, output_prefix=args.output,
        maf_t=args.maf, cpus=args.nb_cpus,
    )


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
