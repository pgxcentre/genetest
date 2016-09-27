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
import pickle
import logging
import argparse
from tempfile import TemporaryDirectory
from multiprocessing import Pool, Queue, Process

import numpy as np

from .. import __version__, _LOG_FORMAT
from ..genotypes.core import genotype_reader
from ..configuration import AnalysisConfiguration
from ..statistics.core import statistics_initializer, statistics_worker


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
    reader_processes = []
    writer_proc = None

    # The temporary directory
    tmp_dir = None

    # The worker pool
    pool = None

    try:
        # Parsing the options
        args = parse_args(parser)
        check_args(args)

        if not _TESTING_MODE:
            sh = logging.StreamHandler()
            sh.setFormatter(_LOG_FORMAT)
            logger.addHandler(sh)
            logger.setLevel("INFO" if not args.debug else "DEBUG")

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

        # Some warnings about the readers
        if args.nb_readers > 5:
            logger.warning("Using multiple readers (i.e. {:,d}) might "
                           "decrease performances because of "
                           "disk IO".format(args.nb_readers))
        if conf.get_genotypes_format() == "plink" and args.nb_readers > 1:
            logger.warning("Using plink genotypes with more than one reader "
                           "will increase memory usage (linear to the number "
                           "of readers).")

        # Some warnings about the max queue size
        if args.max_queue_size > 1000:
            logger.warning("Using a high number of element in the waiting "
                           "queue (i.e. {:,d}) will use more memory (linear "
                           "to the number of element waiting in the "
                           "queue".format(args.max_queue_size))

        # Some warnings about the max marker chunk
        if args.max_chunk_size > 10000:
            logger.warning("Using a high number of markers in each chunk "
                           "(i.e. {:,d}) might freeze the workers if the "
                           "number of samples is too high (python "
                           "limitation).".format(args.max_chunk_size))

        # Adding a warning for the OPENBLAS_NUM_THREADS
        if int(os.environ.get("OPENBLAS_NUM_THREADS", "2")) > 1:
            logger.warning("If using openblas, be sure to set "
                           "OPENBLAS_NUM_THREADS environment variable to '1' "
                           "to make sure each worker uses only one processor.")

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

        # Getting the list of markers to extract
        to_extract = get_list_from_file(args.extract)
        logger.info("{:,d} markers will be extracted for the "
                    "analysis".format(len(to_extract)))

        # Creating the reader and writer queue
        reader_queue = Queue(args.max_queue_size)
        writer_queue = Queue()

        # Creating the temporary directory
        tmp_dir = TemporaryDirectory(dir=args.tmp)
        logger.info("Creating temporary directory: {}".format(tmp_dir.name))

        # Starting the reader processes
        logger.info("Starting {:,d} reader{}".format(
            args.nb_readers, "s" if args.nb_readers > 1 else "",
        ))
        logger.info("  - chunk size is {:,d}".format(args.max_chunk_size))
        for i, chunk in enumerate(np.array_split(to_extract, args.nb_readers)):
            proc = Process(
                target=genotype_reader,
                args=(conf.get_genotypes_container(), geno_args, chunk,
                      args.max_chunk_size, reader_queue, tmp_dir.name, i+1),
            )
            proc.start()
            reader_processes.append(proc)

        # Starting the writer process
        logger.info("Starting a writer")
        writer_proc = Process(
            target=writer,
            args=(args.output, writer_queue),
        )
        writer_proc.start()

        # Getting the list of samples to keep
        if args.keep is not None:
            to_keep = get_list_from_file(args.keep)
            logger.info("{:,d} samples will be kept for the "
                        "analysis".format(len(to_keep)))
            pheno.keep_samples(set(to_keep))

        # Creating the worker pool
        logger.info("Creating {:,d} worker{}".format(
            args.nb_workers, "s" if args.nb_workers > 1 else "",
        ))
        pool_args = dict(
            processes=args.nb_workers,
            initializer=statistics_initializer,
            initargs=[conf.get_statistics_container(), stats_args, pheno],
        )
        with Pool(**pool_args) as pool:
            perform_analysis(reader_queue, writer_queue, pool, args)

        # Waiting for the end of the writer process
        writer_queue.put(None)
        writer_proc.join()

    # Catching the Ctrl^C
    except KeyboardInterrupt:
        logger.info("Cancelled by user")
        sys.exit(0)

    # Catching the CliError
    except CliError as e:
        logger.critical(e.message)
        parser.error(e.message)

    finally:
        # Terminating the readers
        for i, proc in enumerate(reader_processes):
            if proc.is_alive():
                logger.info("Terminating reader {}".format(i+1))
                proc.terminate()

        # Terminating the writer
        if writer_proc is not None and writer_proc.is_alive():
            logger.info("Terminating the writer")
            writer_proc.terminate()

        # Cleaning up the temporary directory
        if tmp_dir is not None:
            logger.info("Cleaning up temporary directory")
            tmp_dir.cleanup()

        # Closing the log file
        if logging_fh:
            logging_fh.close()


def perform_analysis(reader_queue, writer_queue, worker_pool, arguments):
    """Performs the analysis.

    Args:
        reader_queue (multiprocessing.Queue): The readers' queue.
        writer_queue (multiprocessing.Queue): The writer's queue.
        worker_pool (multiprocessing.Pool): The worker pool.
        arguments (argparse.Namespace): The options and arguments.

    """
    # The total number of markers
    total_markers = 0

    # The actual analysis
    nb_finished = 0
    while nb_finished < arguments.nb_readers:
        # Getting the data
        fn = reader_queue.get()
        if fn is None:
            nb_finished += 1
            continue

        # Getting the data
        chunk = None
        with open(fn, "rb") as f:
            chunk = pickle.load(f)
        os.unlink(fn)

        # Logging
        logger.info("Analysing {:,d} markers".format(len(chunk)))
        total_markers += len(chunk)

        # Analyzing the data
        writer_queue.put(worker_pool.map(statistics_worker, chunk))

    # Final logging
    logger.info("Analysis performed on {:,d} markers".format(total_markers))


def writer(filename, queue):
    """Writes results from a queue to a file.

    Args:
        filename (str): The name of the file to write.
        queue (multiprocessing.Queue): The queue containing the results.

    """
    with open(filename, "w") as f:
        print_header = True
        chunk = queue.get()
        while chunk is not None:
            for result in chunk:
                if print_header:
                    print(*result._fields[:-2], sep="\t", end="\t", file=f)
                    print(*result.stats_n, sep="\t", file=f)
                    print_header = False

                # Printing the result
                print(*result[:-2], sep="\t", end="\t", file=f)
                print(*result.stats, sep="\t", file=f)

            # Reading the next chunk
            chunk = queue.get()

    logger.info("Closing the writer")


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

    # Checking the number of readers
    if args.nb_readers <= 0:
        raise CliError("needs at least 1 reader (--nb-reader)")

    # Checking the number of workers
    if args.nb_readers <= 0:
        raise CliError("needs at least 1 worker (--nb-worker)")

    # Checking the maximal size of the waiting queue
    if args.max_queue_size <= 0:
        raise CliError("needs at least 1 element in queue (--max-queue-size)")

    # Checking the maximal size of the marker chunk
    if args.max_chunk_size <= 0:
        raise CliError("needs at least 1 marker per chunk (--max-chunk-size)")


def parse_args(parser):     # pragma: no cover
    """Parses the command line options and arguments."""
    parser.add_argument("-v", "--version", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("--test", action=TestAction, nargs=0,
                        help="Execute the test suite and exit.")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Enters debug mode.")
    parser.add_argument("--tmp", metavar="DIR", default=".",
                        help="Path to temporary directory [$PWD].")

    # The input options
    group = parser.add_argument_group("Input Options")
    group.add_argument(
        "--configuration", type=str, metavar="CONF", required=True,
        help="The configuration file that describe the phenotypes, genotypes, "
             "and statistical model.",
    )

    # The output options
    group = parser.add_argument_group("Output Options")
    group.add_argument(
        "--output", type=str, metavar="FILE", default="results.txt",
        help="The output file that will contain the results from the "
             "statistical analysis. [%(default)s]",
    )

    # Multiprocessing options
    group = parser.add_argument_group("Multiprocessing Options")
    group.add_argument(
        "--nb-readers", type=int, metavar="NB", default=1,
        help="The number of reader processes to use. [%(default)d]",
    )
    group.add_argument(
        "--nb-workers", type=int, metavar="NB", default=1,
        help="The number of worker processes to use. [%(default)d]",
    )
    group.add_argument(
        "--max-queue-size", type=int, metavar="SIZE", default=100,
        help="The maximal number of marker chunks in the waiting queue. This "
             "will impact the amount of RAM the analysis will require. "
             "[%(default)d]",
    )
    group.add_argument(
        "--max-chunk-size", type=int, metavar="SIZE", default=10000,
        help="The maximal number of marker in a chunk. This will impact the "
             "amount of RAM in the analysis will require. [%(default)d]",
    )

    # Other statistical options
    group = parser.add_argument_group("Other Statistical Options")
    group.add_argument(
        "--marker-adjust", type=str, metavar="MARKER",
        help="Add a marker from the genotypes data to the predictors list.",
    )

    # Some other options
    group = parser.add_argument_group("Other Options")
    group.add_argument(
        "--extract", type=argparse.FileType("r"), metavar="FILE",
        required=True,
        help="A file containing a list of marker to extract prior to the "
             "statistical analysis. One marker per line.",
    )
    group.add_argument(
        "--keep", type=argparse.FileType("r"), metavar="FILE",
        help="A file containing a list of samples to keep prior to the "
             "statistical analysis. One sample per line.",
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
