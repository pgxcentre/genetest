"""
Run a full statistical analysis.
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import sys
import queue
import pprint
import multiprocessing
import logging

from .modelspec import SNPs
from .statistics.descriptive import get_maf

import numpy as np


logger = logging.getLogger(__name__)


class Subscriber(object):
    """Abstract class for subscribers."""
    def init(self, modelspec):
        self.modelspec = modelspec

    def close(self):
        pass

    def handle(self, results):
        """Handle results from a statistical test."""
        raise NotImplementedError()

    @staticmethod
    def _apply_translation(translation, results):
        out = {}
        for k in results:
            if k in translation:
                out[translation[k]] = results[k]
            else:
                out[k] = results[k]
        return out


class ResultsMemory(Subscriber):
    def __init__(self):
        self.results = []

    def handle(self, results):
        self.results.append(Subscriber._apply_translation(
            self.modelspec.get_translations(),
            results
        ))

    def _get_gwas_results(self):
        out = {}
        for result in self.results:
            name = result["SNPs"]["name"]
            out[name] = result
        return out


class Print(Subscriber):
    def __init__(self, raw=False):
        self.raw = raw

    def handle(self, results):
        if self.raw:
            pprint.pprint(results)
            return

        pprint.pprint(Subscriber._apply_translation(
            self.modelspec.get_translations(),
            results
        ))


class RowWriter(Subscriber):
    def __init__(self, filename=None, columns=None, header=False, sep="\t",
                 append=False):
        self.header = header
        self.columns = columns
        self.sep = sep
        self.filename = filename

        if filename:
            self._f = open(filename, "a" if append else "w")
        else:
            self._f = None

        if self.header:
            header = self.sep.join([i[0] for i in self.columns])
            if filename is not None:
                self._f.write(header + "\n")
            else:
                print(header)

    def close(self):
        if self._f:
            self._f.close()

    def handle(self, results):
        row = []
        for name, result in self.columns:
            if isinstance(result, str):
                row.append(result)
            else:
                row.append(str(result.get(results)))

        row = self.sep.join(row)
        if self.filename is not None:
            self._f.write(row + "\n")
            self._f.flush()
        else:
            print(row)


def _invalid_subscriber(message, abort=None):
    """Logs the error from the subscriber."""
    logger.critical(
        "A subscriber for this analysis raised an exception. "
        "This is be because an invalid key was accessed from the results of "
        "the statistical test.\n"
        "Unknown field: '{}'".format(message)
    )
    if abort:
        abort.set()
    sys.exit(1)


def _missing(y, X):
    """Return a boolean vector indexing non-missing values."""
    y_missing = y.isnull().any(axis=1)
    X_missing = X.isnull().any(axis=1)
    return ~(y_missing | X_missing)


def _gwas_worker(q, results_q, failed, abort, fit, y, X):
    # Get a SNP.
    while not abort.is_set():
        # Get a SNP from the Queue.
        snp = q.get()

        # This is a check for a sentinel.
        if snp is None:
            q.put(None)
            results_q.put(None)
            logger.debug("Worker Done")
            return

        # Compute union between indices.
        union = snp.genotypes.index & X.index
        if len(union) == 0:
            raise ValueError(
                "Genotype and phenotype data have non-overlapping indices."
            )
            abort.set()
        no_geno = X.index.difference(snp.genotypes.index)

        # Set the genotypes.
        if no_geno.shape[0] > 0:
            X.loc[no_geno, "SNPs"] = np.nan
        X.loc[union, "SNPs"] = snp.genotypes.loc[union, "geno"]

        missing = _missing(y, X)

        # Computing MAF
        maf, minor, major, flip = get_maf(
            genotypes=X.loc[missing, "SNPs"],
            minor=snp.minor,
            major=snp.major,
        )
        if flip:
            X.loc[:, "SNPs"] = 2 - X.loc[:, "SNPs"]

        # Compute.
        try:
            results = fit(y[missing], X[missing])
        except Exception as e:
            logger.debug("Exception raised during fitting:", e)
            if snp.marker:
                failed.put(snp.marker)
            continue

        # Update the results for the SNP with metadata.
        results["SNPs"].update({
            "chrom": snp.chrom, "pos": snp.pos, "major": major,
            "minor": minor, "name": snp.marker,
        })
        results["SNPs"]["maf"] = maf

        results_q.put(results)


def execute(phenotypes, genotypes, modelspec, subscribers=None,
            variant_predicates=None, output_prefix=None):
    if subscribers is None:
        subscribers = [Print()]

    if variant_predicates is None:
        variant_predicates = []

    data = modelspec.create_data_matrix(phenotypes, genotypes)

    # Exclude samples with missing outcome or covariable.
    data = data.dropna()

    # Extract y and X matrices
    y_cols = tuple(modelspec.outcome.keys())
    y = data[[modelspec.outcome[col].id for col in y_cols]]
    X = data.drop(y.columns, axis=1)

    # Rename y columns
    y.columns = y_cols

    # GWAS context.
    if SNPs in modelspec.predictors:
        _execute_gwas(genotypes, modelspec, subscribers, y, X,
                      variant_predicates, output_prefix)

    # Simple statistical test.
    else:
        # There shouldn't be variant_predicates.
        if len(variant_predicates) != 0:
            logger.warning("Variant predicates are only used for GWAS "
                           "analyses.")

        # Get the statistical test.
        test = modelspec.test()

        # We don't need to worry about indexing or the sample order because
        # both parameters are from the same df.
        results = test.fit(y, X)

        # Update the results with the variant metadata.
        for entity in results:
            if entity in modelspec.variant_metadata:
                results[entity].update(modelspec.variant_metadata[entity])

        # Dispatch the results to the subscribers.
        for subscriber in subscribers:
            subscriber.init(modelspec)
            try:
                subscriber.handle(results)
            except KeyError as e:
                return _invalid_subscriber(e.args[0])


def _execute_gwas(genotypes, modelspec, subscribers, y, X, variant_predicates,
                  output_prefix):
        cpus = multiprocessing.cpu_count() - 1

        # Pre-initialize the subscribers.
        for subscriber in subscribers:
            subscriber.init(modelspec)

        # Create queues for failing SNPs and the consumer queue.
        failed = multiprocessing.Queue()
        q = multiprocessing.Queue(500)
        results = multiprocessing.Queue()
        abort = multiprocessing.Event()

        # Spawn the worker processes.
        workers = []
        for worker in range(cpus):
            this_y = y.copy()
            this_X = X.copy()
            fit = modelspec.test().fit

            worker = multiprocessing.Process(
                target=_gwas_worker,
                args=(q, results, failed, abort, fit, this_y, this_X)
            )

            workers.append(worker)
            worker.start()

        # Works signal the end of their work by appending None to the results
        # queue. Hence, there should be as many Nones as workers in the
        # results queue by the end of the results processing.
        done_workers = 0

        def _handle_result():
            """Asynchronously processes an entry from the results queue.

            Does nothing if the queue is empty, passes through the subscribers
            if it is not.

            Returns 1 if there was a None in the queue, 0 otherwise. This is
            useful to track the number of workers that finish.

            """
            try:
                res = results.get(False)
            except queue.Empty:
                return 0

            if res is None:
                return 1

            for subscriber in subscribers:
                try:
                    subscriber.handle(res)
                except KeyError as e:
                    _invalid_subscriber(e.args[0], abort)

            return 0

        # Start filling the consumer queue and listening for results.
        not_analyzed = []
        for snp in genotypes.iter_marker_genotypes():
            # Pass through the list of variant filtering predicates.
            try:
                if not all([f(snp) for f in variant_predicates]):
                    not_analyzed.append(snp.marker)
                    continue

            except StopIteration:
                break

            q.put(snp)

            # Handle results at the same time to avoid occupying too much
            # memory as the results queue gets filled.
            val = _handle_result()
            assert val == 0

        # Signal that there are no more SNPs to add.
        logger.debug("Done pushing SNPs")
        q.put(None)

        # Handle the remaining results.
        while done_workers != len(workers):
            done_workers += _handle_result()

        # Close the subscribers (important for opened IO streams).
        for subscriber in subscribers:
            subscriber.close()

        # Dump the failed SNPs to disk.
        failed.put(None)
        prefix = (output_prefix + "_") if output_prefix is not None else ""

        with open(prefix + "failed_snps.txt", "w") as f:
            while not failed.empty():
                snp = failed.get()

                if snp:
                    f.write(snp + '\n')

        # Dump the not analyzed SNPs to disk.
        with open(prefix + "not_analyzed_snps.txt", "w") as f:
            for snp in not_analyzed:
                f.write(snp + '\n')

        # Sanity check that there is nothing important left in the queues.
        queues_iter = zip(
            ('results', 'failed', 'q'), (results, failed, q)
        )
        for name, a_queue in queues_iter:
            while not a_queue.empty():
                val = a_queue.get()
                assert val is None, (name, val)

        # Join the workers.
        for worker in workers:
            worker.join()

        logger.info("Analysis complete.")
