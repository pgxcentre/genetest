"""
Run a full statistical analysis.
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import queue
import pprint
import multiprocessing
import logging

from .modelspec import SNPs
from .statistics import model_map


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


class Print(Subscriber):
    def __init__(self, raw=False):
        self.raw = raw

    def handle(self, results):
        if self.raw:
            pprint.pprint(results)
            return

        out = {}
        translations = self.modelspec.get_translations()

        for k in results:
            if k in translations:
                out[translations[k]] = results[k]
            else:
                out[k] = results[k]

        pprint.pprint(out)


class RowWriter(Subscriber):
    def __init__(self, filename=None, columns=None, header=False, sep="\t"):
        self.header = header
        self.columns = columns
        self.sep = sep
        self.filename = filename

        if filename:
            self._f = open(filename, "w")
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
            try:
                row.append(str(result.get(results)))
            except Exception as e:
                logger.warning(
                    "Error in RowWriter: Results key not found: {}"
                    "".format(e)
                )

        row = self.sep.join(row)
        if self.filename is not None:
            self._f.write(row + "\n")
            self._f.flush()
        else:
            print(row)


def _gwas_worker(q, results_q, failed, fit, y, X):
    # Get a SNP.
    while True:
        # Get a SNP from the Queue.
        snp = q.get()

        # This is a check for a sentinel.
        if snp is None:
            q.put(None)
            results_q.put(None)
            return

        # Set the genotypes.
        # TODO this assumes that the containers are synced.
        X["SNPs"] = snp.genotypes

        # Compute.
        try:
            results = fit(y, X)
        except Exception:
            if snp.marker:
                failed.put(snp.marker)
            continue

        # Update the results for the SNP with metadata.
        results["SNPs"].update({
            "chr": snp.chrom, "pos": snp.pos, "major": snp.major,
            "minor": snp.minor, "name": snp.marker
        })
        results_q.put(results)


def execute(phenotypes, genotypes, modelspec, subscribers=None):
    if subscribers is None:
        subscribers = [Print()]

    data = modelspec.create_data_matrix(phenotypes, genotypes)
    data = data.dropna()

    y = data[modelspec.outcome.id]
    X = data[[c for c in data.columns if c != modelspec.outcome.id]]

    # GWAS context.
    if SNPs in modelspec.predictors:
        _execute_gwas(genotypes, modelspec, subscribers, y, X)

    # Simple statistical test.
    else:
        # Get the statistical test.
        test = model_map[modelspec.test]()

        results = test.fit(y, X)

        # Update the results with the variant metadata.
        for entity in results:
            if entity in modelspec.variant_metadata:
                results[entity].update(modelspec.variant_metadata[entity])

        # Dispatch the results to the subscribers.
        for subscriber in subscribers:
            subscriber.init(modelspec)
            subscriber.handle(results)


def _execute_gwas(genotypes, modelspec, subscribers, y, X):
        test_class = model_map[modelspec.test]
        cpus = multiprocessing.cpu_count()

        # Pre-initialize the subscribers.
        for subscriber in subscribers:
            subscriber.init(modelspec)

        # Spawn the manager process.
        cpus -= 1

        # Create queues for failing SNPs and the consumer queue.
        failed = multiprocessing.Queue()
        q = multiprocessing.Queue(500)
        results = multiprocessing.Queue()

        # Spawn the worker processes.
        workers = []
        for worker in range(cpus):
            this_y = y.copy()
            this_X = X.copy()
            fit = test_class().fit

            worker = multiprocessing.Process(
                target=_gwas_worker,
                args=(q, results, failed, fit, this_y, this_X)
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
                subscriber.handle(res)

            return 0

        # Start filling the consumer queue and listening for results.
        for snp in genotypes.iter_marker_genotypes():
            q.put(snp)

            # Handle results at the same time to avoid occupying too much
            # memory as the results queue gets filled.
            _handle_result()

        # Signal that there are no more SNPs to add.
        q.put(None)

        # Handle the remaining results.
        while done_workers != len(workers):
            done_workers += _handle_result()

        # Close the subscribers (important for opened IO streams).
        for subscriber in subscribers:
            subscriber.close()

        # Dump the failed SNPs to disk.
        failed.put(None)
        with open("failed_snps.txt", "w") as f:
            while not failed.empty():
                snp = failed.get()

                if snp:
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
