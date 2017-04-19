"""
Run a full statistical analysis.
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import time
import queue
import itertools
import multiprocessing
import logging
from itertools import chain

from .modelspec import SNPs, ModelSpec, PheWAS
from .modelspec.grammar import parse_formula
from .statistics import model_map
from .statistics.descriptive import get_maf
from . import subscribers as subscribers_module

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
_SUBSCRIBER_DEPRECATED = ("DeprecationWarning: Subscribers are now in the "
                          "'genetest.subscribers' module.")


class Subscriber(subscribers_module.Subscriber):
    def __init__(self, *args, **kwargs):
        logger.warning(_SUBSCRIBER_DEPRECATED)
        super().__init__(*args, **kwargs)


class ResultsMemory(subscribers_module.ResultsMemory):
    def __init__(self, *args, **kwargs):
        logger.warning(_SUBSCRIBER_DEPRECATED)
        super().__init__(*args, **kwargs)


class Print(subscribers_module.Print):
    def __init__(self, *args, **kwargs):
        logger.warning(_SUBSCRIBER_DEPRECATED)
        super().__init__(*args, **kwargs)


class RowWriter(subscribers_module.RowWriter):
    def __init__(self, *args, **kwargs):
        logger.warning(_SUBSCRIBER_DEPRECATED)
        super().__init__(*args, **kwargs)


def _missing(y, X):
    """Return a boolean vector indexing non-missing values."""
    y_missing = y.isnull().any(axis=1)
    X_missing = X.isnull().any(axis=1)
    return ~(y_missing | X_missing)


def _generate_sample_order(x_samples, geno_samples):
    """Create the order for samples between X and genotypes."""
    # Generating the order for the genotypes
    geno_order = np.array(
        [i for i, s in enumerate(geno_samples) if s in x_samples],
        dtype=int,
    )

    # Generating the order for the X matrix
    x_order = geno_samples.values[geno_order]

    if x_samples.duplicated().any():
        # The are duplicated samples, so we need to duplicate the genotype
        # indexes
        logger.debug("Duplicated samples found")
        counts = x_samples.value_counts()
        geno_order = np.array(
            list(chain(*[[geno_samples.get_loc(s)]*counts[s]
                         for s in x_order])),
            dtype=int,
        )

    return geno_order, x_order


def _gwas_worker(q, results_q, failed, abort, fit, y, X, samples):
    # The sample order (to add genotypes to the X data frame
    geno_index = None
    sample_order = None

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

        # Compute union between indices if not already done
        if sample_order is None or geno_index is None:
            # Checking the intersection
            geno_index, sample_order = _generate_sample_order(X.index, samples)

            # Do we have an intersect
            if geno_index.shape[0] == 0:
                abort.set()
                raise ValueError(
                    "Genotype and phenotype data have non-overlapping indices."
                )

        # Set all to missing genotypes
        X.loc[:, "SNPs"] = np.nan

        # Set the genotypes
        X.loc[sample_order, "SNPs"] = snp.genotypes[geno_index]

        not_missing = _missing(y, X)

        # Computing MAF
        maf, minor, major, flip = get_maf(
            genotypes=X.loc[not_missing, "SNPs"],
            minor=snp.coded,
            major=snp.reference,
        )
        if flip:
            X.loc[:, "SNPs"] = 2 - X.loc[:, "SNPs"]

        # Computing
        try:
            results = fit(y[not_missing], X[not_missing])
        except Exception as e:
            logger.debug("Exception raised during fitting:", e)
            if snp.variant.name:
                failed.put(snp.variant.name)
            continue

        # Update the results for the SNP with metadata.
        results["SNPs"].update({
            "chrom": snp.variant.chrom, "pos": snp.variant.pos, "major": major,
            "minor": minor, "name": snp.variant.name,
        })
        results["SNPs"]["maf"] = maf

        results_q.put(results)


def execute_formula(phenotypes, genotypes, formula, test, test_kwargs=None,
                    subscribers=None, variant_predicates=None,
                    output_prefix=None):

    model = parse_formula(formula)

    # Handle the statistical test.
    if test_kwargs is None:
        test_kwargs = {}

    if hasattr(test, "__call__"):
        model["test"] = lambda: test(**test_kwargs)
    else:
        model["test"] = lambda: model_map[test](**test_kwargs)

    # Handle the conditions and stratification.
    conditions = model.pop("conditions")
    if conditions is not None:
        model["stratify_by"] = [i["name"] for i in conditions]
        subgroups = [i["level"] for i in conditions]
    else:
        subgroups = None

    modelspec = ModelSpec(**model)

    return execute(phenotypes, genotypes, modelspec, subscribers,
                   variant_predicates, output_prefix, subgroups)


def execute(phenotypes, genotypes, modelspec, subscribers=None,
            variant_predicates=None, output_prefix=None, subgroups=None):
    if subscribers is None:
        subscribers = [subscribers_module.Print()]

    # Initialize the subscribers.
    for subscriber in subscribers:
        subscriber.init(modelspec)

    if variant_predicates is None:
        variant_predicates = []

    # We branch out early if it is a pheWAS analysis because the data
    # preparation steps are pretty different.
    if isinstance(modelspec.outcome, PheWAS):
        return _execute_phewas(phenotypes, genotypes, modelspec, subscribers,
                               variant_predicates, output_prefix, subgroups)

    data = modelspec.create_data_matrix(phenotypes, genotypes)

    # Exclude samples with missing outcome or covariable.
    data = data.dropna()

    # Extract y and X matrices
    y_cols = tuple(modelspec.outcome.keys())
    y = data[[modelspec.outcome[col].id for col in y_cols]]
    X = data.drop(y.columns, axis=1)

    # Rename y columns
    y.columns = y_cols

    # Drop uninformative factors.
    bad_cols = _get_uninformative_factors(X, modelspec)
    if len(bad_cols):
        logger.info(
            "After removing missing values, dropping ({}) factor levels that "
            "have no variation."
            "".format(len(bad_cols))
        )
        X = X.drop(bad_cols, axis=1)

    messages = {
        "skipped": [],
        "failed": []
    }

    if modelspec.stratify_by is not None:
        _execute_stratified(genotypes, modelspec, subscribers, y, X,
                            variant_predicates, subgroups, messages)
    elif SNPs in modelspec.predictors:
        _execute_gwas(genotypes, modelspec, subscribers, y, X,
                      variant_predicates, messages)
    else:
        _execute_simple(modelspec, subscribers, y, X, variant_predicates,
                        messages)

    for subscriber in subscribers:
        subscriber.close()

    prefix = (output_prefix + "_") if output_prefix is not None else ""
    if messages["failed"]:
        with open(prefix + "failed_snps.txt", "w") as f:
            for failed_snp in messages["failed"]:
                f.write("{}\n".format(failed_snp))

    if messages["skipped"]:
        with open(prefix + "not_analyzed_snps.txt", "w") as f:
            for failed_snp in messages["skipped"]:
                f.write("{}\n".format(failed_snp))


def _execute_phewas(phenotypes, genotypes, modelspec, subscribers,
                    variant_predicates, output_prefix, subgroups):
    # Check that invalid options were not set.
    if subgroups:
        raise NotImplementedError(
            "Stratified analyses for pheWAS are not supported yet."
        )

    if variant_predicates:
        raise ValueError(
            "Variant predicates can only be used in the context of GWAS "
            "analyses."
        )

    # Create the matrices.
    data = modelspec.create_data_matrix(phenotypes, genotypes)

    # Prepare synchronized data structures.
    results_queue = multiprocessing.Queue()
    phen_queue = multiprocessing.Queue()
    abort = multiprocessing.Event()

    # Create workers.
    cpus = 2
    predictors = modelspec.predictors
    workers = []
    fit = modelspec.test().fit
    for worker in range(cpus):
        this_data = data.copy()

        worker = multiprocessing.Process(
            target=_phewas_worker,
            args=(this_data, predictors, abort, fit, phen_queue, results_queue)
        )

        workers.append(worker)
        worker.start()

    # Fill the phenotypes queue.
    for entity in modelspec.outcome.li:
        if entity not in modelspec.predictors:
            phen_queue.put(entity)

    phen_queue.put(None)

    # Pass results to the subscribers until it's done.
    translations = modelspec.get_translations()
    n_workers_done = 0
    while n_workers_done != len(workers):
        try:
            res = results_queue.get(False)
        except queue.Empty:
            time.sleep(1)
            continue

        if res is None:
            n_workers_done += 1
            continue

        res["outcome"] = translations[res["outcome"].id]
        for subscriber in subscribers:
            try:
                subscriber.handle(res)
            except KeyError as e:
                subscribers_module.subscriber_error(e.args[0], abort)

    for worker in workers:
        worker.join()

    logger.info("PheWAS complete.")


def _phewas_worker(data, predictors, abort, fit, phen_queue, results_queue):

    predictors = [i.id for i in predictors]
    if "intercept" in data.columns:
        predictors.append("intercept")

    X = data[predictors]

    while not abort.is_set():
        # Get an outcome.
        y = phen_queue.get()

        # Sentinel check.
        if y is None:
            phen_queue.put(None)
            results_queue.put(None)
            logger.debug("Worker Done")
            return

        y_data = data[[y.id]]
        not_missing = _missing(y_data, X)

        try:
            results = fit(y_data[not_missing], X[not_missing])
        except Exception as e:
            logger.debug("Exception raised during fitting:", e)
            print(e)
            continue

        results["outcome"] = y
        results_queue.put(results)


def _execute_stratified(genotypes, modelspec, subscribers, y, X,
                        variant_predicates, subgroups, messages):
    # Levels.
    assert len(modelspec.stratify_by) == len(subgroups)

    var_levels = []
    for i, subgroup in enumerate(subgroups):
        if subgroup is None:
            levels = X[modelspec.stratify_by[i].id].dropna().unique()
            var_levels.append(levels)
        elif hasattr(subgroup, "__iter__"):
            var_levels.append(subgroup)
        else:
            var_levels.append([subgroup])

    # This create a iterable of all level combinations to analyze:
    # assume x = 1, 2 and y = 3
    # Then subsets = [[1, 3], [2, 3]]
    # The order from modelspec.stratify_by is kept.
    subsets = itertools.product(*var_levels)

    gwas_mode = SNPs in modelspec.predictors
    translations = modelspec.get_translations()

    for levels in subsets:
        # current_subset is an iterable of (entity, level) pairs.
        current_subset = list(zip(modelspec.stratify_by, levels))

        # Build the filtering vector.
        idx = np.logical_and.reduce(
            [X[var.id] == level for var, level in current_subset]
        )

        # Extract the stratification and execute the analysis.
        subset_info = {
            translations[var.id]: level for var, level in current_subset
        }
        for sub in subscribers:
            sub._update_current_subset(subset_info)

        # Drop columns that become uninformative after stratification.
        this_x = X.loc[idx, :]
        bad_cols = _get_uninformative_factors(this_x, modelspec)
        this_x = this_x.drop(bad_cols, axis=1)

        # Also drop the columns from the stratification variables.
        this_x = this_x.drop([i.id for i in modelspec.stratify_by], axis=1)

        # Make sure everything went ok.
        if this_x.shape[0] == 0:
            raise ValueError(
                "No samples left in subgroup analysis ({}). Are all requested "
                "levels valid?"
                "".format(subset_info)
            )
        elif this_x.shape[1] == 0:
            raise ValueError(
                "No columns left in subgroup analysis ({}). Maybe there are "
                "no samples with non-null values in the requested subgroup."
                "".format(subset_info)
            )

        if len(bad_cols):
            logger.info(
                "After stratification, dropping factor levels ({}) that have "
                "no variation ".format(len(bad_cols))
            )

        if gwas_mode:
            _execute_gwas(
                genotypes, modelspec, subscribers, y.loc[idx, :], this_x,
                variant_predicates, messages
            )
        else:
            _execute_simple(
                modelspec, subscribers, y.loc[idx, :], this_x,
                variant_predicates, messages
            )


def _get_uninformative_factors(df, modelspec):
    factor_cols = [
        i[2] for i in modelspec.transformations if i[0] == "ENCODE_FACTOR"
    ]

    bad_cols = []
    zero_cols = df.columns[df.sum() == 0]
    for col in zero_cols:
        for factor_col in factor_cols:
            if col.startswith(factor_col.id):
                bad_cols.append(col)
                break

    return bad_cols


def _execute_simple(modelspec, subscribers, y, X, variant_predicates,
                    messages):
    # There shouldn't be variant_predicates.
    if len(variant_predicates) != 0:
        logger.warning("Variant predicates are only used for GWAS "
                       "analyses.")

    # Get the statistical test.
    test = modelspec.test()

    logger.info(
        "Executing {} - Design matrix has shape: {}".format(test, X.shape)
    )

    # We don't need to worry about indexing or the sample order because
    # both parameters are from the same df.
    results = test.fit(y, X)

    # Update the results with the variant metadata.
    for entity in results:
        if entity in modelspec.variant_metadata:
            results[entity].update(modelspec.variant_metadata[entity])

    # Dispatch the results to the subscribers.
    for subscriber in subscribers:
        try:
            subscriber.handle(results)
        except KeyError as e:
            return subscribers_module.subscriber_error(e.args[0])


def _execute_gwas(genotypes, modelspec, subscribers, y, X, variant_predicates,
                  messages):
        cpus = multiprocessing.cpu_count() - 1

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
            samples = pd.Index(genotypes.get_samples())

            worker = multiprocessing.Process(
                target=_gwas_worker,
                args=(q, results, failed, abort, fit, this_y, this_X, samples)
            )

            workers.append(worker)
            worker.start()

        # Workers signal the end of their work by appending None to the results
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
                    subscribers_module.subscriber_error(e.args[0], abort)

            return 0

        # Start filling the consumer queue and listening for results.
        not_analyzed = []
        for snp in genotypes.iter_genotypes():
            # Pass through the list of variant filtering predicates.
            try:
                if not all([f(snp.genotypes) for f in variant_predicates]):
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

        # Dump the failed SNPs to disk.
        failed.put(None)

        while not failed.empty():
            snp = failed.get()
            if snp:
                messages["failed"].append(snp)

        # Dump the not analyzed SNPs to disk.
        for snp in not_analyzed:
            messages["skipped"].append(snp)

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
