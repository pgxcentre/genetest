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
import traceback
import logging
from itertools import chain
import warnings

import numpy as np
import pandas as pd

from .statistics.core import StatsError
from .statistics.descriptive import get_maf
from . import subscribers as subscribers_module
from .modelspec import SNPs, PheWAS, modelspec_from_formula


logger = logging.getLogger(__name__)


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


def _gwas_worker(q, results_q, failed, abort, fit, y, X, samples, maf_t=None):
    # The sample order (to add genotypes to the X data frame
    geno_index = None
    sample_order = None

    # Get a SNP.
    while not abort.is_set():
        # Get a SNP from the Queue.
        try:
            snp = q.get(timeout=1)
        except queue.Empty:
            # We waited for 1 seconds, just in case abort was set and no more
            # SNP is getting pushed in the Queue
            continue

        # This is a check for a sentinel.
        if snp is None:
            break

        # Compute union between indices if not already done
        if sample_order is None or geno_index is None:
            # Checking the intersection
            geno_index, sample_order = _generate_sample_order(X.index, samples)

            # Do we have an intersect
            if geno_index.shape[0] == 0:
                logger.critical("Genotype and phenotype data have "
                                "non-overlapping indices.")
                abort.set()
                continue

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

        # Is the MAF below the threshold?
        if maf_t is not None and maf < maf_t:
            failed.put((snp.variant.name, "MAF: {} < {}".format(maf, maf_t)))
            continue

        # Flipping if required
        if flip:
            X.loc[:, "SNPs"] = 2 - X.loc[:, "SNPs"]

        # Computing
        results = None
        try:
            with warnings.catch_warnings(record=True) as warning_list:
                results = fit(y[not_missing], X[not_missing])

                # Logging warnings
                _log_warnings(snp.variant.name, warning_list)

        except StatsError as e:
            logger.warning("{}: {}".format(snp.variant.name, e))
            if snp.variant.name:
                failed.put((snp.variant.name, str(e)))
            continue

        except Exception as e:
            logger.critical("{} was raised in worker\n{}".format(
                type(e).__name__, traceback.format_exc(),
            ))
            abort.set()
            continue

        # Update the results for the SNP with metadata.
        results["SNPs"].update({
            "chrom": snp.variant.chrom, "pos": snp.variant.pos, "major": major,
            "minor": minor, "name": snp.variant.name,
        })
        results["SNPs"]["maf"] = maf

        results_q.put(results)

    # The main loop was exited either because of an abort or because the
    # sentinel was encountered.
    # We put None to the queues to signal the worker is done and exit.
    q.put(None)
    results_q.put(None)
    failed.put(None)
    logger.debug("Worker Done")


def _log_warnings(identifier, warning_list):
    """Logs the warnings."""
    done = set()
    for w in [str(_.message) for _ in warning_list]:
        if w not in done:
            logger.warning("{}: {}".format(identifier, w))
            done.add(w)


def execute_formula(phenotypes, genotypes, formula, test, test_kwargs=None,
                    subscribers=None, variant_predicates=None,
                    output_prefix=None, maf_t=None, cpus=None):

    # Getting the model specification and the subgroups (if any)
    modelspec, subgroups = modelspec_from_formula(formula, test, test_kwargs)

    # Executing
    execute(phenotypes, genotypes, modelspec, subscribers, variant_predicates,
            output_prefix, subgroups, maf_t, cpus)


def execute(phenotypes, genotypes, modelspec, subscribers=None,
            variant_predicates=None, output_prefix=None, subgroups=None,
            maf_t=None, cpus=None):
    """Execute an analysis.

    Args:
        phenotypes (): The phenotypes container.
        genotypes (): The genotypes container.
        modelspec (genetest.modelspec.ModelSpec): The model specification.
        subscribers (list): A list of subscribers.
        variant_predicates (list): A list of variant predicates.
        output_prefix (str): The output prefix.
        subgroups (list): The subgroup analysis.
        maf_t (float): The MAF threshold.
        cpus (int): The number of CPUs to perform the analysis.

    """
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
                               variant_predicates, output_prefix, subgroups,
                               cpus)

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
                            variant_predicates, subgroups, messages, maf_t,
                            cpus)
    elif SNPs in modelspec.predictors:
        _execute_gwas(genotypes, modelspec, subscribers, y, X,
                      variant_predicates, messages, maf_t, cpus)
    else:
        _execute_simple(modelspec, subscribers, y, X, variant_predicates,
                        messages)

    for subscriber in subscribers:
        subscriber.close()

    prefix = (output_prefix + "_") if output_prefix is not None else ""
    if messages["failed"]:
        with open(prefix + "failed_snps.txt", "w") as f:
            for failed_snp in messages["failed"]:
                f.write("{}\t{}\n".format(*failed_snp))

    if messages["skipped"]:
        with open(prefix + "not_analyzed_snps.txt", "w") as f:
            for failed_snp in messages["skipped"]:
                f.write("{}\n".format(failed_snp))


def _execute_phewas(phenotypes, genotypes, modelspec, subscribers,
                    variant_predicates, output_prefix, subgroups, cpus=None):
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

    # The number if CPUs
    if cpus is None:
        cpus = multiprocessing.cpu_count() - 1

    # Create workers.
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
                        variant_predicates, subgroups, messages, maf_t=None,
                        cpus=None):
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

        logger.info("Analysing subgroup {}".format(
            ";".join("{}:{}".format(*_) for _ in sorted(subset_info.items()))
        ))

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
                variant_predicates, messages, maf_t, cpus,
            )
        else:
            _execute_simple(
                modelspec, subscribers, y.loc[idx, :], this_x,
                variant_predicates, messages,
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
                  messages, maf_t=None, cpus=None):
        if cpus is None:
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
                args=(q, results, failed, abort, fit, this_y, this_X, samples,
                      maf_t)
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
                if not all([f(snp) for f in variant_predicates]):
                    not_analyzed.append(snp.variant.name)
                    continue

            except StopIteration:
                break

            q.put(snp)

            # Handle results at the same time to avoid occupying too much
            # memory as the results queue gets filled.
            if abort.is_set():
                raise RuntimeError("Exception raised in worker processes")
            val = _handle_result()

            assert val == 0

        # Signal that there are no more SNPs to add.
        logger.debug("Done pushing SNPs")
        q.put(None)

        # Handle the remaining results.
        while done_workers != len(workers):
            done_workers += _handle_result()

        # Emptying the failed queue
        nb_failed_done = 0
        while nb_failed_done < cpus:
            snp = failed.get()
            if snp is None:
                nb_failed_done += 1
            else:
                messages["failed"].append(snp)
        failed.put(None)

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

        logger.info("Analysis completed")
