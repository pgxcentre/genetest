"""
Run a full statistical analysis.
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import pprint

from .modelspec import SNPs
from .statistics import model_map


class Subscriber(object):
    """Abstract class for subscribers."""
    def init(self, modelspec):
        self.modelspec = modelspec

    def handle(self, results):
        """Handle results from a statistical test."""
        raise NotImplementedError()


class Print(Subscriber):
    def handle(self, results):
        pprint.pprint(results)


class RowWriter(Subscriber):
    def __init__(self, filename=None, columns=None, header=False, sep="\t"):
        self.header = header
        self.columns = columns
        self.sep = sep
        self.filename = filename

    def handle(self, results):
        if self.header:
            print(self.sep.join([i[0] for i in self.columns]))

        row = []
        for name, result in self.columns:
            row.append(str(result.get(results)))

        if self.filename is None:
            print(self.sep.join(row))


def execute(phenotypes, genotypes, modelspec, subscribers=None):

    # TODO if SNPs in modelspec.predictors:
    #          # GWAS Context.
    #          for snp in genotypes.iter_marker_genotypes():
    #              # snp has marker, chrom, pos, genotypes, major, minor.
    if subscribers is None:
        subscribers = [Print()]

    data = modelspec.create_data_matrix(phenotypes, genotypes)
    data = data.dropna()  # CHECK THIS TODO

    # Get the statistical test.
    test = model_map[modelspec.test]()

    # Assemble the data matrices.
    y = data[modelspec.outcome.id]

    print(modelspec.predictors)

    predictor_ids = set([i.id for i in modelspec.predictors])
    columns = [i for i in data.columns
               if i in predictor_ids or i.startswith("TRANSFORM:")]
    X = data[columns]

    results = test.fit(y, X)

    # Update the results with the variant metadata.
    for entity in results:
        if entity in modelspec.variant_metadata:
            results[entity].update(modelspec.variant_metadata[entity])

    # Dispatch the results to the subscribers.
    for subscriber in subscribers:
        subscriber.init(modelspec)
        subscriber.handle(results)
