"""
Run a full statistical analysis.
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


from .modelspec import SNPs 
from .statistics import model_map


class Subscriber(object):
    """Abstract class for subscribers."""
    def init(self, schema, translations):
        self.schema = schema
        self.translations = translations

    def handle(self, results):
        """Handle results from a statistical test."""
        raise NotImplementedError()


class Print(Subscriber):
    def handle(self, results):
        print(results)


def execute(phenotypes, genotypes, modelspec, subscribers=None):

    # TODO if SNPs in modelspec.predictors:
    #          # GWAS Context.
    #          for snp in genotypes.iter_marker_genotypes():
    #              # snp has marker, chrom, pos, genotypes, major, minor.
    if subscribers is None:
        subscribers = []

    translations = modelspec.get_translations()
    data = modelspec.create_data_matrix(phenotypes, genotypes)

    # Get the statistical test.
    test = model_map[modelspec.test]()

    # Assemble the data matrices.
    y = data[modelspec.outcome.id]
    X = data[[i.id for i in modelspec.predictors]]

    results = test.fit(y, X)
    for subscriber in subscribers:
        subscriber.init(test.schema, translations)
        subscriber.handle(results)
