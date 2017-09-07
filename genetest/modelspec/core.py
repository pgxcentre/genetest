"""
Utilities to build statistical models.
"""

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import pandas as pd


from geneparse.utils import genotype_to_df


class Variable(object):
    def __init__(self, name):
        self.name = name

    def get_data(self):
        raise NotImplemented()

    def __repr__(self):
        return "{}:{}".format(self.__class__.__name__, self.name)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.__class__ is other.__class__ and str(self) == str(other)


class Phenotype(Variable):
    def get_data(self, phenotypes, genotypes):
        # Returns a DataFrame
        return phenotypes.get_phenotypes([self.name])


class Genotype(Variable):
    def get_genotype(self, phenotypes, genotypes):
        li = genotypes.get_variant_by_name(self.name)
        if len(li) == 0:
            raise ValueError("Could not find variant: {}".format(self.name))
        elif len(li) == 1:
            return li[0]
        else:
            raise NotImplementedError(
                "Multi-allelic variants are not yet handled ({})."
                "".format(self.name)
            )

    def get_data(self, phenotypes, genotypes, allele_string=False):
        return genotype_to_df(
            self.get_genotype(phenotypes, genotypes),
            samples=genotypes.get_samples(),
            as_string=allele_string
        )


class Transformation(object):
    def __init__(self, *args, **kwargs):
        raise NotImplemented()

    def __call__(self, phenotypes, genotypes):
        raise NotImplemented()

    def __hash__(self):
        raise NotImplemented()

    def __eq__(self, other):
        raise NotImplemented()


class Factor(Transformation):
    def __init__(self, variable):
        self.variable = variable

    def __hash__(self):
        return hash("{}:{}".format(self.__class__.__name__, self.variable))

    def __eq__(self, other):
        return (
            self.__class__ is other.__class__ and
            self.variable == other.variable
        )

    def __call__(self, phenotypes, genotypes):
        if isinstance(self.variable, Genotype):
            g = self.variable.get_genotype(phenotypes, genotypes)

            df = _make_factor_from_genotype(
                g, samples=genotypes.get_samples()
            )

        else:
            df = self.variable.get_data(phenotypes, genotypes)

        # Take the unique levels and make the dummy variables.
        assert df.shape[1] == 1
        levels = sorted(df.iloc[:, 0].dropna().unique())

        out = {}
        for i, level in enumerate(levels):
            if i == 0:
                continue

        # TODO


def _make_factor_from_genotype(g, samples):
    """Encode genotypes as factors by using reference allele as reference."""
    name = g.variant.name if g.variant.name else "genotypes"
    df = pd.DataFrame(g.genotypes, index=samples, columns=[name])

    df["alleles"] = None

    hard_calls = df[name].round()
    df.loc[hard_calls == 0, "alleles"] = "0-{0}/{0}".format(g.reference)
    df.loc[hard_calls == 1, "alleles"] = "1-{0}/{1}".format(g.reference,
                                                            g.coded)
    df.loc[hard_calls == 2, "alleles"] = "2-{0}/{0}".format(g.coded)

    df = df[["alleles"]]
    df.columns = [name]

    return df


class ModelSpec(object):
    def __init__(self, outcome, predictors, test, no_intercept=False,
                 stratify_by=None):

        self.outcome = self._clean_outcome(outcome)
        self.predictors = self._clean_predictors(predictors)
        self.test = self._clean_test(test)
        self.no_intercept = no_intercept

        if hasattr(stratify_by, "__iter__"):
            self.stratify_by = stratify_by
        elif stratify_by is not None:
            self.stratify_by = list(stratify_by)
        else:
            self.stratify_by = None

        # SNP metadata is stored in the modelspec because it is obtained as a
        # consequence of building the data matrix.
        # Hence, it is more efficient to cache it than to request it.
        self.variant_metadata = {}

        # Is there GWAS interaction involved?
        self.has_gwas_interaction = False

    @property
    def dependencies(self):
        raise NotImplemented()

    def get_tested_variants(self):
        raise NotImplemented()

    def create_data_matrix(self, phenotypes, genotypes):
        raise NotImplemented()


# phenotypes = DependencyManager("PHENOTYPES")
# genotypes = DependencyManager("GENOTYPES")

# factor = TransformationManager("ENCODE_FACTOR")
# log10 = TransformationManager("LOG10")
# ln = TransformationManager("LN")
# pow = TransformationManager("POW")
# interaction = TransformationManager("INTERACTION")
# gwas_interaction = TransformationManager("GWAS_INTERACTION")
