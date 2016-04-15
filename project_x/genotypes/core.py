"""
"""


# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import numpy as np

from types import SimpleNamespace
from collections import namedtuple


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["GenotypesContainer"]


# Genotype representation
Representation = SimpleNamespace(
    ADDITIVE="additive",
    GENOTYPIC="genotypic",
)


# The genotypes that will be returned by the function
MarkerGenotypes = namedtuple(
    "MarkerGenotypes",
    ["marker", "chrom", "pos", "genotypes", "major", "minor"],
)


class GenotypesContainer(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        raise NotImplementedError()

    def get_genotypes(self, marker, representation=Representation.ADDITIVE):
        """Returns a dataframe of genotypes encoded using the provided model.

        Args:
            marker (str): A marker ID (e.g. rs123456).
            representation (str): A valid genotype representation format (e.g.
                                  genotypes.core.REPRESENTATION.ADDITIVE).

        Returns:
            Genotypes: A named tuple containing the dataframe with the encoded
                       genotypes for all samples (the index of the dataframe
                       will be the sample IDs), the minor and major alleles.

        """
        raise NotImplementedError()

    def iter_marker_genotypes(self, representation=Representation.ADDITIVE):
        """Returns a dataframe of genotypes encoded using the provided model.

        Args:
            representation (str): A valid genotype representation format (e.g.
                                  genotypes.core.REPRESENTATION.ADDITIVE).

        Returns:
            Genotypes: A named tuple containing the dataframe with the encoded
                       genotypes for all samples (the index of the dataframe
                       will be the sample IDs), the minor and major alleles.

        """
        raise NotImplementedError()

    @staticmethod
    def check_representation(representation):
        """Checks the representation.

        Args:
            representation (str): The representation to check.

        Raises a ValueError if the representation is invalid.

        """
        if representation not in vars(Representation).values():
            raise ValueError("{} is an invalid "
                             "representation".format(representation))

    @staticmethod
    def additive2genotypic(genotypes):
        """Converts from 'additive' representation to 'genotypic'.

        Args:
            genotypes (pandas.DataFrame): The dataframe containing the
                                          genotypes (with a 'geno' column).

        Returns:
            pandas.DataFrame: The dataframe containing the genotypic
                              representation (with 'geno_ab' and 'geno_bb'
                              column).

        """
        genotypes["geno_ab"] = [1 if g == 1 else 0 for g in genotypes.geno]
        genotypes["geno_bb"] = [1 if g == 2 else 0 for g in genotypes.geno]
        genotypes.loc[genotypes.geno.isnull(), ["geno_ab", "geno_bb"]] = np.nan
        return genotypes.loc[:, ["geno_ab", "geno_bb"]]
