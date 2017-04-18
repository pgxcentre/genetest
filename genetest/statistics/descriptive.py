"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import numpy as np


__all__ = ["get_maf"]


def get_maf(genotypes, minor, major):
    """Computes the alternative allele frequency using genotypes.

    Args:
        genotypes (pandas.Series): The genotypes.
        minor (str): The minor allele.
        major (str): The major allele.

    Returns:
        tuple: Returns the MAF, the minor allele, the major allele and a
               boolean telling if the markers were flip or not.

    Note
    ====
        The frequency is computed using the alternative allele (i.e. the ``2``
        genotype). If there are more ``2`` genotypes than ``0`` genotypes, the
        alternative allele frequency will be higher than 0.5.

    Note
    ====
        When computing the alternative allele frequency, the missing genotypes
        are excluded. If there are no genotypes, ``NaN`` is returned.

    """
    maf = np.nanmean(genotypes) / 2

    if maf > 0.5:
        return 1 - maf, major, minor, True

    return maf, minor, major, False
