"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import numpy as np


__all__ = ["get_freq"]


def get_freq(genotypes):
    """Computes the alternative allele frequency using genotypes.

    Args:
        genotypes (pandas.Series): The genotypes.

    Returns:
        float: the alternative allele frequency.

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
    # The sum of all genotypes (excluding missing ones)
    geno_sum = genotypes.sum(skipna=True)

    # Computing the number of genotypes (excluding the missing ones)
    nb_geno = genotypes.shape[0] - np.sum(genotypes.isnull())

    if nb_geno == 0:
        return np.nan

    # Returning the alternative allele frequency
    return geno_sum / (nb_geno * 2)
