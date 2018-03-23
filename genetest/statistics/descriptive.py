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
    """Computes the minor allele frequency using genotypes.

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


def get_sex_maf(genotypes, sexes, minor, major):
    """Computes the minor allele frequency for sexual chromosomes.

    Args:
        genotypes (pandas.Series): The genotypes.
        sexes (pandas.Series): The sex (0=female, 1=male).
        minor (str): The minor allele.
        major (str): The major allele.

    Returns:
        tuple: Returns the MAF, the minor allele, the major allele and a
               boolean telling if the markers were flip or not.

    Warning
    =======
        The ``sexes`` vector should have the same order as the genotypes one,
        and the males are coded as ``1``, while females are coded as ``0``.
        Missing sexes should be coded as ``NaN``.

    """
    # Keeping only the non-missing data
    not_missing = ~(np.isnan(genotypes) | np.isnan(sexes))
    if (~not_missing).all():
        return np.nan, minor, major, False

    # keeping only non missing values
    genotypes = genotypes[not_missing]
    sexes = sexes[not_missing]

    # The number of males and females
    nb_males = np.sum(sexes)
    nb_females = sexes.shape[0] - nb_males

    # Computing the MAF
    maf = (
        (-0.5 * np.dot(genotypes, sexes) + np.sum(genotypes))
        / (nb_males + (2 * nb_females))
    )

    if maf > 0.5:
        return 1 - maf, major, minor, True

    return maf, minor, major, False
