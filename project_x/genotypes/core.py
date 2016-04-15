"""
"""


# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import numpy as np
import pandas as pd

from types import SimpleNamespace
from collections import namedtuple


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["GenotypesContainer"]


# Genotype representation
Representation = SimpleNamespace(
    ADDITIVE="additive",
    GENOTYPIC="genotypic",
    DOSAGE="dosage",
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
    def create_geno_df(genotypes, samples):
        """Creates the genotypes datafrane.

        Args:
            genotypes (numpy.ndarray): The genotypes.
            samples (list): The sample (in the required order).

        Returns:
            pandas.DataFrame: The genotypes

        """
        genotypes = pd.DataFrame({"geno": genotypes}, index=samples)
        genotypes.loc[genotypes.geno == -1, "geno"] = np.nan
        return genotypes

    @staticmethod
    def check_genotypes(genotypes, minor, major):
        """Checks that 0 -> homo major and 2 -> homo minor.

        Args:
            genotypes (pandas.DataFrame): The genotypes.
            minor (str): The minor allele.
            major (str): The major allele.

        Returns:
            tuple: The genotypes, the minor and major alleles.

        """
        # Checking we have 0 -> homo major and 2 -> homo minor
        geno_sum = genotypes.geno.sum(skipna=True)
        nb_geno = genotypes.shape[0] - np.sum(genotypes.geno.isnull())
        if geno_sum / (nb_geno * 2) > 0.5:
            genotypes.geno = 2 - genotypes.geno
            minor, major = major, minor

        return genotypes, minor, major

    @staticmethod
    def check_representation(representation):
        """Checks the representation.

        Args:
            representation (str): The representation to check.

        Raises a ValueError if the representation is invalid.

        """
        if representation not in vars(Representation).values():
            raise ValueError("{} is an invalid "
                             "representation".format(representation.upper()))

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

    @staticmethod
    def dosage2additive(genotypes):
        """Converts from 'dosage' representation to 'additive'.

        Args:
            genotypes (pandas.DataFrame): The dataframe containing the
                                          genotypes (with  a 'geno' column).

        Returns:
            pandas.DataFrame: The dataframe containing the additive
                              representation of the genotypes.

        """
        genotypes["geno"] = genotypes.geno.round(0)
        return genotypes

    @staticmethod
    def encode_chrom(chrom):
        """Encodes the chromosome to its numerical value.

        Args:
            chrom (str): The chromosome in string representation.

        Returns:
            int: The chromosome in integer representation.

        """
        if isinstance(chrom, int):
            return chrom

        chrom = chrom.upper()
        if chrom == "X":
            return 23
        elif chrom == "Y":
            return 24
        elif chrom == "XY" or chrom == "YX":
            return 25
        elif chrom == "M" or chrom == "MT":
            return 26

        try:
            return int(chrom)

        except:
            raise ValueError("{}: invalid chromosome".format(chrom))
