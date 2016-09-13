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

from ..statistics.descriptive import get_freq


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["GenotypesContainer", "Representation", "MarkerGenotypes"]


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

    @classmethod
    def get_required_arguments(cls):
        """Returns the required arguments.

        Returns:
            tuple: The required arguments of the genotype container.

        """
        return cls._required_args

    @classmethod
    def get_optional_arguments(cls):
        """Returns the optional arguments.

        Returns:
            dict: The optional arguments (with default values) of the
                  genotype container.

        """
        return cls._optional_args

    @classmethod
    def get_arguments_type(cls):
        """Returns the arguments type.

        Returns:
            dict: The type of each arguments (both required and optional).

        """
        return cls._args_type

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
        genotypes.index.name = "sample_id"
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
        # Computing the frequency
        freq = get_freq(genotypes.geno)
        if freq > 0.5:
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
        if chrom.startswith("CHR"):
            chrom = chrom[3:]

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

    def get_nb_samples(self):
        """Returns the number of samples.

        Returns:
            int: The number of samples.

        """
        raise NotImplementedError()

    def get_nb_markers(self):
        """Returns the number of markers.

        Returns:
            int: The number of markers.

        """
        raise NotImplementedError()
