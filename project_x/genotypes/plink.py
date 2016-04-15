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
from pyplink import PyPlink

from .core import GenotypesContainer, Representation, MarkerGenotypes


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["PlinkGenotypes"]


class PlinkGenotypes(GenotypesContainer):
    def __init__(self, prefix):
        """Instantiate a new PlinkGenotypes object.

        Args:
            prefix (str): the prefix of the Plink binary files.

        """
        self.bed = PyPlink(prefix)
        self.bim = self.bed.get_bim()
        self.fam = self.bed.get_fam()

        # We want to set the index for the FAM file
        try:
            self.fam = self.fam.set_index("iid", verify_integrity=True)

        except ValueError:
            self.fam["fid_iid"] = [
                "{fid}_{iid}".format(fid=fid, iid=iid)
                for fid, iid in zip(self.fam.fid, self.fam.iid)
            ]
            self.fam = self.fam.set_index("fid_iid", verify_integrity=True)

    def __repr__(self):
        """The string representation."""
        return "PlinkGenotypes({:,d} samples; {:,d} markers)".format(
            self.bed.get_nb_samples(),
            self.bed.get_nb_markers(),
        )

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

        Note
        ====
            PyPlink returns the genotypes in the additive format. So we only
            require to convert to genitypic representation if required.

        Note
        ====
            If the sample IDs are not unique, the index is changed to be the
            sample family ID and individual ID (i.e. fid_iid).

        """
        self.check_representation(representation)

        # Getting and formatting the genotypes
        result = self.create_geno_df(
            genotypes=self.bed.get_geno_marker(marker),
            samples=self.fam.index,
        )

        # Checking the format is fine
        result, minor, major = self.check_genotypes(
            genotypes=result,
            minor=self.bim.loc[marker, "a1"],
            major=self.bim.loc[marker, "a2"],
        )

        # Returning the value as ADDITIVE representation
        if representation == Representation.ADDITIVE:
            return MarkerGenotypes(genotypes=result, marker=marker,
                                   chrom=self.bim.loc[marker, "chrom"],
                                   pos=self.bim.loc[marker, "pos"],
                                   major=major, minor=minor)

        # Returning the value as GENOTYPIC representation
        if representation == Representation.GENOTYPIC:
            return MarkerGenotypes(genotypes=self.additive2genotypic(result),
                                   chrom=self.bim.loc[marker, "chrom"],
                                   pos=self.bim.loc[marker, "pos"],
                                   marker=marker, major=major, minor=minor)

    def iter_marker_genotypes(self, representation=Representation.ADDITIVE):
        """Returns a dataframe of genotypes encoded using the provided model.

        Args:
            representation (str): A valid genotype representation format (e.g.
                                  genotypes.core.REPRESENTATION.ADDITIVE).

        Returns:
            Genotypes: A named tuple containing the dataframe with the encoded
                       genotypes for all samples (the index of the dataframe
                       will be the sample IDs), the minor and major alleles.

        Note
        ====
            If the sample IDs are not unique, the index is changed to be the
            sample family ID and individual ID (i.e. fid_iid).

        """
        self.check_representation(representation)

        # Iterating over all markers
        for marker, result in self.bed.iter_geno():
            # Getting and formatting the genotypes
            result = self.create_geno_df(
                genotypes=result,
                samples=self.fam.index,
            )

            # Checking the format is fine
            result, minor, major = self.check_genotypes(
                genotypes=result,
                minor=self.bim.loc[marker, "a1"],
                major=self.bim.loc[marker, "a2"],
            )

            # Returning the value as ADDITIVE representation
            if representation == Representation.ADDITIVE:
                yield MarkerGenotypes(genotypes=result, marker=marker,
                                      chrom=self.bim.loc[marker, "chrom"],
                                      pos=self.bim.loc[marker, "pos"],
                                      major=major, minor=minor)

            # Returning the value as GENOTYPIC representation
            if representation == Representation.GENOTYPIC:
                yield MarkerGenotypes(
                    genotypes=self.additive2genotypic(result),
                    chrom=self.bim.loc[marker, "chrom"],
                    pos=self.bim.loc[marker, "pos"],
                    marker=marker, major=major, minor=minor,
                )
