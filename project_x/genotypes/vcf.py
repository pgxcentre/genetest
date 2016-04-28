"""
"""

# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import gzip
import pandas as pd
from pysam import VariantFile

from ..decorators import parameters
from .core import GenotypesContainer, Representation, MarkerGenotypes


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["VCFGenotypes"]


@parameters(required=("filename", ))
class VCFGenotypes(GenotypesContainer):
    def __init__(self, filename):
        """Instantiate a new VCFGenotypes object.

        Args:
            filename (str): The name of the VCF file.

        """
        self._vcf_file = filename
        self._vcf_reader = VariantFile(self._vcf_file)

        # Getting the samples for the file
        with gzip.open(self._vcf_file, "rb") as f:
            # Finding the header line
            line = next(f)
            while not(line.startswith(b"#CHROM")):
                line = next(f)
            row = line.decode().rstrip("\r\n").split("\t")

            # The samples are the columns after "FORMAT"
            self.samples = pd.Index(data=row[row.index("FORMAT")+1:],
                                    name="SampleID")

        # Checking for duplicated samples in the file
        if self.samples.has_duplicates:
            raise ValueError("duplicated samples")

        # The reader has not fetched yet
        self._has_fetched = False

    def close(self):
        self._vcf_reader.close()

    def __repr__(self):
        """The string representation."""
        return "VCFGenotypes({:,d} samples)".format(
            self.samples.shape[0],
        )

    def get_genotypes(self, chrom, pos,
                      representation=Representation.ADDITIVE):
        """Returns a dataframe of genotypes encoded using the provided model.

        Args:
            chrom (str): The chromosome on which the marker is located.
            pos (int): The position of the marker.
            representation (str): A valid genotype representation format (e.g.
                                  genotypes.core.REPRESENTATION.ADDITIVE).

        Returns:
            Genotypes: A named tuple containing the dataframe with the encoded
                       genotypes for all samples (the index of the dataframe
                       will be the sample IDs), the minor and major alleles.

        Note
        ====
            VCFs are only indexed according to genomic location (i.e.
            chromosome/position). Hence, we cannot retrieve the genotypes of a
            marker by using its name.

        """
        # Checking the genotype representation
        self.check_representation(representation)
        if representation == Representation.DOSAGE:
            raise ValueError("{} is an invalid representation for sequenced "
                             "data (it is usually used for imputed "
                             "data)".format(representation.upper()))

        # The parser has fetched once
        self._has_fetched = True

        # Returning the genotypes
        vcf_line = None
        try:
            vcf_line = next(self._vcf_reader.fetch(chrom, pos-1, pos))
        except StopIteration:
            raise ValueError("no marker positioned on chromosome {chrom}, "
                             "position {pos}".format(chrom=chrom, pos=pos))

        # Checking the number of alleles
        if len(vcf_line.alts) > 1:
            raise ValueError("{}: {}: more than two "
                             "alleles".format(chrom, pos))

        return self._create_genotypes(
            vcf_line=vcf_line,
            representation=representation,
        )

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
        # Checking the genotype representation
        self.check_representation(representation)
        if representation == Representation.DOSAGE:
            raise ValueError("{} is an invalid representation for sequenced "
                             "data (it is usually used for imputed "
                             "data)".format(representation.upper()))

        # The parser has fetched, so we need to close it and reopen it
        if self._has_fetched:
            self._vcf_reader.close()
            self._vcf_reader = VariantFile(self._vcf_file)
            self._has_fetched = False

        # Iterating over all markers
        for vcf_line in self._vcf_reader:
            # Checking the number of alleles
            if len(vcf_line.alts) > 1:
                continue

            yield self._create_genotypes(
                vcf_line=vcf_line,
                representation=representation,
            )

    def _create_genotypes(self, vcf_line, representation):
        """Creates the genotype dataframe from an binary Plink file.

        Args:
            vcf_line (vcf.model._Record): The VCF line to process.
            representation (str): The final genotype representation to use.

        Returns:
            pandas.DataFrame: The genotypes in the required representation.
        """
        additive = self.create_geno_df(
            genotypes=[
                self.format_genotype(vcf_line.samples[sample]["GT"])
                for sample in self.samples
            ],
            samples=self.samples,
        )

        # Checking the format is fine
        additive, minor, major = self.check_genotypes(
            genotypes=additive,
            minor=vcf_line.alts[0],
            major=vcf_line.ref,
        )

        # The ID of the marker could be unknown (.)
        chrom = self.encode_chrom(vcf_line.chrom)
        pos = vcf_line.pos
        marker = vcf_line.id
        if marker is None:
            marker = "{chrom}:{pos}".format(chrom=chrom, pos=pos)

        # Returning the value as ADDITIVE representation
        if representation == Representation.ADDITIVE:
            return MarkerGenotypes(genotypes=additive, marker=marker,
                                   chrom=chrom, pos=pos, major=major,
                                   minor=minor)

        # Returning the value as GENOTYPIC representation
        if representation == Representation.GENOTYPIC:
            return MarkerGenotypes(genotypes=self.additive2genotypic(additive),
                                   chrom=chrom, pos=pos, marker=marker,
                                   major=major, minor=minor)

    @staticmethod
    def format_genotype(geno):
        """Format a VCF genotype.

        Args:
            geno (list): A list of two alleles.

        Returns:
            int: The additive representation of the marker.

        Warning
        =======
            Only genotypes from a marker with two alleles should be analyzed by
            this function. The function recode allele higher than 0 to 1. For
            example, 0|2 is encoded as 0|1. This helps the creation of the
            additive format.

        """
        # Converting to integer and recoding the alternative allele
        if geno[0] is None or geno[1] is None:
            return -1

        # Formatting
        geno = [0 if int(g) == 0 else 1 for g in geno]

        # Returning the additive format
        return sum(geno)
