"""
Container with fake genotype data for testing.
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import pandas as pd
import numpy as np

from .core import (
    GenotypesContainer, MarkerGenotypes, MarkerInfo, Representation
)


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["_DummyGenotypes"]


class _DummyGenotypes(GenotypesContainer):
    def __init__(self):
        self.data = pd.DataFrame({
            "snp1": [0, 0, 0, 1, 2, 0, 0, 0, 1, 0],
            "snp2": [1, 2, 0, 0, 0, 0, 1, 0, 0, 0],
            "snp3": [2, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "snp4": [0, 1, 2, 0, 0, 0, 1, 0, 1, 2],
            "snp5": [np.nan, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        }, index=["s{}".format(i) for i in range(1, 11)])

        self.snp_info = {
            "snp1": {"chrom": "3", "pos": 1234, "major": "C", "minor": "T"},
            "snp2": {"chrom": "3", "pos": 9618, "major": "A", "minor": "C"},
            "snp3": {"chrom": "2", "pos": 1519, "major": "T", "minor": "G"},
            "snp4": {"chrom": "1", "pos": 5871, "major": "A", "minor": "G"},
            "snp5": {"chrom": "X", "pos": 2938, "major": "C", "minor": "T"},
        }

        # Subset to 5 samples for now.
        self.data = self.data.loc[["s1", "s2", "s3", "s4", "s5"], :]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        pass

    def iter_marker_info(self):
        for snp, info in self.snp_info.values():
            yield MarkerInfo(marker=snp, **info)

    def get_genotypes(self, marker, representation=Representation.ADDITIVE):
        """Returns a dataframe of genotypes encoded using the provided model.

        Args:
            marker (str): A marker ID (e.g. rs123456).
            representation (str): A valid genotype representation format (e.g.
                                  genotypes.core.REPRESENTATION.ADDITIVE).

        Returns:
            MarkerGenotypes: A named tuple containing the dataframe with the
            encoded genotypes for all samples (the index of the dataframe will
            be the sample IDs), the minor and major alleles.

        """
        if representation is not Representation.ADDITIVE:
            raise NotImplementedError()

        genotypes = self.create_geno_df(self.data[marker], self.data.index)
        info = MarkerInfo(marker=marker, **self.snp_info[marker])
        return MarkerGenotypes(info, genotypes)

    def iter_marker_genotypes(self, representation=Representation.ADDITIVE):
        """Returns a dataframe of genotypes encoded using the provided model.

        Args:
            representation (str): A valid genotype representation format (e.g.
                                  genotypes.core.REPRESENTATION.ADDITIVE).

        Returns:
            MarkerGenotypes: A named tuple containing the dataframe with the
            encoded genotypes for all samples (the index of the dataframe will
            be the sample IDs), the minor and major alleles.

        """
        for snp in self.data.columns:
            yield self.get_genotypes(snp)

    def get_nb_samples(self):
        """Returns the number of samples.

        Returns:
            int: The number of samples.

        """
        return self.data.shape[0]

    def get_nb_markers(self):
        """Returns the number of markers.

        Returns:
            int: The number of markers.

        """
        return self.data.shape[1]
