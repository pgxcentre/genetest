"""
"""

# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import pandas as pd

from .core import PhenotypesContainer


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["TextPhenotypes"]


class TextPhenotypes(PhenotypesContainer):
    def __init__(self, fn, sample_c="sample", sep="\t", missing_values=None):
        """Instantiate a new Impute2Genotypes object.

        Args:
            fn (str): The name of the text file containing the phenotypes.
            sample_c (str): The name of the column containing the sample
                            identification number (to fit with the genotypes).
            sep (str): The field separator (default is tabulation).

        """
        self._phenotypes = pd.read_csv(fn, sep=sep, na_values=missing_values)

        # Setting the index
        self._phenotypes = self._phenotypes.set_index(
            sample_c,
            verify_integrity=True,
        )

    def close(self):
        pass

    def __repr__(self):
        """The string representation."""
        return "TextPhenotypes({:,d} samples, {:,d} variables)".format(
            self._phenotypes.shape[0],
            self._phenotypes.shape[1],
        )

    def get_phenotypes(self):
        """Returns a dataframe of phenotypes.

        Returns:
            pandas.DataFrame: A dataframe containing the phenotypes (with the
                              sample IDs as index).

        """
        return self._phenotypes
