"""
"""

# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


from collections import defaultdict

import pandas as pd

from .core import PhenotypesContainer


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["TextPhenotypes"]


class TextPhenotypes(PhenotypesContainer):
    def __init__(self, fn, sample_c="sample", sep="\t", missing_values=None,
                 repeated_measurements=False):
        """Instantiate a new Impute2Genotypes object.

        Args:
            fn (str): The name of the text file containing the phenotypes.
            sample_c (str): The name of the column containing the sample
                            identification number (to fit with the genotypes).
            sep (str): The field separator (default is tabulation).
            missing_values (str or list or dict): The missing value(s).
            repeated_measurements (bool): Are the data containing repeated
                                          measurements (e.g. for MixedLM).

        """
        self._phenotypes = pd.read_csv(fn, sep=sep, na_values=missing_values)

        # If there are repeated measurements, the sample column will have
        # duplicated values. We need to recode this to be able to set the index
        # properly. We will save the old samples in a different column for
        # later.
        if repeated_measurements:
            if "_ori_sample_names" in self._phenotypes.columns:
                raise ValueError("phenotypes should not contain a column "
                                 "named '_ori_sample_names'")

            # Recoding the samples
            sample_counter = defaultdict(int)
            sample_index = [s for s in self._phenotypes[sample_c]]
            for i in range(len(sample_index)):
                sample = sample_index[i]
                sample_index[i] = "{}_{}".format(
                    sample,
                    sample_counter[sample],
                )
                sample_counter[sample] += 1

            # Saving the original values
            self._phenotypes["_ori_sample_names"] = self._phenotypes[sample_c]

            # Changing the sample column
            self._phenotypes[sample_c] = sample_index

        # Setting the index
        self._phenotypes = self._phenotypes.set_index(
            sample_c,
            verify_integrity=True,
        )

        # Saving the original sample names for later use (if required)
        if repeated_measurements:
            self._ori_sample_names = self._phenotypes[["_ori_sample_names"]]
            self._phenotypes = self._phenotypes.drop(
                "_ori_sample_names",
                axis=1,
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

    def get_original_sample_names(self):
        """Returns the original samples (different if repeated measurements."""
        return self._ori_sample_names