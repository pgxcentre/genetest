"""
"""

# This file is part of genetest.
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
    def __init__(self, filename, sample_column="sample", field_separator="\t",
                 missing_values=None, repeated_measurements=False,
                 keep_sample_column=False):
        """Instantiate a new TextPhenotypes object.

        Args:
            filename (str): The name of the text file containing the
                            phenotypes.
            sample_column (str): The name of the column containing the sample
                                 identification number (to fit with the
                                 genotypes).
            field_separator (str): The field separator (default is tabulation).
            missing_values (str or list or dict): The missing value(s).
            repeated_measurements (bool): Are the data containing repeated
                                          measurements (e.g. for MixedLM).

        """
        # TODO: Check the different column names
        self._phenotypes = pd.read_csv(filename, sep=field_separator,
                                       na_values=missing_values)

        # If there are repeated measurements, the sample column will have
        # duplicated values. We will set the index, but we won't verify its
        # integrity. Otherwise, we will check the index's integrity.
        self._repeated = repeated_measurements
        self._phenotypes = self._phenotypes.set_index(
            sample_column,
            verify_integrity=not repeated_measurements,
            drop=not keep_sample_column,
        )

        # Renaming the index
        self._phenotypes.index.name = "sample_id"

    def close(self):
        pass

    def __repr__(self):
        """The string representation."""
        return "TextPhenotypes({:,d} samples, {:,d} variables)".format(
            self.get_nb_samples(),
            self.get_nb_variables(),
        )

    def merge(self, other):
        """Merge this instance with another."""
        self._phenotypes = self._phenotypes.join(other._phenotypes)

    def get_phenotypes(self, li=None):
        """Returns a dataframe of phenotypes.

        Returns:
            pandas.DataFrame: A dataframe containing the phenotypes (with the
            sample IDs as index).

        """
        if li is None:
            return self._phenotypes

        # Check that all requested variables are available.
        missing_variables = set(li) - set(self._phenotypes.columns)
        if missing_variables:
            raise KeyError(
                "Some of the requested phenotypes are unavailable in the "
                "PhenotypeContainer: {}.".format(missing_variables)
            )

        return self._phenotypes.loc[:, list(li)]

    def get_nb_samples(self):
        """Returns the number of samples.

        Returns:
            int: The number of samples.

        """
        if self._repeated:
            # There are duplicated samples, so we count the number of unique
            # samples
            return self._phenotypes.index.unique().shape[0]
        else:
            return self._phenotypes.shape[0]

    def get_nb_variables(self):
        """Returns the number of variables.

        Returns:
            int: The number of variables.

        """
        return self._phenotypes.shape[1]

    def is_repeated(self):
        """Check if the phenotypes contain repeated measurements.

        Returns:
            bool: ``True`` if the data contains repeated measurements,
            ``False`` otherwise.

        """
        return self._repeated

    def keep_samples(self, keep):
        """Keeps only a subset of samples.

        Args:
            keep (set): The list of samples to keep.

        """
        self._phenotypes = self._phenotypes[self._phenotypes.index.isin(keep)]
