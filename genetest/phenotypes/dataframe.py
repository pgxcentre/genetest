"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["DataFrameContainer"]


class DataFrameContainer(object):
    def __init__(self, dataframe):
        self._phenotypes = dataframe

        # Checking for repeated measurements
        self._repeated = self._phenotypes.index.duplicated().any()

        # Renaming the index
        self._phenotypes.index.name = "sample_id"

    def __repr__(self):
        return "DataFramePhenotypes({:,d} samples, {:,d} variables)".format(
            self.get_nb_samples(),
            self.get_nb_variables(),
        )

    def close(self):
        pass

    def merge(self, other):
        """Merge this instance with another."""
        self._phenotypes = self._phenotypes.join(other._phenotypes)

    def get_phenotypes(self, li=None):
        """Returns a dataframe of phenotypes.

        Args:
            li (list): A list of phenotypes to extract (default: None means
                       all phenotypes).

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
            keep (list): The list of samples to keep.

        """
        self._phenotypes = self._phenotypes[self._phenotypes.index.isin(keep)]
