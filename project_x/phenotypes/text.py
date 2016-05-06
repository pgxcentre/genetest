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

from ..decorators import arguments
from .core import PhenotypesContainer


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["TextPhenotypes"]


@arguments(required=(("filename", str), ),
           optional={"sample_column": (str, "sample"),
                     "field_separator": (str, "\t"),
                     "missing_values": ([str], None),
                     "repeated_measurements": (bool, False)})
class TextPhenotypes(PhenotypesContainer):
    def __init__(self, filename, sample_column, field_separator,
                 missing_values, repeated_measurements):
        """Instantiate a new Impute2Genotypes object.

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
        self._phenotypes = pd.read_csv(filename, sep=field_separator,
                                       na_values=missing_values)

        # If there are repeated measurements, the sample column will have
        # duplicated values. We need to recode this to be able to set the index
        # properly. We will save the old samples in a different column for
        # later.
        if repeated_measurements:
            if "_ori_sample_names_" in self._phenotypes.columns:
                raise ValueError("phenotypes should not contain a column "
                                 "named '_ori_sample_names_'")

            # Recoding the samples
            sample_counter = defaultdict(int)
            sample_index = [s for s in self._phenotypes[sample_column]]
            for i in range(len(sample_index)):
                sample = sample_index[i]
                sample_index[i] = "{}_{}".format(
                    sample,
                    sample_counter[sample],
                )
                sample_counter[sample] += 1

            # Saving the original values
            self._phenotypes["_ori_sample_names_"] = self._phenotypes[
                sample_column
            ]

            # Changing the sample column
            self._phenotypes[sample_column] = sample_index

        # Setting the index
        self._phenotypes = self._phenotypes.set_index(
            sample_column,
            verify_integrity=True,
        )

        # Renaming the index
        self._phenotypes.index.name = "sample_id"

        # Saving the original sample names for later use (if required)
        if repeated_measurements:
            self._ori_sample_names = self._phenotypes[["_ori_sample_names_"]]
            self._phenotypes = self._phenotypes.drop(
                "_ori_sample_names_",
                axis=1,
            )

        # Saving the repeated measurements
        self._repeated = repeated_measurements

    def close(self):
        pass

    def __repr__(self):
        """The string representation."""
        return "TextPhenotypes({:,d} samples, {:,d} variables)".format(
            self.get_nb_samples(),
            self.get_nb_variables(),
        )

    def get_phenotypes(self):
        """Returns a dataframe of phenotypes.

        Returns:
            pandas.DataFrame: A dataframe containing the phenotypes (with the
                              sample IDs as index).

        """
        return self._phenotypes

    def get_original_sample_names(self):
        """Returns the original samples (different if repeated measurements).

        Returns:
            pandas.DataFrame: The original sample names.

        """
        if self._repeated:
            return self._ori_sample_names
        raise ValueError("The phenotypes doesn't contain repeated "
                         "measurements")

    def get_nb_samples(self):
        """Returns the number of samples.

        Returns:
            int: The number of samples.

        """
        if self._repeated:
            return len(self._ori_sample_names._ori_sample_names_.unique())
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
            bool: True if the data contains repeated measurements, False
                  otherwise.

        """
        return self._repeated
