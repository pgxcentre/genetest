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


__all__ = ["PhenotypesContainer"]


class PhenotypesContainer(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        raise NotImplementedError()

    def get_phenotypes(self, li=None):
        """Returns a dataframe of phenotypes.

        Args:
            li (list): A list of phenotypes to extract (default: None means
                       all phenotypes).

        Returns:
            pandas.DataFrame: A dataframe containing the phenotypes (with the
            sample IDs as index).

        """
        raise NotImplementedError()

    def get_nb_samples(self):
        """Returns the number of samples.

        Returns:
            int: The number of samples.

        """
        raise NotImplementedError()

    def get_nb_variables(self):
        """Returns the number of variables.

        Returns:
            int: The number of variables.

        """
        raise NotImplementedError()

    def is_repeated(self):
        """Check if the phenotypes contain repeated measurements.

        Returns:
            bool: ``True`` if the data contains repeated measurements,
            ``False`` otherwise.

        """
        raise NotImplementedError()

    def keep_samples(self, keep):
        """Keeps only a subset of samples.

        Args:
            keep (list): The list of samples to keep.

        """
        raise NotImplementedError()

    def get_sex(self):
        """Returns the sex for all samples.

        Returns:
            pandas.Series: The sex for all samples.

        """
        if not hasattr(self, "_sex_column") or self._sex_column is None:
            raise ValueError("No sex column was specified in Phenotype "
                             "container")

        # Extracting the sex
        sex = self._phenotypes.loc[:, self._sex_column]

        if self._repeated:
            # Checking if the sex are the same across repeated measures
            invalid = []
            group = sex.groupby(sex.index)
            for sample, values in group:
                if len(values.unique()) != 1:
                    invalid.append(sample)

            if len(invalid) > 0:
                raise ValueError(
                    "Samples with duplicated different sex: {}"
                    "".format(",".join(sorted(invalid))),
                )

            sex = group.first()

        # Checking we only have 0 and 1 (and maybe NaN)
        if len(set(sex.dropna().unique()) - {0.0, 1.0}) != 0:
            raise ValueError("Sex should only contain 0 and 1 (and maybe NaN)")

        return sex
