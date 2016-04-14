"""
"""


# Genotype representation
ADDITIVE = "additive"
GENOTYPIC = "genotypic"


class GenotypeContainer(object):
    def get_genotypes(self, marker, representation=ADDITIVE):
        """Returns a dataframe of genotypes encoded using the provided model.

        Args:
            marker (str): A marker ID (e.g. rs123456).
            representation (str): A valid genotype representation format (e.g.
                                  genotypes.core.ADDITIVE).

        Returns:
            pandas.DataFrame: A dataframe containing the encoded genotypes for
                              all samples. The index of the dataframe will be
                              the sample IDs.

        """
        pass


    def iter_marker_genotypes(self, representation=ADDITIVE):
        """Returns a dataframe of genotypes encoded using the provided model.

        Args:
            representation (str): A valid genotype representation format (e.g.
                                  genotypes.core.ADDITIVE).

        Returns:
            pandas.DataFrame: A dataframe containing the encoded genotypes for
                              all samples. The index of the dataframe will be
                              the sample IDs.

        """
        pass
