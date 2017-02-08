"""
Dummy implementation of the phenotype container with test data.
"""

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import pandas as pd
import numpy as np

from .core import PhenotypesContainer


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["_DummyPhenotypes"]


class _DummyPhenotypes(PhenotypesContainer):
    def __init__(self):
        self.data = pd.DataFrame({
            "y": [0, 1, 0, np.nan, 0],
            "V1": [1, 2, 3, 4, 5],
            "V2": [0, np.nan, 1, 0, 1],
            "V3": [-2, 1, np.nan, 0, -1],
            "V4": [0.1, 0.2, 0.3, 0.4, 0.5],
            "V5": [0, 0, 0, 1, 1],
        }, index=["s{}".format(i) for i in range(1, 6)])

    def close(self):
        pass

    def get_phenotypes(self, li=None):
        if li is None:
            li = self.data.columns

        for i in li:
            if i not in self.data.columns:
                raise KeyError(i)

        return self.data.loc[:, list(li)]

    def get_nb_samples(self):
        return self.data.shape[0]

    def get_nb_variables(self):
        return self.data.shape[1]

    def is_repeated(self):
        return False

    def keep_samples(self, keep):
        self.data = self.data.loc[keep, :]
