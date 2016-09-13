

# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import unittest

import numpy as np
import pandas as pd

from ..statistics import descriptive


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestdescriptiveStats(unittest.TestCase):
    """Tests the descriptive statistics module."""
    def test_get_freq_without_null(self):
        """Tests when there are no missing genotypes."""
        genotypes = pd.Series([0, 0, 1, 0, 1, 0, 2, 0, 1, 0])
        freq = descriptive.get_freq(genotypes)
        self.assertAlmostEqual(5/20, freq, places=10)

    def test_get_freq_with_null(self):
        """Tests when there are missing genotypes."""
        genotypes = pd.Series([0, 0, 1, 0, np.nan, 0, np.nan, 0, 1, 0])
        freq = descriptive.get_freq(genotypes)
        self.assertAlmostEqual(2/16, freq, places=10)

    def test_get_freq_no_valid_genotypes(self):
        """Tests when all genotypes are missing."""
        genotypes = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                               np.nan, np.nan, np.nan, np.nan])
        freq = descriptive.get_freq(genotypes)
        self.assertTrue(np.isnan(freq))
