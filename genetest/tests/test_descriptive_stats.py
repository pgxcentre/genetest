

# This file is part of genetest.
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


class TestDescriptiveStats(unittest.TestCase):
    """Tests the descriptive statistics module."""
    def test_get_maf_without_null(self):
        """Tests when there are no missing genotypes."""
        genotypes = pd.Series([0, 0, 1, 0, 1, 0, 2, 0, 1, 0])
        maf, minor, major, flip = descriptive.get_maf(
            genotypes, "A", "B",
        )
        self.assertAlmostEqual(5 / 20, maf, places=10)
        self.assertEqual("A", minor)
        self.assertEqual("B", major)
        self.assertFalse(flip)

    def test_get_maf_flip(self):
        """Tests when there are no missing genotypes."""
        genotypes = pd.Series([2, 2, 1, 2, 1, 2, 0, 2, 1, 2])
        maf, minor, major, flip = descriptive.get_maf(
            genotypes, "A", "B",
        )
        self.assertAlmostEqual(5 / 20, maf, places=10)
        self.assertEqual("B", minor)
        self.assertEqual("A", major)
        self.assertTrue(flip)

    def test_get_maf_with_null(self):
        """Tests when there are missing genotypes."""
        genotypes = pd.Series([0, 0, 1, 0, np.nan, 0, np.nan, 0, 1, 0])
        maf, minor, major, flip = descriptive.get_maf(
            genotypes, "A", "B",
        )
        self.assertAlmostEqual(2 / 16, maf, places=10)
        self.assertEqual("A", minor)
        self.assertEqual("B", major)
        self.assertFalse(flip)

    def test_get_maf_no_valid_genotypes(self):
        """Tests when all genotypes are missing."""
        genotypes = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                               np.nan, np.nan, np.nan, np.nan])
        maf, minor, major, flip = descriptive.get_maf(
            genotypes, "A", "B",
        )
        self.assertTrue(np.isnan(maf))
        self.assertEqual("A", minor)
        self.assertEqual("B", major)
        self.assertFalse(flip)
