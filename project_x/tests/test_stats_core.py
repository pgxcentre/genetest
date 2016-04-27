

# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import unittest

import numpy as np
import pandas as pd

from ..statistics.core import StatsModels


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestStatsLinear(unittest.TestCase):
    """Tests the 'StatsLinear' class."""
    @classmethod
    def setUpClass(cls):
        cls.core_model = StatsModels()

    def setUp(self):
        # The data
        data = pd.DataFrame(
            [("s0", 3.0, 12.0, "f1"), ("s1", 3.4, 30.8, "f2"),
             ("s2", 5.3, 50.2, "f1"), ("s3", 6.0, 30.6, "f3"),
             ("s4", 0.5, 40.0, "f2"), ("s5", 2.4, 80.5, "f1"),
             ("s6", 5.6, 70.0, "f1"), ("s7", 7.6, 87.4, "f2"),
             ("s8", 0.3, 63.0, "f1"), ("s9", 1.9, 54.3, "f3")],
            columns=["sample_id", "pheno", "var1", "var2"],
        ).set_index("sample_id")
        self.data = data.iloc[np.random.permutation(len(data))]

        # The genotypes
        genotypes = pd.DataFrame(
            [("s0", 0.0), ("s1", 1.0), ("s2", 0.0), ("s3", np.nan),
             ("s4", 0.0), ("s5", 1.0), ("s6", 2.0), ("s7", np.nan),
             ("s8", 0.0), ("s9", 0.0)],
            columns=["iid", "geno"],
        ).set_index("iid")
        self.genotypes = genotypes.iloc[np.random.permutation(len(genotypes))]

        # The expected values for y and X
        self.expected_y = pd.DataFrame(
            [("s0", 3.0), ("s1", 3.4), ("s2", 5.3), ("s3", 6.0), ("s4", 0.5),
             ("s5", 2.4), ("s6", 5.6), ("s7", 7.6), ("s8", 0.3), ("s9", 1.9)],
            columns=["sample_id", "pheno"],
        ).set_index("sample_id")
        self.expected_X = pd.DataFrame(
            [("s0", 1.0, 0.0, 0.0, 12.0), ("s1", 1.0, 1.0, 0.0, 30.8),
             ("s2", 1.0, 0.0, 0.0, 50.2), ("s3", 1.0, 0.0, 1.0, 30.6),
             ("s4", 1.0, 1.0, 0.0, 40.0), ("s5", 1.0, 0.0, 0.0, 80.5),
             ("s6", 1.0, 0.0, 0.0, 70.0), ("s7", 1.0, 1.0, 0.0, 87.4),
             ("s8", 1.0, 0.0, 0.0, 63.0), ("s9", 1.0, 0.0, 1.0, 54.3)],
            columns=["sample_id", "Intercept", "C(var2)[T.f2]",
                     "C(var2)[T.f3]", "var1"],
        ).set_index("sample_id")

        # The expected values for the new y and the new X
        self.expected_new_y = pd.DataFrame(
            [("s0", 3.0), ("s1", 3.4), ("s2", 5.3), ("s4", 0.5), ("s5", 2.4),
             ("s6", 5.6), ("s8", 0.3), ("s9", 1.9)],
            columns=["sample_id", "pheno"],
        ).set_index("sample_id")
        self.expected_new_X = pd.DataFrame(
            [("s0", 1.0, 0.0, 0.0, 12.0, 0.0),
             ("s1", 1.0, 1.0, 0.0, 30.8, 1.0),
             ("s2", 1.0, 0.0, 0.0, 50.2, 0.0),
             ("s4", 1.0, 1.0, 0.0, 40.0, 0.0),
             ("s5", 1.0, 0.0, 0.0, 80.5, 1.0),
             ("s6", 1.0, 0.0, 0.0, 70.0, 2.0),
             ("s8", 1.0, 0.0, 0.0, 63.0, 0.0),
             ("s9", 1.0, 0.0, 1.0, 54.3, 0.0)],
            columns=["sample_id", "Intercept", "C(var2)[T.f2]",
                     "C(var2)[T.f3]", "var1", "geno"],
        ).set_index("sample_id")

    def test_create_matrices(self):
        """Tests the 'create_matrices' function."""
        # Creating the matrices
        y, X = self.core_model.create_matrices(
            formula="pheno ~ var1 + C(var2)",
            data=self.data,
        )

        # Checking the results
        self.assertTrue(np.array_equal(y.index.values, X.index.values))
        self.assertTrue(self.expected_y.equals(y.sortlevel()))
        self.assertTrue(self.expected_X.equals(X.sortlevel()))

    def test_merge_matrices_genotypes(self):
        """Tests the 'merge_matrices_genotypes' function."""
        # Shuffling the rows
        y = self.expected_y.iloc[np.random.permutation(len(self.expected_y))]
        X = self.expected_X.iloc[np.random.permutation(len(self.expected_X))]

        # Merging the matrices with the genotypes
        new_y, new_X = self.core_model.merge_matrices_genotypes(
            y=y, X=X, genotypes=self.genotypes,
        )

        # Checking the results
        self.assertTrue(np.array_equal(new_y.index.values, new_X.index.values))
        self.assertTrue(self.expected_new_y.equals(new_y.sortlevel()))
        self.assertTrue(self.expected_new_X.equals(new_X.sortlevel()))
