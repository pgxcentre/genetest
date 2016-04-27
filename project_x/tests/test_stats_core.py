

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
        self.data1 = pd.DataFrame(
            [("s0", 3.0, 12.0, "f1"), ("s1", 3.4, 30.8, "f2"),
             ("s2", 5.3, 50.2, "f1"), ("s3", 6.0, 30.6, "f3"),
             ("s4", 0.5, 40.0, "f2"), ("s5", 2.4, 80.5, "f1"),
             ("s6", 5.6, 70.0, "f1"), ("s7", 7.6, 87.4, "f2"),
             ("s8", 0.3, 63.0, "f1"), ("s9", 1.9, 54.3, "f3")],
            columns=["SampleID", "pheno", "var1", "var2"],
        ).set_index("SampleID")

        # The expected values
        self.expected_y1 = pd.DataFrame(
            [("s0", 3.0), ("s1", 3.4), ("s2", 5.3), ("s3", 6.0), ("s4", 0.5),
             ("s5", 2.4), ("s6", 5.6), ("s7", 7.6), ("s8", 0.3), ("s9", 1.9)],
            columns=["SampleID", "pheno"],
        ).set_index("SampleID")
        self.expected_X1 = pd.DataFrame(
            [("s0", 1.0, 0.0, 0.0, 12.0), ("s1", 1.0, 1.0, 0.0, 30.8),
             ("s2", 1.0, 0.0, 0.0, 50.2), ("s3", 1.0, 0.0, 1.0, 30.6),
             ("s4", 1.0, 1.0, 0.0, 40.0), ("s5", 1.0, 0.0, 0.0, 80.5),
             ("s6", 1.0, 0.0, 0.0, 70.0), ("s7", 1.0, 1.0, 0.0, 87.4),
             ("s8", 1.0, 0.0, 0.0, 63.0), ("s9", 1.0, 0.0, 1.0, 54.3)],
            columns=["SampleID", "Intercept", "C(var2)[T.f2]",
                     "C(var2)[T.f3]", "var1"],
        ).set_index("SampleID")

    def test_create_matrices(self):
        """Tests the 'create_matrices' function."""
        # Creating the matrices
        y, X = self.core_model.create_matrices(
            formula="pheno ~ var1 + C(var2)",
            data=self.data1,
        )

        # Checking the results
        self.assertTrue(np.array_equal(y.index.values, X.index.values))
        self.assertTrue(self.expected_y1.equals(y))
        self.assertTrue(self.expected_X1.equals(X))
