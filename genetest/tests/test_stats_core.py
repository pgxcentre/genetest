

# This file is part of genetest.
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


@unittest.skip("Not implemented")
class TestStatsLinear(unittest.TestCase):
    """Tests the 'StatsLinear' class."""
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

        # Creating an instance of StatsModels core object
        self.core_model = StatsModels(
            outcomes=["test"],
            predictors=["var1", "var2", "var3"],
            interaction=None,
            intercept=True,
        )

    def test__create_model(self):
        """Tests the '_create_model' function."""
        self.core_model._create_model(
            outcomes=["y"],
            predictors=["var1", "C(var2)"],
            interaction=None,
            intercept=True,
        )

        # Checking the model (by looking at the formula)
        self.assertEqual(
            "y ~ geno + var1 + C(var2)",
            self.core_model.get_model_description(),
        )

        # Checking the resulting column
        self.assertEqual("geno", self.core_model._result_col)

    def test__create_model_no_intercept(self):
        """Tests the '_create_model' function with no intercept."""
        self.core_model._create_model(
            outcomes=["y"],
            predictors=["var1", "C(var2)"],
            interaction=None,
            intercept=False,
        )

        # Checking the model (by looking at the formula)
        self.assertEqual(
            "y ~ 0 + geno + var1 + C(var2)",
            self.core_model.get_model_description(),
        )

        # Checking the resulting column
        self.assertEqual("geno", self.core_model._result_col)

    def test__create_model_with_interaction(self):
        """Tests the '_create_model' function with an interaction."""
        self.core_model._create_model(
            outcomes=["y"],
            predictors=["var1", "C(var2)"],
            interaction="var1",
            intercept=False,
        )

        # Checking the model (by looking at the formula)
        self.assertEqual(
            "y ~ 0 + geno + var1 + C(var2) + geno:var1",
            self.core_model.get_model_description(),
        )

        # Checking the resulting column
        self.assertEqual("geno:var1", self.core_model._result_col)

    def test__create_model_with_interaction2(self):
        """Tests with an interaction (categorical)."""
        self.core_model._create_model(
            outcomes=["y"],
            predictors=["var1", "C(var2)"],
            interaction="C(var2)",
            intercept=True,
        )

        # Checking the model (by looking at the formula)
        self.assertEqual(
            "y ~ geno + var1 + C(var2) + geno:C(var2)",
            self.core_model.get_model_description(),
        )

        # Checking the resulting column
        self.assertEqual("geno:C(var2)", self.core_model._result_col)

    def test__create_model_with_interaction_not_in_predictors(self):
        """Tests with an interaction but not in predictors list."""
        self.core_model._create_model(
            outcomes=["y"],
            predictors=["var1", "C(var2)"],
            interaction="var3",
            intercept=True,
        )

        # Checking the model (by looking at the formula)
        self.assertEqual(
            "y ~ geno + var1 + C(var2) + var3 + geno:var3",
            self.core_model.get_model_description(),
        )

        # Checking the resulting column
        self.assertEqual("geno:var3", self.core_model._result_col)
