"""
"""


# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import numpy as np
import pandas as pd
import statsmodels.api as sm

from ...decorators import arguments
from ..core import StatsModels, StatsResults, StatsError


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsMixedLM"]


@arguments(required=(("outcome", str), ("predictors", [str])),
           optional={"interaction": (str, None),
                     "reml": (bool, True)})
class StatsMixedLM(StatsModels):
    def __init__(self, outcome, predictors, interaction, reml):
        """Initializes a 'StatsMixedLM' instance."""
        # Creating the result object
        self.results = StatsResults(
            coef="MixedLM regression coefficient",
            std_err="Standard error of the regression coefficient",
            lower_ci="Lower 95% confidence interval",
            upper_ci="Upper 95% confidence interval",
            z_value="z-statistics",
            p_value="p-value",
            print_order=["coef", "std_err", "lower_ci", "upper_ci", "z_value",
                         "p_value"]
        )

        # Saving the REML boolean
        self._reml = reml

        # Executing the super init class
        super().__init__(outcomes=[outcome], predictors=predictors,
                         interaction=interaction, intercept=True)

    def fit(self, y, X, groups):
        """Fit the model.

        Args:
            y (pandas.DataFrame): The vector of endogenous variable.
            X (pandas.DataFrame): The matrix of exogenous variables.
            groups (numpy.ndarray): The samples (for repeated measurements).

        """
        # Resetting the statistics
        self.results.reset()

        # Creating the OLS model from StatsModels and fitting it
        model = sm.MixedLM(y, X, groups)

        try:
            fitted = model.fit(reml=self._reml)
        except np.linalg.linalg.LinAlgError as e:
            raise StatsError(str(e))

        # Saving the statistics
        self.results.coef = fitted.params[self._result_col]
        self.results.std_err = fitted.bse[self._result_col]
        self.results.lower_ci, self.results.upper_ci = tuple(
            fitted.conf_int().loc[self._result_col, :].values
        )
        self.results.z_value = fitted.tvalues[self._result_col]
        self.results.p_value = fitted.pvalues[self._result_col]

    def merge_matrices_genotypes(self, y, X, genotypes, ori_samples,
                                 compute_interaction=True):
        """Merges the genotypes to X, remove missing values, and subset y.

        Args:
            y (pandas.DataFrame): The y dataframe.
            X (pandas.DataFrame): The X dataframe.
            genotypes (pandas.DataFrame): The genotypes dataframe.
            ori_samples (pandas.DataFrame): The original sample names.
            compute_interaction (bool): If True, interaction will be computed
                                        with the genotype.

        Returns:
            tuple: The y and X dataframes (with the genotypes merged) and the
                   groups for the model (the sample names) as numpy.ndarray.

        """
        # Merging the old sample names and the genotypes
        new_X = pd.merge(
            left=pd.merge(X, ori_samples, left_index=True, right_index=True),
            right=genotypes,
            left_on="_ori_sample_names_",
            right_index=True,
        ).dropna().drop("_ori_sample_names_", axis=1)

        # Keeping only the required X values
        new_y = y.loc[new_X.index, :]

        # Check if there is interaction
        if compute_interaction and self._inter is not None:
            # There is, so we multiply
            new_X[self._result_col] = new_X.geno * new_X[self._inter_col]

        # Getting the sample order
        groups = ori_samples.loc[new_X.index, "_ori_sample_names_"].values

        return new_y, new_X, groups
