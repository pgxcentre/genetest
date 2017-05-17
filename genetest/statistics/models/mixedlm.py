"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..core import StatsModels, StatsError


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsMixedLM"]


logger = logging.getLogger(__name__)


class StatsMixedLM(StatsModels):
    def __init__(self, reml=True):
        """Initializes a 'StatsMixedLM' instance.

        Args:
            reml (bool): Whether to use REML or ML for the test.

        """
        self._reml = reml

    @staticmethod
    def _prepare_data(y, X):
        """Prepares the data for the MixedLM analysis."""
        # Checking y as the outcome and the groups
        missing_cols = {"outcome", "groups"} - set(y.columns)
        if missing_cols:
            raise ValueError(
                "missing column in y: mixedlm requires 'outcome' and 'groups'"
            )
        if len(y.columns) > 2:
            extra = set(y.columns) - {"outcome", "groups"}
            logger.warning("{}: unknown column in y, will be ignored".format(
                ",".join(extra),
            ))

        return y[["outcome"]], X, y.groups

    @staticmethod
    def _format_re(re):
        """Formats the random effects."""
        # Statsmodels returns a dictionary for the random effects
        return pd.Series(
            {sample: value.iloc[0] for sample, value in re.items()}
        )

    def fit(self, y, X):
        """Fit the model.

        Args:
            y (pandas.DataFrame): The vector of endogenous variable.
            X (pandas.DataFrame): The matrix of exogenous variables.

        """
        # Retrieving the data
        y, X, groups = self._prepare_data(y, X)

        # Creating the MixedLM model from StatsModels and fitting it
        model = sm.MixedLM(y, X, groups)
        try:
            # fitted = model.fit(reml=self._reml)
            fitted = model.fit(reml=self._reml)

        except np.linalg.linalg.LinAlgError as e:
            raise StatsError(str(e))

        out = {}
        parameters = fitted.params.index

        # Results about the model fit
        out = {
            "MODEL": {
                "log_likelihood": fitted.llf,
                "nobs": X.shape[0],
                "random_effects": self._format_re(fitted.random_effects),
            },
        }

        # Getting the confidence intervals
        conf_ints = fitted.conf_int()

        for param in parameters:
            # If GWAS, check that inference could be done on the SNP
            if param == "SNPs" and np.isnan(fitted.pvalues[param]):
                raise StatsError("Inference did not converge.")

            out[param] = {
                "coef": fitted.params[param],
                "std_err": fitted.bse[param],
                "lower_ci": conf_ints.loc[param, 0],
                "upper_ci": conf_ints.loc[param, 1],
                "z_value": fitted.tvalues[param],
                "p_value": fitted.pvalues[param],
            }

        return out
