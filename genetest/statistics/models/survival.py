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

import statsmodels.api as sm

from ..core import StatsModels, StatsError


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsCoxPH"]


logger = logging.getLogger(__name__)


class StatsCoxPH(StatsModels):
    @staticmethod
    def _prepare_data(y, X):
        """Prepares the data for the CoxPH analysis."""
        # Checking y has tte and event columns
        missing_cols = {"tte", "event"} - set(y.columns)
        if missing_cols:
            raise ValueError(
                "missing column in y: coxph requires 'tte' and 'event'"
            )
        if len(y.columns) > 2:
            extra = set(y.columns) - {"tte", "event"}
            logger.warning("{}: unknown column in y, will be ignored".format(
                ",".join(extra),
            ))

        # Removing the intercept (if present)
        if "intercept" in X.columns:
            X = X.drop("intercept", axis=1)

        return y.tte, y.event, X

    def fit(self, y, X):
        """Fit the model.

        Args:
            y (pandas.DataFrame): The vector of endogenous variable.
            X (pandas.DataFrame): The matrix of exogenous variables.

        Warning
        =======
            The y dataframe (outcome) should contain two columns: tte and
            event.

        """
        # Retrieving the data
        tte, event, X = self._prepare_data(y, X)

        # Creating the CoxPH model and fitting it
        model = sm.PHReg(tte, X, status=event, ties="efron")
        try:
            fitted = model.fit()
        except np.linalg.linalg.LinAlgError as e:
            raise StatsError(str(e))

        # Getting the results
        out = {}
        parameters = model.exog_names

        # Results about the model fit
        out = {
            "MODEL": {
                "log_likelihood": fitted.llf,
                "nobs": X.shape[0],
            },
        }

        # Computing the confidence intervals only once
        conf_ints = fitted.conf_int()

        # Getting the values for each parameters
        for i, param in enumerate(parameters):
            # If GWAS, check that inference could be done on the SNP
            if param == "SNPs" and np.isnan(fitted.pvalues[i]):
                raise StatsError("Inference did not converge.")

            out[param] = {
                "coef": fitted.params[i],
                "std_err": fitted.bse[i],
                "hr": np.exp(fitted.params[i]),
                "hr_lower_ci": np.exp(conf_ints[i, 0]),
                "hr_upper_ci": np.exp(conf_ints[i, 1]),
                "z_value": fitted.tvalues[i],
                "p_value": fitted.pvalues[i],
            }

        return out
