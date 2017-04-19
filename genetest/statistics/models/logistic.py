"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from ..core import StatsModels, StatsError

import numpy as np


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsLogistic"]


class StatsLogistic(StatsModels):
    def fit(self, y, X):
        """Fit the model.

        Args:
            y (pandas.DataFrame): The vector of endogenous variable.
            X (pandas.DataFrame): The matrix of exogenous variables.

        """
        # Creating the GLM model from StatsModels and fitting it
        model = sm.GLM(y, X, family=sm.families.Binomial())
        try:
            fitted = model.fit()

        except PerfectSeparationError as e:
            raise StatsError(str(e))

        out = {}
        parameters = fitted.params.index

        # Results about the model fit.
        out = {
            "MODEL": {
                "log_likelihood": fitted.llf,
                "nobs": fitted.nobs
            },
        }

        # Getting the confidence intervals
        conf_ints = fitted.conf_int()

        for param in parameters:
            # If GWAS, check that inference could be done on the SNP.
            if param == "SNPs" and np.isnan(fitted.pvalues[param]):
                raise StatsError("Inference did not converge.")

            out[param] = {
                "coef": fitted.params[param],
                "std_err": fitted.bse[param],
                "lower_ci": conf_ints.loc[param, 0],
                "upper_ci": conf_ints.loc[param, 1],
                "t_value": fitted.tvalues[param],
                "p_value": fitted.pvalues[param],
            }

        return out
