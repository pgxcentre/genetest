"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import statsmodels.api as sm

from ..core import StatsModels, StatsError


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsLinear"]


class StatsLinear(StatsModels):
    def __init__(self, condition_value_t=1000):
        """Initializes a 'StatsLinear' instance.

        Args:
            condition_value_t (int): The condition value threshold (for
                                     multicollinearity).

        """
        # Saving the condition value threshold
        self._condition_value_t = condition_value_t

    def _results_handler(self, fitted):
        # Checking the condition number (according to StatsModels, condition
        # number higher than 1000 indicate that there are strong
        # multicollinearity or other numerical problems)
        if fitted.condition_number > self._condition_value_t:
            raise StatsError("condition number is large, {}".format(
                fitted.condition_number,
            ))

        # Checking the smallest eigenvalue (according to StatsModels, values
        # lower than 1e-10 might indicate that there are strong
        # multicollinearity problems or that the design matrix is singular)
        if fitted.eigenvals.min() < 1e-10:
            raise StatsError("smallest eigenvalue is small, {}".format(
                fitted.eigenvals.min(),
            ))

        # Results about the model fit.
        out = {
            "MODEL": {
                "r_squared_adj": fitted.rsquared_adj,
                "log_likelihood": fitted.llf,
                "nobs": fitted.nobs
            }
        }

        # Results about individual model parameters.
        parameters = fitted.params.index
        for param in parameters:
            out[param] = {
                "coef": fitted.params[param],
                "std_err": fitted.bse[param],
                "lower_ci": fitted.conf_int().loc[param, 0],
                "upper_ci": fitted.conf_int().loc[param, 1],
                "t_value": fitted.tvalues[param],
                "p_value": fitted.pvalues[param],
            }

        return out

    def fit(self, y, X, handler=None):
        """Fit the model.

        Args:
            y (pandas.DataFrame): The vector of endogenous variable.
            X (pandas.DataFrame): The matrix of exogenous variables.

        """
        # Creating the OLS model from StatsModels and fitting it
        model = sm.OLS(y, X)
        handler = self._results_handler if handler is None else handler
        return handler(model.fit())
