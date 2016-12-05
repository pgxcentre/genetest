"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import statsmodels.api as sm

from ...decorators import arguments
from ..core import StatsModels, StatsResults, StatsError


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsLinear"]


@arguments(required=(("outcome", str), ("predictors", [str])),
           optional={"interaction": (str, None),
                     "condition_value_t": (int, 1000)})
class StatsLinear(StatsModels):
    def __init__(self, outcome, predictors, interaction, condition_value_t):
        """Initializes a 'StatsLinear' instance.

        Args:
            outcome (str): The outcome of the model.
            predictors (list): The list of predictor variables in the model.
            interaction (str): The interaction variable to add to the model
                               with the genotype.
            condition_value_t (int): The condition value threshold (for
                                     multicollinearity).

        """
        # Creating the result object
        self.results = StatsResults(
            coef="Linear regression coefficient",
            std_err="Standard error of the regression coefficient",
            lower_ci="Lower 95% confidence interval",
            upper_ci="Upper 95% confidence interval",
            t_value="t-statistics",
            p_value="p-value",
            rsquared_adj="adjusted r-squared",
            print_order=["coef", "std_err", "lower_ci", "upper_ci", "t_value",
                         "p_value", "rsquared_adj"]
        )

        # Saving the condition value threshold
        self._condition_value_t = condition_value_t

        # Executing the super init class
        super().__init__(outcomes=[outcome], predictors=predictors,
                         interaction=interaction, intercept=True)

    def fit(self, y, X):
        """Fit the model.

        Args:
            y (pandas.DataFrame): The vector of endogenous variable.
            X (pandas.DataFrame): The matrix of exogenous variables.

        """
        # Resetting the statistics
        self.results.reset()

        # Creating the OLS model from StatsModels and fitting it
        model = sm.OLS(y, X)
        fitted = model.fit()

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

        # Saving the statistics
        self.results.coef = fitted.params[self._result_col]
        self.results.std_err = fitted.bse[self._result_col]
        self.results.lower_ci, self.results.upper_ci = tuple(
            fitted.conf_int().loc[self._result_col, :].values
        )
        self.results.t_value = fitted.tvalues[self._result_col]
        self.results.p_value = fitted.pvalues[self._result_col]
        self.results.rsquared_adj = fitted.rsquared_adj
