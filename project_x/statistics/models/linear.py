"""
"""


# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


from patsy import dmatrices
import statsmodels.api as sm

from ..core import StatsModels, StatsResults, StatsError


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsLinear"]


class StatsLinear(StatsModels):
    def __init__(self):
        """Initializes a 'StatsLinear' instance."""
        self.results = StatsResults(
            coef="Linear regression coefficient",
            std_err="Standard error of the regression coefficient",
            lower_ci="Lower 95% confidence interval",
            upper_ci="Upper 95% confidence interval",
            t_value="t-statistics",
            p_value="p-value",
            rsquared_adj="adjusted r-squared",
        )

    def fit(self, y, X, result_col, condition_value_t=1000):
        """Fit the model.

        Args:
            y (pandas.DataFrame): The vector of endogenous variable.
            X (pandas.DataFrame): The matrix of exogenous variables.
            result_col (str): The variable for which the results are required.
            condition_value_t (int): The condition value threshold (for
                                     multicollinearity).

        """
        # Resetting the statistics
        self.results.reset()

        # Creating the OLS model from StatsModels and fitting it
        model = sm.OLS(y, X)
        fitted = model.fit()

        # Checking the condition number (according to StatsModels, condition
        # number higher than 1000 indicate that there are strong
        # multicollinearity or other numerical problems)
        if fitted.condition_number > condition_value_t:
            raise StatsError("condition number is large, {}".format(
                fitted.condition_number,
            ))

        # Checking the smallest eigenvalue (according to StatsModels, values
        # loser than 1e-10 might indicate that there are strong
        # multicollinearity problems or that the design matrix is singular)
        if fitted.eigenvals.min() < 1e-10:
            raise StatsError("smallest eigenvalue is small, {}".format(
                fitted.eigenvals.min(),
            ))

        # Saving the statistics
        self.results.coef = fitted.params[result_col]
        self.results.std_err = fitted.bse[result_col]
        self.results.lower_ci, self.results.upper_ci = tuple(
            fitted.conf_int().loc[result_col, :].values
        )
        self.results.t_value = fitted.tvalues[result_col]
        self.results.p_value = fitted.pvalues[result_col]
        self.results.rsquared_adj = fitted.rsquared_adj

    def from_formula(self, formula, data, result_col, condition_value_t=1000):
        """Fit the model using a formula.

        Args:
            formula (str): The formula explaining the model.
            data (pandas.DataFrame): The data to fit.
            result_col (str): The variable for which the results are required.
            condition_value_t (int): The condition value threshold (for
                                     multicollinearity).

        """
        y, X = dmatrices(formula, data, return_type="dataframe")
        self.fit(y, X, result_col, condition_value_t)
