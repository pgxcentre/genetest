"""
"""


# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from ...decorators import arguments
from ..core import StatsModels, StatsResults, StatsError


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsLogistic"]


@arguments(required=("outcome", "predictors"),
           optional={"interaction": None})
class StatsLogistic(StatsModels):
    def __init__(self, outcome, predictors, interaction):
        """Initializes a 'StatsLogistic' instance.

        Args:
            outcome (str): The outcome of the model.
            predictors (list): The list of predictor variables in the model.
            interaction (list): The list of interaction variable to add to the
                                model with the genotype.

        """
        # Creating the result object
        self.results = StatsResults(
            coef="Logistic regression coefficient",
            std_err="Standard error of the regression coefficient",
            lower_ci="Lower 95% confidence interval",
            upper_ci="Upper 95% confidence interval",
            z_value="z-statistics",
            p_value="p-value",
        )

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
        model = sm.GLM(y, X, family=sm.families.Binomial())
        try:
            fitted = model.fit()
        except PerfectSeparationError as e:
            raise StatsError(str(e))

        # Saving the statistics
        self.results.coef = fitted.params[self._result_col]
        self.results.std_err = fitted.bse[self._result_col]
        self.results.lower_ci, self.results.upper_ci = tuple(
            fitted.conf_int().loc[self._result_col, :].values
        )
        self.results.z_value = fitted.tvalues[self._result_col]
        self.results.p_value = fitted.pvalues[self._result_col]
