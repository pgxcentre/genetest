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
from lifelines import CoxPHFitter

from ..core import StatsModels, StatsResults, StatsError


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsCoxPH"]


class StatsCoxPH(StatsModels):
    def __init__(self):
        """Initializes a 'StatsCoxPH' instance."""
        self.results = StatsResults(
            coef="Cox proportional hazard regression coefficient",
            std_err="Standard error of the regression coefficient",
            hr="Hazard ratio",
            hr_lower_ci="Lower 95% confidence interval of the hazard ratio",
            hr_upper_ci="Upper 95% confidence interval of the hazard ratio",
            z_value="z-statistics",
            p_value="p-value",
        )

    def fit(self, y, X, tte, event, result_col, normalize=False):
        """Fit the model.

        Args:
            y (pandas.DataFrame): The vector of endogenous variable.
            X (pandas.DataFrame): The matrix of exogenous variables.
            tte (str): The name of the column containing the time to event.
            event (str): The name of the column containing the event.
            result_col (str): The variable for which the results are required.
            normalize (bool): Is normalization is required? (default is False).

        """
        # Resetting the statistics
        self.results.reset()

        # Creating one data frame
        data = pd.merge(y, X, left_index=True, right_index=True)

        # Creating the model
        model = CoxPHFitter(normalize=normalize)

        # Fitting the model
        try:
            model.fit(data, tte, event_col=event)
        except np.linalg.linalg.LinAlgError as e:
            raise StatsError(str(e))

        # Gathering the results for the required column
        results = model.summary.loc[result_col, :]

        # Saving the statistics
        self.results.coef = results.coef
        self.results.std_err = results["se(coef)"]
        self.results.hr = np.exp(results.coef)
        self.results.hr_lower_ci = np.exp(results["lower 0.95"])
        self.results.hr_upper_ci = np.exp(results["upper 0.95"])
        self.results.z_value = results.z
        self.results.p_value = results.p
