"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from ...decorators import arguments
from ..core import StatsModels, StatsResults, StatsError


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsCoxPH"]


@arguments(required=(("time_to_event", str), ("event", str),
                     ("predictors", [str])),
           optional={"interaction": (str, None),
                     "normalize": (bool, False)})
class StatsCoxPH(StatsModels):
    def __init__(self, time_to_event, event, predictors, interaction,
                 normalize):
        """Initializes a 'StatsCoxPH' instance.

        Args:
            time_to_event (str): The name of the variable containing the time
                                 at which the event occurred.
            event (str): The name of the variable containing the event.
            predictors (list): The list of predictor variables in the model.
            interaction (str): The interaction variable to add to the model
                               with the genotype.
            normalize (bool): Whether to normalize or not.

        """
        # Creating the model
        self._create_model(outcomes=[time_to_event, event],
                           predictors=predictors, interaction=interaction,
                           intercept=True)

        self.results = StatsResults(
            coef="Cox proportional hazard regression coefficient",
            std_err="Standard error of the regression coefficient",
            hr="Hazard ratio",
            hr_lower_ci="Lower 95% confidence interval of the hazard ratio",
            hr_upper_ci="Upper 95% confidence interval of the hazard ratio",
            z_value="z-statistics",
            p_value="p-value",
            print_order=["coef", "std_err", "hr", "hr_lower_ci", "hr_upper_ci",
                         "z_value", "p_value"]
        )

        # Saving the two variables for time to event and event
        self._tte = time_to_event
        self._event = event

        # Saving the normalization status
        self._normalize = normalize

        # Executing the super init class
        super().__init__(outcomes=[time_to_event, event],
                         predictors=predictors,
                         interaction=interaction, intercept=True)

    def fit(self, y, X):
        """Fit the model.

        Args:
            y (pandas.DataFrame): The vector of endogenous variable.
            X (pandas.DataFrame): The matrix of exogenous variables.

        """
        # Resetting the statistics
        self.results.reset()

        # Creating one data frame
        data = pd.merge(y, X, left_index=True, right_index=True)

        # Creating the model
        model = CoxPHFitter(normalize=self._normalize)

        # Fitting the model
        try:
            model.fit(data, self._tte, event_col=self._event)

        except np.linalg.linalg.LinAlgError as e:
            raise StatsError(str(e))

        # Gathering the results for the required column
        results = model.summary.loc[self._result_col, :]

        # Saving the statistics
        self.results.coef = results.coef
        self.results.std_err = results["se(coef)"]
        self.results.hr = np.exp(results.coef)
        self.results.hr_lower_ci = np.exp(results["lower 0.95"])
        self.results.hr_upper_ci = np.exp(results["upper 0.95"])
        self.results.z_value = results.z
        self.results.p_value = results.p

    def create_matrices(self, data, create_dummy=True):
        """Creates the y and X matrices for a linear regression.

        Args:
            data (genetest.phenotypes.core.PhenotypesContainer): The data.
            create_dummy (bool): If True, a dummy column will be added for the
                                 genotypes.

        Returns:
            tuple: y and X as pandas dataframes (according to the formula).

        Note
        ----
            This method calls the super method, but remove the 'Intercept'
            column (since it's not required by lifelines).

        """
        y, X = super().create_matrices(data, create_dummy=create_dummy)

        # Removing the 'Intercept' column
        if 'Intercept' in X.columns:
            X = X.drop("Intercept", axis=1)

        return y, X
