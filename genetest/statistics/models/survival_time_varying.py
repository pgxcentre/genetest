"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import logging
import warnings

import numpy as np
import pandas as pd

from lifelines import CoxTimeVaryingFitter

from ..core import StatsModels, StatsError

__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsCoxTimeVarying"]


logger = logging.getLogger(__name__)


class StatsCoxTimeVarying(StatsModels):

    def __init__(self, include_likelihood=True, normalize=False):
        """Initializes a 'StatsCoxTimeVarying' instance.
        Args:	
            include_likelihood (bool): Whether to compute the log likelihood or
                                       not (might increase computation time).
            normalize (bool): Whether to normalize or not.	
        """
        self._incl_ll = include_likelihood
        self._normalize = normalize

    @staticmethod
    def _prepare_data(y, X):
        """Prepares the data for the CoxPH analysis."""
        # Checking y has tte and event columns
        missing_cols = {"tte", "event", "stop"} - set(y.columns)
        if missing_cols:
            raise ValueError(
                "missing column in y: CoxTimeVaryingFitter requires 'tte', 'stop' "
                "and 'event'"
            )
        if len(y.columns) > 4 and "strata" not in y.columns:
            extra = set(y.columns) - {"tte", "event", "stop"}
            logger.warning("{}: unknown column in y, will be ignored".format(
                ",".join(extra),
            ))

        # Removing the intercept (if present)
        if "intercept" in X.columns:
            X = X.drop("intercept", axis=1)

         # Creating one data frame
        return pd.merge(y[["tte", "stop", "event"]], X, left_index=True,
                        right_index=True)

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
        data = self._prepare_data(y, X)
        strata = None
        cluster = None
        robust = False

        if "strata" in y.columns:
            strata = y.strata.values
        # Clustering covariance is not yet available in Time varying survival
        # only robust estimator of variance is forced
        if "cluster" in y.columns:
            cluster = "cluster"
            robust =  True

        # Creating the CoxPH model and fitting it
        model = CoxTimeVaryingFitter(penalizer=0.1)

        # Fitting the model
        try:
            model.fit(data, start_col="tte", stop_col="stop",
                event_col="event", strata=strata, robust=robust)

        except np.linalg.linalg.LinAlgError as e:
            raise StatsError(str(e))

        # Getting the fitted results
        fitted = model.summary

        # Gathering the results
        out = {}
        parameters = fitted.index

        # Results about the model fit.
        if self._incl_ll:
            out = {
                "MODEL": {"log_likelihood": model.log_likelihood_,
                "nobs": data.shape[0],
                "nevents": np.sum(data.event),
                }
            }
        else:
            out = {
                "MODEL": {
                "nobs": data.shape[0],
                "nevents": np.sum(data.event),
                },
            }        
        # Getting the values for each parameters
        for param in parameters:
            # If GWAS, check that inference could be done on the SNP.
            if param == "SNPs" and np.isnan(fitted.loc[param, "p"]):
                raise StatsError("Inference did not converge.")

            out[param] = {
                "coef": fitted.loc[param, "coef"],
                "std_err": fitted.loc[param, "se(coef)"],
                "hr": fitted.loc[param, "exp(coef)"],
                "hr_lower_ci": fitted.loc[param, "exp(coef) lower 95%"],
                "hr_upper_ci": fitted.loc[param, "exp(coef) upper 95%"],
                "z_value": fitted.loc[param, "z"],
                "p_value": fitted.loc[param, "p"],
            }

        return out