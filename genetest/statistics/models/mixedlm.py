"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import re

import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ...decorators import arguments
from ..core import StatsModels, StatsResults, StatsError


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsMixedLM"]


@arguments(required=(("outcome", str), ("predictors", [str])),
           optional={"interaction": (str, None),
                     "reml": (bool, True),
                     "p_threshold": (float, 1e-4)})
class StatsMixedLM(StatsModels):
    def __init__(self, outcome, predictors, interaction, reml, p_threshold):
        """Initializes a 'StatsMixedLM' instance.

        Args:
            outcome (str): The outcome of the model.
            predictors (list): The list of predictor variables in the model.
            interaction (str): The interaction variable to add to the model
                               with the genotype.
            reml (bool): Whether to use REML or ML for the test.
            p_threshold (float): The p-value threshold for which the real
                                 MixedLM will be computed (instead of the
                                 approximation).

        """
        # Creating the result object
        self.results = StatsResults(
            coef="MixedLM regression coefficient",
            std_err="Standard error of the regression coefficient",
            lower_ci="Lower 95% confidence interval",
            upper_ci="Upper 95% confidence interval",
            z_value="z-statistics",
            p_value="p-value",
            ts_p_value="Two-step p-value",
            print_order=["coef", "std_err", "lower_ci", "upper_ci", "z_value",
                         "p_value", "ts_p_value"]
        )

        # The original samples and the grouping
        self._ori_samples = None
        self._groups = None

        # Saving the REML boolean
        self._reml = reml

        # Creating a variable to save the estimated random effects (for the
        # mixedlm optimization called two-step LMM)
        self._re = None

        # Saving the p-value threshold
        self._p_threshold = p_threshold

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

        # We perform the optimization if there is no interaction
        if not self._inter:
            # Creating the matrices (for random effects and genotypes)
            t_y, t_X = patsy.dmatrices(
                "RE ~ geno",
                pd.merge(self._re, self._original_genotypes, left_index=True,
                         right_index=True),
                return_type="dataframe",
            )

            # Fitting the linear model
            fitted = sm.OLS(t_y, t_X).fit()

            # We keep only the p value
            self.results.ts_p_value = fitted.pvalues["geno"]

            # Should we perform the standard LMM?
            if self.results.ts_p_value >= self._p_threshold:
                return

        # Creating the OLS model from StatsModels and fitting it
        model = sm.MixedLM(y, X, self._groups)

        try:
            fitted = model.fit(reml=self._reml)

        except np.linalg.linalg.LinAlgError as e:
            raise StatsError(str(e))

        # Saving the statistics
        self.results.coef = fitted.params[self._result_col]
        self.results.std_err = fitted.bse[self._result_col]
        self.results.lower_ci, self.results.upper_ci = tuple(
            fitted.conf_int().loc[self._result_col, :].values
        )
        self.results.z_value = fitted.tvalues[self._result_col]
        self.results.p_value = fitted.pvalues[self._result_col]

    def create_matrices(self, data, create_dummy=True):
        """Creates the y and X matrices for a mixedlm analysis.

        Args:
            data (genetest.phenotypes.core.PhenotypesContainer): The data.
            create_dummy (bool): If True, a dummy column will be added for the
                                 genotypes.

        Returns:
            tuple: y and X as pandas dataframes (according to the formula).

        For the optimization, we required to run the MixedLM on the phenotypes
        only (i.e. no genotypes) and to save the estimated random slopes (as
        describe in Sikorska et al. 2015 doi:10.1038/ejhg.2015.1).

        """
        # Saving the original samples
        self._ori_samples = data.get_original_sample_names()

        # First, we get the matrix
        y, X = super().create_matrices(data=data, create_dummy=create_dummy)

        # if create_dummy is False, it means the genotypes are in the dataframe
        if not create_dummy:
            # Creating the genotypes array
            self._original_genotypes = pd.merge(
                X.loc[:, ["geno"]], self._ori_samples, left_index=True,
                right_index=True,
            ).drop_duplicates(subset="_ori_sample_names_").set_index(
                "_ori_sample_names_",
            )

        # We merge the y and X matrices
        full_data = pd.merge(data.get_phenotypes().dropna(), self._ori_samples,
                             left_index=True, right_index=True)

        # We get the original samples
        self._groups = full_data._ori_sample_names_

        # TODO: Check if this is true
        # If we have interaction, the optimization doesn't old
        if self._inter is not None:
            return y, X

        # FIXME: This is a quick fix, because of an error in statsmodels. When
        # fitting a MixedLM from y and X (instead of from formula), we get the
        # following AttributeError: 'PandasData' object has no attribute
        # 'exog_re_names'. We need to fit using a formula (after removing the
        # geno term).
        formula = re.sub(r"geno( \+ )?", "", self.get_model_description())
        if formula.endswith(" + "):
            formula = formula[:-3]

        # Fitting the model
        model = sm.MixedLM.from_formula(formula, full_data,
                                        groups=self._groups)
        fitted = model.fit(reml=self._reml)

        # Extracting the random effects
        self._re = fitted.random_effects.rename(columns={"Intercept": "RE"})

        # Returning the matrices
        return y, X

    def merge_matrices_genotypes(self, y, X, genotypes,
                                 compute_interaction=True):
        """Merges the genotypes to X, remove missing values, and subset y.

        Args:
            y (pandas.DataFrame): The y dataframe.
            X (pandas.DataFrame): The X dataframe.
            genotypes (pandas.DataFrame): The genotypes dataframe.
            compute_interaction (bool): If True, interaction will be computed
                                        with the genotype.

        Returns:
            tuple: The y and X dataframes (with the genotypes merged).

        """
        # We save the original genotypes
        self._original_genotypes = genotypes

        # Merging the old sample names and the genotypes
        new_X = pd.merge(
            pd.merge(X, self._ori_samples, left_index=True, right_index=True),
            genotypes,
            left_on="_ori_sample_names_",
            right_index=True,
        ).dropna()

        # Saving the groups
        self._groups = new_X._ori_sample_names_

        # Keeping only the required X values
        new_y = y.loc[new_X.index, :]

        # Check if there is interaction
        if compute_interaction and self._inter is not None:
            # There is, so we multiply
            new_X[self._result_col] = new_X.geno * new_X[self._inter_col]

        return new_y, new_X.drop("_ori_sample_names_", axis=1)
