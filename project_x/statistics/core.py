"""
"""


# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


from collections import namedtuple

import patsy
import numpy as np
import pandas as pd

from .descriptive import get_freq


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsModels", "StatsResults", "StatsError",
           "statistics_initializer", "statistics_worker"]


# The statistical results
_StatsResults = namedtuple(
    "_StatsResults",
    ["chr", "pos", "snp", "major", "minor", "maf", "n", "stats_n", "stats"],
)


def statistics_initializer(model, arguments, phenotypes):
    """A statistics initializer used by the statistics workers.

    Args:
        model (StatsModels): The statistical model to perform.
        arguments (dict): The arguments of the statistical model.
        phenotypes (PhenotypesContainer): The phenotypes.

    This function initialize the statistics worker.

    """
    # Initializing the model
    statistics_worker.stats = model(**arguments)

    # Creating the matrices
    matrices = statistics_worker.stats.create_matrices(phenotypes)
    statistics_worker.y, statistics_worker.X = matrices

    # Getting the statistics names
    names = statistics_worker.stats.results.get_statistic_names()
    statistics_worker.stats_names = names


def statistics_worker(marker):
    """A statistics worker that will run in its own process.

    Args:
        genotypes (GenotypesContainer): The genotypes.

    """
    y, X = statistics_worker.stats.merge_matrices_genotypes(
        y=statistics_worker.y, X=statistics_worker.X,
        genotypes=marker.genotypes,
    )

    # Computing the MAF
    maf = get_freq(X.geno)

    # Checking it's a MAF
    minor = marker.minor
    major = marker.major
    if (not np.isnan(maf)) and maf > 0.5:
        maf = 1 - maf
        X.geno = 2 - X.geno
        minor, major = major, minor

    if np.isnan(maf) or maf < 0.01:
        return _StatsResults(
            marker.chrom, marker.pos, marker.marker, major, minor, maf,
            X.shape[0], statistics_worker.stats_names,
            tuple([np.nan] * len(statistics_worker.stats_names)),
        )

    # Performing the fitting
    try:
        statistics_worker.stats.fit(y, X)
    except StatsError as e:
        pass
    except ValueError as e:
        pass

    # Returning the results
    return _StatsResults(
        marker.chrom, marker.pos, marker.marker, major, minor, maf, X.shape[0],
        statistics_worker.stats_names,
        tuple(statistics_worker.stats.results.get_statistics()),
    )


class StatsModels(object):
    def __init__(self, outcomes, predictors, interaction, intercept):
        """Initializes a 'StatsLinear' instance.

        Args:
            outcome (str): The outcome of the model.
            predictors (list): The list of predictor variables in the model.
            interaction (list): The list of interaction variable to add to the
                                model with the genotype.

        """
        # Creating the model
        self._create_model(outcomes=outcomes, predictors=predictors,
                           interaction=interaction, intercept=intercept)

        # Saving the interaction term
        self._inter = interaction

    def fit(self, y, X, result_col):
        """Fit the model.

        Args:
            y (pandas.DataFrame): The vector of endogenous variable.
            X (pandas.DataFrame): The matrix of exogenous variables.
            result_col (str): The variable for which the results are required.

        """
        raise NotImplementedError()

    @classmethod
    def get_required_arguments(cls):
        """Returns the required arguments.

        Returns:
            tuple: The required arguments of the genotype container.

        """
        return cls._required_args

    @classmethod
    def get_optional_arguments(cls):
        """Returns the optional arguments.

        Returns:
            dict: The optional arguments (with default values) of the genotype
            container.

        """
        return cls._optional_args

    @classmethod
    def get_arguments_type(cls):
        """Returns the arguments type.

        Returns:
            dict: The type of each arguments (both required and optional).

        """
        return cls._args_type

    def _create_model(self, outcomes, predictors, interaction=None,
                      intercept=True):
        """Creates a statistical model.

        Args:
            outcomes (list): The list of response variables (the left hand side
                             terms in the model formula).
            predictors (list): The list of predictor variables (the right hand
                               side terms in the model formula).
            interaction (str): The interactions to include in the model (with
                               the genotypes).
            intercept (bool): If True, an intercept will be added to the model.

        Returns:
            patsy.desc.ModelDesc: The description of the statistical model.

        Note
        ----
            When including an interaction (e.g. 'treatment'), the model will
            include the 'geno' and 'treatment' variable, and the interaction
            between them (e.g. 'geno + treatment + geno:treatment').

        Note
        ----
            If the 'geno' variable is missing from the list of predictors, it
            will be automatically added in the model.

        """
        # Checking if the 'geno' variable is present in the predictor variables
        if "geno" not in predictors:
            predictors = ["geno"] + predictors

        # Creating the terms for the left hand side of the equation
        lhs = [patsy.Term([patsy.EvalFactor(o)]) for o in outcomes]

        # Creating the terms for the right hand side of the equation
        rhs = [patsy.Term([patsy.EvalFactor(p)]) for p in predictors]

        # Adding an intercept if required
        if intercept:
            rhs.append(patsy.Term([]))

        # The variable to collect results from
        self._result_col = "geno"

        # Adding the interaction
        if interaction is not None:
            if interaction not in predictors:
                rhs.append(patsy.Term([patsy.EvalFactor(interaction)]))
            rhs.append(patsy.Term([patsy.EvalFactor("geno"),
                                   patsy.EvalFactor(interaction)]))

            # Updating the name with the interaction term
            self._result_col = rhs[-1].name()

        # Saving the variables
        self._model = patsy.ModelDesc(lhs, rhs)

    def get_model_description(self):
        """Returns the string representing the model."""
        return self._model.describe()

    def create_matrices(self, data, create_dummy=True):
        """Creates the y and X matrices for a statistical analysis.

        Args:
            data (project_x.phenotypes.core.PhenotypesContainer): The data.
            create_dummy (bool): If True, a dummy column will be added for the
                                 genotypes.

        Returns:
            tuple: y and X as pandas dataframes (according to the formula).

        """
        # We get the phenotypes
        pheno = data.get_phenotypes().dropna()

        # Creating dummy columns (if required)
        if create_dummy:
            pheno = pheno.copy(deep=True)
            pheno["geno"] = np.zeros(pheno.shape[0])

        # Getting the matrices
        y, X = patsy.dmatrices(self._model, pheno, return_type="dataframe")

        # Deleting the dummy columns (if required)
        if create_dummy:
            X = X.drop("geno", axis=1)

        if self._inter is not None:
            # Getting the column containing the interaction term
            inter_term = [term for term in X.columns
                          if term.startswith("geno:" + self._inter)]

            # If there are more than one term, then the interaction was with a
            # categorical value with more than two values
            if len(inter_term) > 1:
                raise ValueError("For the interaction between 'geno' and "
                                 "'{inter}', the categorical '{inter}' "
                                 "variable should have no more than 2 "
                                 "categories.".format(inter=self._inter))
            assert len(inter_term) == 1, "Interaction term is weird..."

            # The results should be gathered for the interaction term
            self._result_col = inter_term[0]

            # Also, there will be a new name for the interaction term
            new_col = [term for term in X.columns
                       if term.startswith(self._inter)][0]
            assert "geno:" + new_col == inter_term[0]
            self._inter_col = new_col

            # Dropping the column
            if create_dummy:
                X = X.drop(self._result_col, axis=1)

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

        Note
        ----
            If there is an interaction term, the 'geno' column is multiplied
            with the other variable.

        """
        new_X = pd.merge(
            X, genotypes, left_index=True, right_index=True,
        ).dropna()
        new_y = y.loc[new_X.index, :]

        # Check if there is interaction
        if compute_interaction and self._inter is not None:
            # There is, so we multiply
            new_X[self._result_col] = new_X.geno * new_X[self._inter_col]

        return new_y, new_X


class StatsResults(object):
    def __init__(self, **kwargs):
        # Creating the empty '_index_of' dictionary
        self.__dict__["_index_of"] = {}

        # Removing the 'print_order', if any
        if "print_order" in kwargs:
            self._print_order = kwargs.pop("print_order")
            if set(self._print_order) != set(kwargs.keys()):
                raise ValueError("Missing statistics in the print order")
        else:
            self._print_order = sorted(kwargs.keys())

        # '_index_of' has all the possible statistics
        self._index_of = {
            name: i for i, name in enumerate(self._print_order)
        }

        # Saving the description of each statistics
        self._description = kwargs

        # Creating the array that will contain the values
        self._results = np.full(len(self._index_of), np.nan, dtype=float)

    def __getattr__(self, name):
        if name in self._index_of:
            return self._results[self._index_of[name]]
        raise ValueError("{}: unknown statistic".format(name))

    def __setattr__(self, name, value):
        if name in self._index_of:
            self._results[self._index_of[name]] = value
        else:
            super().__setattr__(name, value)

    def reset(self):
        """Resets the statistics (sets all the values to NaN)."""
        self._results[:] = np.nan

    def get_statistic_names(self):
        """Returns the print order of the statistics."""
        return self._print_order

    def get_statistics(self):
        """Returns the statistics."""
        return self._results


class StatsError(Exception):
    """An Exception raised if there is any statistical problem."""
    def __init__(self, msg):
        self.message = str(msg)

    def __str__(self):
        return self.message
