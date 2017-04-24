"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.

import importlib.util
import os
import logging
from collections import namedtuple


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["StatsModels", "StatsError", "parse_modelspec"]


# Logging
logger = logging.getLogger(__name__)


# The statistical results
_StatsResults = namedtuple(
    "_StatsResults",
    ["chr", "pos", "snp", "major", "minor", "maf", "n", "stats_n", "stats"],
)


class StatsModels(object):
    def fit(self, y, X):
        """Fit the model.

        Args:
            y (pandas.DataFrame): The vector of endogenous variable.
            X (pandas.DataFrame): The matrix of exogenous variables.

        Returns:
            dict: All available statistics for a given test.

        """
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__


class StatsError(Exception):
    """An Exception raised if there is any statistical problem."""
    def __init__(self, msg):
        self.message = str(msg)

    def __str__(self):
        return self.message


def parse_modelspec(filename):
    """Dynamically import a ModelSpec instance from a Python file.

    The file needs to define a model variable corresponding to the ModelSpec
    instance.

    """
    spec = importlib.util.spec_from_file_location(
        "model",
        os.path.abspath(filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model
