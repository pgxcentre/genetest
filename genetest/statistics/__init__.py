"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


from .models.linear import StatsLinear
from .models.survival import StatsCoxPH
from .models.mixedlm import StatsMixedLM
from .models.logistic import StatsLogistic


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "CC BY-NC 4.0"


__all__ = ["available_models"]


available_models = {
    "linear": "Linear regression (ordinary least squares).",
    "logistic": "Logistic regression (GLM with binomial distribution).",
    "mixedlm": "Linear mixed effect model (random intercept).",
    "coxph": "Cox's proportional hazard regression (survival analysis).",
}


# The model map (which maps the name to the class)
model_map = {
    "linear": StatsLinear,
    "logistic": StatsLogistic,
    "mixedlm": StatsMixedLM,
    "coxph": StatsCoxPH,
}
