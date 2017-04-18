"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.

__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "CC BY-NC 4.0"


from . import core
from .core import ModelSpec
from .grammar import parse_modelspec, parse_formula


__all__ = [
    "ModelSpec", "parse_modelspec", "parse_formula", "result", "phenotypes",
    "genotypes", "factor", "log10", "pow", "interaction", "SNPs", "_reset",
    "pheWAS"
]


result = core.result

phenotypes = core.phenotypes
genotypes = core.genotypes

factor = core.factor
log10 = core.log10
ln = core.ln
pow = core.pow
interaction = core.interaction

SNPs = core.SNPs

_reset = core._reset

PheWAS = core.PheWAS
