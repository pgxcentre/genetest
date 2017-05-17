"""
Semantics for the grako parser.
"""

import logging
import warnings

from .core import (SNPs, interaction, phenotypes, genotypes, factor, pow,
                   log10, ln)


logger = logging.getLogger(__name__)


try:
    from .parser import ModelSpecParser
    PARSER_AVAIL = True
except ImportError as e:
    logger.warning("No parser available: " + str(e))
    PARSER_AVAIL = False


class ModelSpecSemantics(object):
    def __init__(self):
        self.entities = {}

    def SNPs(self, ast):
        return SNPs

    def labelled_outcome_group(self, ast):
        return {tree["key"]: tree["name"] for tree in ast["tags"]}

    def interaction(self, ast):
        return interaction(*ast["interaction"], name=ast["as_"])

    def phenotype(self, ast):
        return phenotypes[ast["name"]]

    def integer(self, ast):
        return int(ast["int"])

    def genotype(self, ast):
        return genotypes[ast["variant"]]

    def factor(self, ast):
        return factor(ast["phen"], name=ast["as_"])

    def pow(self, ast):
        return pow(ast["phen"], ast["power"], name=ast["as_"])

    def ln(self, ast):
        return ln(ast["phen"])

    def log10(self, ast):
        return log10(ast["phen"])

    def _default(self, ast):
        return ast


def parse_modelspec(s):
    """Use the modelspec grammar to parse kwargs for modelspec."""
    warnings.warn("use 'parse_formula' instead", DeprecationWarning)
    return parse_formula(s)


def parse_formula(f):
    """Use the modelspec grammar to parse a formula for the modelspec."""
    if not PARSER_AVAIL:
        return

    parser = ModelSpecParser(parseinfo=False)
    return parser.parse(f, rule_name="model", semantics=ModelSpecSemantics())
