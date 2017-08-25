"""
Semantics for the grako parser.
"""

import os
import logging
import warnings

from .core import (SNPs, interaction, phenotypes, genotypes, factor, pow,
                   log10, ln, gwas_interaction)


logger = logging.getLogger(__name__)


try:
    from .parser import ModelSpecParser
    PARSER_AVAIL = True
except ImportError as e:
    PARSER_AVAIL = False


class ModelSpecSemantics(object):
    def __init__(self):
        self.entities = {}

    def SNPs(self, ast):
        return SNPs

    def labelled_outcome_group(self, ast):
        return {tree["key"]: tree["name"] for tree in ast["tags"]}

    def interaction(self, ast):
        if SNPs in ast["interaction"]:
            # This is a GWAS interaction
            return gwas_interaction(
                *tuple(term for term in ast["interaction"] if term != SNPs),
                name=ast["as_"] if ast["as_"] else "GWAS_INTER"
            )

        else:
            # This is a normal interaction term
            return interaction(*ast["interaction"], name=ast["as_"])

    def phenotype(self, ast):
        return phenotypes[ast["name"]]

    def string(self, ast):
        return ast["str"][1:-1]

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
        raise RuntimeError("Impossible to parse the grammar. Is the 'grako' "
                           "module installed?")

    parser = ModelSpecParser(parseinfo=False)

    try:
        return parser.parse(
            f, rule_name="model", semantics=ModelSpecSemantics(),
        )

    except:
        msg = ("Something went wrong while parsing the grammar. This might be "
               "because you are using a different 'grako' version than the "
               "one used to compile the grammar. Please perform the following "
               "command:\n\n{command}")

        # Getting the path of the files
        basedir = os.path.abspath(os.path.dirname(__file__))

        # The command to build the parser
        command = [
            "python", "-m", "grako", os.path.join(basedir, "modelspec.ebnf"),
            "-o", os.path.join(basedir, "parser.py"),
        ]

        raise RuntimeError(msg.format(command=" ".join(command)))
