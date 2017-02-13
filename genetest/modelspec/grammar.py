"""
Semantics for the grako parser.
"""

from .core import SNPs, interaction, phenotypes, genotypes, factor


try:
    from .parser import ModelSpecParser
    PARSER_AVAIL = True
except ImportError:
    PARSER_AVAIL = False


class ModelSpecSemantics(object):
    def __init__(self):
        self.entities = {}

    def SNPs(self, ast):
        return SNPs

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

    def _default(self, ast):
        return ast


def parse_modelspec(s):
    """Use the modelspec grammar to parse kwargs for modelspec."""
    if not PARSER_AVAIL:
        return

    parser = ModelSpecParser()
    return parser.parse(s, rule_name="model", semantics=ModelSpecSemantics())
