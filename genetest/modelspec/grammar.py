"""
Semantics for the grako parser.
"""

from .core import SNPs, interaction, phenotypes


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
        return interaction(*ast["interaction"])

    def name(self, ast):
        return phenotypes[ast["name"]]

    def _default(self, ast):
        return ast


def parse_modelspec(s):
    """Use the modelspec grammar to parse kwargs for modelspec."""
    if not PARSER_AVAIL:
        return

    parser = ModelSpecParser()
    return parser.parse(s, rule_name="model", semantics=ModelSpecSemantics())
