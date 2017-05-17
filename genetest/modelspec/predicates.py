"""
Utilities to build statistical models.
"""

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


from ..statistics.descriptive import get_maf


__all__ = ["MAFFilter", "NameFilter"]


class VariantPredicate(object):
    def __init__(self):
        """Initialize a callable that will serve as a variant filtering
        function.

        The predicate should return True if the variant is to be analyzed and
        False otherwise.

        Variant predicates can also raise StopIteration to stop pushing SNPs.

        """
        pass


class MAFFilter(VariantPredicate):
    def __init__(self, maf):
        """Filters variants with a MAF under the specified threshold.

        Args:
            maf (float): The MAF threshold.

        """
        self.maf = maf

    def __call__(self, snp):
        """Executes the filter.

        Args:
            snp (): The variant object.

        Returns:
            bool: ``True`` if MAF >= threshold, ``False`` otherwise.

        """
        maf = get_maf(snp.genotypes.values, None, None)[0]
        return maf >= self.maf


class NameFilter(VariantPredicate):
    def __init__(self, extract):
        """Filters variant using a set of variant name.

        Args:
            extract (set): A set of variant names to extract.

        """
        self.extract = extract

    def __call__(self, snp):
        """Execute the filter.

        Args:
            snp (geneparse.Genotypes): The variant object.

        Returns:
            bool: ``True`` if the variant is in the list to extract, ``False``
            otherwise.

        """
        return snp.variant.name in self.extract
