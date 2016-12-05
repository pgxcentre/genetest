"""
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


from .impute2 import Impute2Genotypes
from .plink import PlinkGenotypes
from .vcf import VCFGenotypes


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "CC BY-NC 4.0"


__all__ = ["available_formats", "format_map"]


# The available formats (with their description)
available_formats = dict(
    impute2="IMPUTE2 files from imputation project.",
    plink="Binary PLINK files from genotyping project.",
    vcf="VCF format from sequencing project.",
)


# The format map (which maps the name to the class)
format_map = dict(
    impute2=Impute2Genotypes,
    plink=PlinkGenotypes,
    vcf=VCFGenotypes,
)
