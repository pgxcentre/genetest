"""
"""


# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "CC BY-NC 4.0"


__all__ = ["arguments"]


class arguments(object):
    """A decorator to add required and optional arguments to a class."""
    def __init__(self, required, optional={}):
        self._required_arguments = required
        self._optional_arguments = optional

    def __call__(self, cls):
        cls._required_args = self._required_arguments
        cls._optional_args = self._optional_arguments
        return cls
