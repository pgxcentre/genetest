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


__all__ = ["arguments"]


class arguments(object):
    """A decorator to add required and optional arguments to a class."""
    def __init__(self, required, optional={}):
        # Saving the required and optional arguments
        self._required_arguments = tuple(req[0] for req in required)
        self._optional_arguments = {k: v[1] for k, v in optional.items()}

        # Saving all the arguments type
        self._arguments_type = {k: v[0] for k, v in optional.items()}
        for req in required:
            self._arguments_type[req[0]] = req[1]

    def __call__(self, cls):
        # Saving the arguments information into the decorated class
        cls._required_args = self._required_arguments
        cls._optional_args = self._optional_arguments
        cls._args_type = self._arguments_type

        # Returning the class
        return cls
