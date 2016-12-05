
# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import logging


try:
    from .version import genetest_version as __version__
except ImportError:
    __version__ = None


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__credits__ = ["Louis-Philippe Lemieux Perreault", "Marc-Andre Legault"]
__license__ = "CC BY-NC 4.0"
__status__ = "Development"


# The log format
_LOG_FORMAT = logging.Formatter(
    fmt="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def test(verbosity=1):
    """Execute all the tests."""
    import unittest
    from .tests import test_suite

    unittest.TextTestRunner(verbosity=verbosity).run(test_suite)
