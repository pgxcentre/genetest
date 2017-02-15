"""
Subscribers used to format the output of analyses.
"""


# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import sys
import pprint
import logging


logger = logging.getLogger(__name__)


class Subscriber(object):
    """Abstract class for subscribers."""
    def __init__(self):
        self._reset_subset()

    def init(self, modelspec):
        """Method that gets called by analysis to initialize or reinitialize
        the Subscriber with a modelspec.

        """
        self.modelspec = modelspec

    def close(self):
        pass

    def _update_current_subset(self, info):
        """Updates information on which part of the dataset is analyzed.

        This method get called by analysis before data is pushed.

        Subclasses should not have to modify this method.

        """
        self.subset_info = info

    def _reset_subset(self):
        self.subset_info = None

    def handle(self, results):
        """Handle results from a statistical test."""
        raise NotImplementedError()

    @staticmethod
    def _apply_translation(translation, results):
        out = {}
        for k in results:
            if k in translation:
                out[translation[k]] = results[k]
            else:
                out[k] = results[k]
        return out


class ResultsMemory(Subscriber):
    def __init__(self):
        self.results = []

    def handle(self, results):
        self.results.append(Subscriber._apply_translation(
            self.modelspec.get_translations(),
            results
        ))

    def _get_gwas_results(self):
        out = {}
        for result in self.results:
            name = result["SNPs"]["name"]
            out[name] = result
        return out


class Print(Subscriber):
    def __init__(self, raw=False):
        self.raw = raw

    def handle(self, results):
        if self.raw:
            pprint.pprint(results)
            return

        pprint.pprint(Subscriber._apply_translation(
            self.modelspec.get_translations(),
            results
        ))


class RowWriter(Subscriber):
    def __init__(self, filename=None, columns=None, header=False, sep="\t",
                 append=False):
        # columns needs to be a list of 2-tuples ('col', result object or str).
        self.header = header
        self.columns = columns
        self.sep = sep
        self.filename = filename

        if filename:
            self._f = open(filename, "a" if append else "w")
        else:
            self._f = None

        if self.header:
            header = self.sep.join([i[0] for i in self.columns])
            if filename is not None:
                self._f.write(header + "\n")
            else:
                print(header)

    def close(self):
        if self._f:
            self._f.close()

    def handle(self, results):
        row = []
        for name, result in self.columns:
            if isinstance(result, str):
                row.append(result)
            else:
                row.append(str(result.get(results)))

        row = self.sep.join(row)
        if self.filename is not None:
            self._f.write(row + "\n")
            self._f.flush()
        else:
            print(row)


def subscriber_error(message, abort=None):
    """Logs the error from the subscriber."""
    logger.critical(
        "A subscriber for this analysis raised an exception. "
        "This is be because an invalid key was accessed from the results of "
        "the statistical test.\n"
        "Unknown field: '{}'".format(message)
    )
    if abort:
        abort.set()
    sys.exit(1)
