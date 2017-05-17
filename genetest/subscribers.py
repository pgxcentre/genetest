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
import json
import logging

from .modelspec import result as analysis_results


__all__ = ["Print", "ResultsMemory", "RowWriter", "GWASWriter"]


logger = logging.getLogger(__name__)


class Subscriber(object):
    """Abstract class for subscribers."""
    def init(self, modelspec):
        """Method that gets called by analysis to initialize or reinitialize
        the Subscriber with a modelspec.

        """
        self.modelspec = modelspec
        self._reset_subset()

    def close(self):
        pass

    def _update_current_subset(self, info):
        """Updates information on which part of the dataset is analyzed.

        This method get called by analysis before data is pushed.

        Subclasses should not have to modify this method.

        """
        self.subset_info = info
        self.subset_info_str = ";".join(
            "{}:{}".format(*_) for _ in sorted(info.items())
        )

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
        if self.subset_info:
            results["MODEL"]["subset_info"] = self.subset_info

        if self.raw:
            print(json.dumps(results, indent=2))
            return

        print(json.dumps(Subscriber._apply_translation(
            self.modelspec.get_translations(),
            results
        ), indent=2))


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
            self.print_header()

    def print_header(self):
        header = self.sep.join([i[0] for i in self.columns])
        if self.filename is not None:
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


class GWASWriter(RowWriter):
    def __init__(self, filename, test, sep="\t"):
        # The columns that are always present
        columns = [
            ("snp", analysis_results["SNPs"]["name"]),
            ("chr", analysis_results["SNPs"]["chrom"]),
            ("pos", analysis_results["SNPs"]["pos"]),
            ("major", analysis_results["SNPs"]["major"]),
            ("minor", analysis_results["SNPs"]["minor"]),
            ("maf", analysis_results["SNPs"]["maf"]),
            ("n", analysis_results["MODEL"]["nobs"]),
        ]

        # The columns required by the 'linear' test
        if test == "linear":
            columns.extend([
                ("coef", analysis_results["SNPs"]["coef"]),
                ("se", analysis_results["SNPs"]["std_err"]),
                ("lower", analysis_results["SNPs"]["lower_ci"]),
                ("upper", analysis_results["SNPs"]["upper_ci"]),
                ("t", analysis_results["SNPs"]["t_value"]),
                ("p", analysis_results["SNPs"]["p_value"]),
                ("ll", analysis_results["MODEL"]["log_likelihood"]),
                ("adj_r2", analysis_results["MODEL"]["r_squared_adj"]),
            ])

        # The columns required by the 'logistic' test
        elif test == "logistic":
            columns.extend([
                ("coef", analysis_results["SNPs"]["coef"]),
                ("se", analysis_results["SNPs"]["std_err"]),
                ("lower", analysis_results["SNPs"]["lower_ci"]),
                ("upper", analysis_results["SNPs"]["upper_ci"]),
                ("t", analysis_results["SNPs"]["t_value"]),
                ("p", analysis_results["SNPs"]["p_value"]),
                ("ll", analysis_results["MODEL"]["log_likelihood"]),
            ])

        # The columns required by the 'coxph' test
        elif test == "coxph":
            columns.extend([
                ("coef", analysis_results["SNPs"]["coef"]),
                ("se", analysis_results["SNPs"]["std_err"]),
                ("hr", analysis_results["SNPs"]["hr"]),
                ("hr_lower", analysis_results["SNPs"]["hr_lower_ci"]),
                ("hr_upper", analysis_results["SNPs"]["hr_upper_ci"]),
                ("z", analysis_results["SNPs"]["z_value"]),
                ("p", analysis_results["SNPs"]["p_value"]),
                ("ll", analysis_results["MODEL"]["log_likelihood"]),
            ])

        # The columns required by the 'coxph' test
        elif test == "mixedlm":
            columns.extend([
                ("coef", analysis_results["SNPs"]["coef"]),
                ("se", analysis_results["SNPs"]["std_err"]),
                ("lower", analysis_results["SNPs"]["lower_ci"]),
                ("upper", analysis_results["SNPs"]["upper_ci"]),
                ("z", analysis_results["SNPs"]["z_value"]),
                ("p", analysis_results["SNPs"]["p_value"]),
                ("ll", analysis_results["MODEL"]["log_likelihood"]),
            ])

        else:
            logger.warning("{}: invalid test: only common columns will be "
                           "written to file.")

        # Calling super __init__
        super().__init__(filename=filename, columns=columns, header=True,
                         append=False, sep=sep)

    def _update_current_subset(self, info):
        """Updates information on which part of the dataset is analyzed."""
        # First, check if the subset_info is None (meaning its the first time
        # this function is called
        if self.subset_info is None:
            # We add the column
            self.columns.append(
                ("subgroup", analysis_results["MODEL"]["subset_info"])
            )

            # We re-write the header
            self._f.seek(0)
            self.print_header()

        # Calling super
        super()._update_current_subset(info=info)

    def handle(self, results):
        # Adding the subgroup, if any
        if self.subset_info is not None:
            results["MODEL"]["subset_info"] = self.subset_info_str

        # Calling super
        super().handle(results=results)


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
