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
import datetime

try:
    from twilio.rest import Client
    TWILIO = True
except ImportError:
    TWILIO = False


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

    def _update_gwas_interaction(self, columns):
        """Updates information (whether there is a gwas interaction or not).

        Args:
            columns: The columns for the interaction.

        """
        pass

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


class TwilioSubscriber(Subscriber):
    def __init__(self, account_sid, auth_token, to, from_):
        if not TWILIO:
            raise ImportError("Install twilio to use this subscriber.")

        self.client = Client(account_sid, auth_token)
        self.from_ = from_
        self.to = to

        self.results = []

    def handle(self, result):
        self.results.append(Subscriber._apply_translation(
            self.modelspec.get_translations(),
            result
        ))

    def close(self):
        logger.info("Sending text message to {}.".format(self.to))

        message = datetime.datetime.now().strftime(
            "Analysis finished on %B %d at %H:%M.\n"
        )
        message += (
            "Received results from {} statistical tests.".format(
                len(self.results)
            )
        )

        self.client.messages.create(
            to=self.to,
            from_=self.from_,
            body=message
        )


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
        self.sep = sep
        self.filename = filename

        self._set_columns(columns)

        if filename:
            self._f = open(filename, "a" if append else "w")
        else:
            self._f = None

        if self.header:
            self.print_header()

    def _set_columns(self, columns):
        self.columns = columns

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
        # A flag for updated values because of GWAS interaction
        self._inter_already_updated = False

        # Setting the final columns to None
        self._final_columns = None

        # The columns that are in common for each model
        self._common_cols = [
            ("snp", analysis_results["SNPs"]["name"]),
            ("chr", analysis_results["SNPs"]["chrom"]),
            ("pos", analysis_results["SNPs"]["pos"]),
            ("major", analysis_results["SNPs"]["major"]),
            ("minor", analysis_results["SNPs"]["minor"]),
            ("maf", analysis_results["SNPs"]["maf"]),
            ("n", analysis_results["MODEL"]["nobs"]),
            ("ll", analysis_results["MODEL"]["log_likelihood"]),
        ]

        # The columns that are specific for each model and analysis
        self._specific_cols = []
        self._specific_model_cols = []

        # Linear
        if test == "linear":
            self._specific_cols = [
                ("coef", ("SNPs", "coef")),
                ("se", ("SNPs", "std_err")),
                ("lower", ("SNPs", "lower_ci")),
                ("upper", ("SNPs", "upper_ci")),
                ("t", ("SNPs", "t_value")),
                ("p", ("SNPs", "p_value")),
            ]
            self._specific_model_cols.append(
                ("adj_r2", analysis_results["MODEL"]["r_squared_adj"]),
            )

        # Logistic
        elif test == "logistic":
            self._specific_cols = [
                ("coef", ("SNPs", "coef")),
                ("se", ("SNPs", "std_err")),
                ("lower", ("SNPs", "lower_ci")),
                ("upper", ("SNPs", "upper_ci")),
                ("t", ("SNPs", "t_value")),
                ("p", ("SNPs", "p_value")),
            ]

        # CoxPH
        elif test == "coxph":
            self._specific_cols = [
                ("coef", ("SNPs", "coef")),
                ("se", ("SNPs", "std_err")),
                ("hr", ("SNPs", "hr")),
                ("hr_lower", ("SNPs", "hr_lower_ci")),
                ("hr_upper", ("SNPs", "hr_upper_ci")),
                ("z", ("SNPs", "z_value")),
                ("p", ("SNPs", "p_value")),
            ]

        # MixedLM
        elif test == "mixedlm":
            self._specific_cols = [
                ("coef", ("SNPs", "coef")),
                ("se", ("SNPs", "std_err")),
                ("lower", ("SNPs", "lower_ci")),
                ("upper", ("SNPs", "upper_ci")),
                ("z", ("SNPs", "z_value")),
                ("p", ("SNPs", "p_value")),
            ]

        else:
            logger.warning("{}: invalid test: only common columns will be "
                           "written to file.")

        # Calling super __init__
        super().__init__(filename=filename, columns=None, header=True,
                         append=False, sep=sep)

    @property
    def columns(self):
        # Generating the columns from the common ones, the ones specific to the
        # model, and the ones for the GWAS results
        if self._final_columns is None:
            specific_cols = [
                (output_name, analysis_results[col_name][param_name])
                for output_name, (col_name, param_name) in self._specific_cols
            ]

            self._final_columns = (
                self._common_cols + self._specific_model_cols + specific_cols
            )

        return self._final_columns

    def _set_columns(self, columns):
        pass

    def _reset_output_header(self):
        """Resets the columns."""
        # We reset the final columns
        logger.debug("Re-writing the output header")
        self._final_columns = None

        # We re-write the header
        self._f.seek(0)
        self.print_header()

    def _update_current_subset(self, info):
        """Updates information on which part of the dataset is analyzed."""
        # First, check if the subset_info is None (meaning its the first time
        # this function is called
        if self.subset_info is None:
            logger.debug("Updating columns for subgroup analysis")
            # We add the column
            self._common_cols.append(
                ("subgroup", analysis_results["MODEL"]["subset_info"])
            )

            # Resetting the columns
            self._reset_output_header()

        # Calling super
        super()._update_current_subset(info=info)

    def _update_gwas_interaction(self, columns):
        # First, check if the results from is the default 'SNPs' (meaning it's
        # the first time this function is called). Otherwise, we're in a
        # subgroup analysis
        if not self._inter_already_updated:
            logger.debug("Updating columns for GWAS interaction")
            self._specific_cols = [
                ("{}:{}".format(inter_col, col_info[0]),
                 (inter_col, col_info[1][1]))
                for inter_col in columns for col_info in self._specific_cols
            ]

            # Resetting the columns
            self._reset_output_header()

            # We don't need to change the header again, even for subgroup
            # analysis
            self._inter_already_updated = True

        # Calling super
        super()._update_gwas_interaction(columns=columns)

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
