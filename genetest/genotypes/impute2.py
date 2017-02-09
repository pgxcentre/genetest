"""
"""

# This file is part of genetest.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import io
import os
import zlib
from collections import namedtuple

import numpy as np
import pandas as pd

from .core import (
    GenotypesContainer, Representation, MarkerGenotypes, MarkerInfo
)


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


__all__ = ["Impute2Genotypes"]


_Impute2Line = namedtuple("_Impute2Line",
                          ["marker", "chrom", "pos", "a1", "a2", "prob"])


class Impute2Genotypes(GenotypesContainer):
    def __init__(self, filename, sample_filename,
                 representation=Representation.ADDITIVE,
                 probability_threshold=0):
        """Instantiate a new Impute2Genotypes object.

        Args:
            filename (str): The name of the IMPUTE2 file.
            sample_filename (str): The name of the sample file.
            representation (str): A valid genotype representation format (e.g.
                                  genotypes.core.Representation.ADDITIVE).
            probability_threshold (float): The probability threshold.

        """
        # Checking the representation
        self.check_representation(representation)
        self._representation = representation

        # Reading the samples
        self.samples = pd.read_csv(sample_filename, sep=" ", skiprows=2,
                                   names=["fid", "iid", "missing", "father",
                                          "mother", "sex", "plink_geno"])

        self.samples["fid"] = self.samples["fid"].astype(str)
        self.samples["iid"] = self.samples["iid"].astype(str)

        # We want to set the index for the samples
        try:
            self.samples = self.samples.set_index("iid", verify_integrity=True)

        except ValueError:
            self.samples["fid_iid"] = [
                "{fid}_{iid}".format(fid=fid, iid=iid)
                for fid, iid in zip(self.samples.fid, self.samples.iid)
            ]
            self.samples = self.samples.set_index("fid_iid",
                                                  verify_integrity=True)

        # The IMPUTE2 file
        self._impute2_file = get_open_func(filename)(filename, "r")

        # If we have an index, we read it
        self._impute2_index = None
        if os.path.isfile(filename + ".idx"):
            self._impute2_index = get_index(
                filename,
                cols=[0, 1, 2],
                names=["chrom", "name", "pos"],
                sep=" ",
            ).set_index("name", verify_integrity=True)

        # Saving the probability threshold
        self.prob_t = probability_threshold

    def close(self):
        if self._impute2_file:
            self._impute2_file.close()

    def __repr__(self):
        """The string representation."""
        nb_markers = self.get_nb_markers()
        if nb_markers is not None:
            return "Impute2Genotypes({:,d} samples, {:,d} markers)".format(
                self.get_nb_samples(),
                nb_markers,
            )
        else:
            return "Impute2Genotypes({:,d} samples)".format(
                self.get_nb_samples(),
            )

    def get_genotypes(self, marker):
        """Returns a dataframe of genotypes encoded using the provided model.

        Args:
            marker (str): A marker ID (e.g. rs123456).

        Returns:
            MarkerGenotypes: A named tuple containing the dataframe with the
            encoded genotypes for all samples (the index of the dataframe will
            be the sample IDs), the minor and major alleles.

        """
        if self._impute2_index is None:
            raise NotImplementedError("Not implemented when IMPUTE2 file is "
                                      "not indexed (see genipe)")

        # Seeking to the right place in the file
        self._impute2_file.seek(int(self._impute2_index.loc[marker, "seek"]))

        # Returning the genotypes
        return self._create_genotypes(
            impute2_line=self._impute2_file.readline(),
        )

    def iter_marker_info(self):
        """Iterate over all available markers without reading genotypes."""
        if self._impute2_index is None:
            raise NotImplementedError("Not implemented when IMPUTE2 file is "
                                      "not indexed (see genipe)")

        for name, row in self._impute2_index.iterrows():
            # Seeking to the right place in the file
            f = self._impute2_file
            f.seek(int(row.seek))
            chrom, name, pos, a1, a2 = f.read(1024).split(" ")[:5]
            pos = int(pos)

            yield MarkerInfo(marker=name, chrom=chrom, pos=pos, a1=a1, a2=a2)

    def iter_marker_genotypes(self):
        """Returns a dataframe of genotypes encoded using the provided model.

        Returns:
            MarkerGenotypes: A named tuple containing the dataframe with the
            encoded genotypes for all samples (the index of the dataframe will
            be the sample IDs), the minor and major alleles.

        """

        for line in self._impute2_file:
            yield self._create_genotypes(
                impute2_line=line,
            )

    def _create_genotypes(self, impute2_line):
        """Creates the genotype dataframe from an IMPUTE2 line.

        Args:
            impute2_line (str): The IMPUTE2 line to process.

        Returns:
            Genotypes: A named tuple containing the dataframe with the encoded
            genotypes for all samples (the index of the dataframe will be the
            sample IDs), the minor and major alleles.

        """
        # Reading the probabilities
        marker_info = self._parse_impute2_line(impute2_line)

        # Creating the dosage data
        dosage = self.create_geno_df(
            genotypes=self._compute_dosage(marker_info.prob, self.prob_t),
            samples=self.samples.index,
        )

        # Checking the format is fine
        dosage, minor, major = self.check_genotypes(
            genotypes=dosage,
            minor=marker_info.a2,
            major=marker_info.a1,
        )

        info = MarkerInfo(
            marker=marker_info.marker,
            chrom=self.encode_chrom(marker_info.chrom),
            pos=marker_info.pos,
            a1=minor, a2=major,
            minor=MarkerInfo.A1
        )

        # Returning the value as DOSAGE representation
        if self._representation == Representation.DOSAGE:
            return MarkerGenotypes(info=info, genotypes=dosage)

        # Normal additive values are necessary for ADDITIVE and GENOTYPIC
        geno = self.dosage2additive(dosage)

        # Returning the value as ADDITIVE representation
        if self._representation == Representation.ADDITIVE:
            return MarkerGenotypes(info=info, genotypes=geno)

        # Returning the value as GENOTYPIC representation
        if self._representation == Representation.GENOTYPIC:
            return MarkerGenotypes(info=info,
                                   genotypes=self.additive2genotypic(geno))

    @staticmethod
    def _parse_impute2_line(line):
        """Parses an IMPUTE2 line (a single marker).

        Args:
            line (str): An IMPUTE2 line.

        Returns:
            _Impute2Line: A named tuple containing information about the
            variation (including the probability matmrix).
        """
        # Splitting
        row = line.rstrip("\r\n").split(" ")

        # Constructing the probabilities
        prob = np.array(row[5:], dtype=float)
        prob.shape = (prob.shape[0] // 3, 3)

        return _Impute2Line(marker=row[1], chrom=row[0], pos=int(row[2]),
                            a1=row[3], a2=row[4], prob=prob)

    @staticmethod
    def _compute_dosage(prob, prob_t):
        """Computes the dosage from a probability matrix (IMPUTE2).

        Args:
            prob (numpy.ndarray): The probability matrix.
            prob_t (float): The probability threshold, for which lower values
                            will be set as missing.

        Returns:
            numpy.ndarray: The dosage vector.

        """
        dosage = 2 * prob[:, 2] + prob[:, 1]

        if prob_t > 0:
            dosage[~np.any(prob >= prob_t, axis=1)] = np.nan

        return dosage

    def get_nb_samples(self):
        """Returns the number of samples.

        Returns:
            int: The number of samples.

        """
        return self.samples.shape[0]

    def get_nb_markers(self):
        """Returns the number of markers.

        Returns:
            int: The number of markers.

        """
        if self._impute2_index is not None:
            return self._impute2_index.shape[0]
        else:
            return None


# This was copied from the 'genipe' module
_CHECK_STRING = b"GENIPE INDEX FILE"

try:
    from Bio.bgzf import BgzfReader
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


def _seek_generator(f):
    """Yields seek position for each line.

    Args:
        f (file): the file object.

    """
    yield 0
    for line in f:
        yield f.tell()


def generate_index(fn, cols=None, names=None, sep=" "):
    """Build a index for the given file.

    Args:
        fn (str): the name of the file.
        cols (list): a list containing column to keep (as int).
        names (list): the name corresponding to the column to keep (as str).
        sep (str): the field separator.

    Returns:
        pandas.DataFrame: the index.

    """
    # Some assertions
    assert cols is not None, "'cols' was not set"
    assert names is not None, "'names' was not set"
    assert len(cols) == len(names)

    # Getting the open function
    bgzip, open_func = get_open_func(fn, return_fmt=True)

    # Reading the required columns
    data = pd.read_csv(fn, sep=sep, engine="c", usecols=cols, names=names,
                       compression="gzip" if bgzip else None)

    # Getting the seek information
    f = open_func(fn, "rb")
    data["seek"] = np.fromiter(_seek_generator(f), dtype=np.uint)[:-1]
    f.close()

    # Saving the index to file
    write_index(get_index_fn(fn), data)

    return data


def get_open_func(fn, return_fmt=False):
    """Get the opening function.

    Args:
        fn (str): the name of the file.
        return_fmt (bool): if the file format needs to be returned.

    Returns:
        tuple: either a tuple containing two elements: a boolean telling if the
        format is bgzip, and the opening function.

    """
    # The file might be compressed using bgzip
    bgzip = None
    with open(fn, "rb") as i_file:
        bgzip = i_file.read(3) == b"\x1f\x8b\x08"

    if bgzip and not HAS_BIOPYTHON:
        raise ValueError("needs BioPython to index a bgzip file")

    open_func = open
    if bgzip:
        open_func = BgzfReader

    # Trying to read
    try:
        with open_func(fn, "r") as i_file:
            if bgzip:
                if not i_file.seekable():
                    raise ValueError
            pass

    except ValueError:
        raise ValueError("{}: use bgzip for compression...".format(fn))

    if return_fmt:
        return bgzip, open_func

    return open_func


def get_index(fn, cols, names, sep):
    """Restores the index for a given file.

    Args:
        fn (str): the name of the file.
        cols (list): a list containing column to keep (as int).
        names (list): the name corresponding to the column to keep (as str).
        sep (str): the field separator.

    Returns:
        pandas.DataFrame: the index.

    If the index doesn't exist for the file, it is first created.

    """
    if not has_index(fn):
        # The index doesn't exists, generate it
        return generate_index(fn, cols, names, sep)

    # Retrieving the index
    file_index = read_index(get_index_fn(fn))

    # Checking the names are there
    if len(set(names) - (set(file_index.columns) - {'seek'})) != 0:
        raise ValueError("{}: missing index columns: reindex".format(fn))

    if "seek" not in file_index.columns:
        raise ValueError("{}: invalid index: reindex".format(fn))

    return file_index


def write_index(fn, index):
    """Writes the index to file.

    Args:
        fn (str): the name of the file that will contain the index.
        index (pandas.DataFrame): the index.

    """
    with open(fn, "wb") as o_file:
        o_file.write(_CHECK_STRING)
        o_file.write(zlib.compress(bytes(
            index.to_csv(None, index=False, encoding="utf-8"),
            encoding="utf-8",
        )))


def read_index(fn):
    """Reads index from file.

    Args:
        fn (str): the name of the file containing the index.

    Returns:
        pandas.DataFrame: the index of the file.

    Before reading the index, we check the first couple of bytes to see if it
    is a valid index file.

    """
    index = None
    with open(fn, "rb") as i_file:
        if i_file.read(len(_CHECK_STRING)) != _CHECK_STRING:
            raise ValueError("{}: not a valid index file".format(fn))

        index = pd.read_csv(io.StringIO(
            zlib.decompress(i_file.read()).decode(encoding="utf-8"),
        ))

    return index


def get_index_fn(fn):
    """Generates the index filename from the path to the indexed file.

    Args:
        fn (str): the name of the file for which we want an index.

    Returns:
        str: the name of the file containing the index.

    """
    return os.path.abspath("{}.idx".format(fn))


def has_index(fn):
    """Checks if the index exists, if not, create it.

    Args:
        fn (str): the name of the file for which we want the index.

    Returns:
        bool: ``True`` if the file contains an index, ``False`` otherwise.

    """
    return os.path.isfile(get_index_fn(fn))
