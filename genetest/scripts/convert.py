"""
Small utility to convert the output from imputed-stats to genetest.
"""

# This file is part of genetest
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import sys


EXPECTED_COLUMNS = {
    "chr",
    "pos",
    "snp",
    "major",
    "minor",
    "maf",
    "n",
    "coef",
    "se",
    "lower",
    "upper",
    "t",
    "p",
}


def print_usage():
    print("Usage:")
    print("imputed-stats2genetest my_file.linear.dosage")
    quit(1)


def print_header(linear):
    cols = ["snp", "chr", "pos", "major", "minor", "maf", "n", "ll"]

    if linear:
        cols.append("adj_r2")

    cols += ["coef", "se", "lower", "upper", "t", "p"]

    print(*cols, sep="\t")


def convert_file(f):
    header = next(f).strip().split("\t")
    header = {col: i for i, col in enumerate(header)}

    linear = "adj.r-squared" in header

    # Check that the file has the appropriate columns.
    missing_cols = EXPECTED_COLUMNS - set(header.keys())
    if missing_cols:
        raise ValueError("Could not find column(s): {}.".format(missing_cols))

    print_header(linear)
    for line in f:
        line = line.strip().split("\t")
        cols = [
            line[header["snp"]],
            line[header["chr"]],
            line[header["pos"]],
            line[header["major"]],
            line[header["minor"]],
            line[header["maf"]],
            line[header["n"]],
            "NA",  # log-likelihood.
        ]

        if linear:
            cols.append(line[header["adj.r-squared"]])

        cols += [
            line[header["coef"]],
            line[header["se"]],
            line[header["lower"]],
            line[header["upper"]],
            line[header["t"]],
            line[header["p"]],
        ]

        print(*cols, sep="\t")


def main():
    if len(sys.argv) != 2:
        print_usage()

    filename = sys.argv[1]
    try:
        with open(filename, "r") as f:
            convert_file(f)
    except FileNotFoundError:
        print_usage()
