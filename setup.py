#!/usr/bin/env python

# How to build source distribution
#   - python setup.py sdist --format bztar
#   - python setup.py sdist --format gztar
#   - python setup.py sdist --format zip
#   - python setup.py bdist_wheel


import os
import sys
import subprocess

from setuptools import setup, find_packages


MAJOR = 0
MINOR = 2
MICRO = 0
VERSION = "{0}.{1}.{2}".format(MAJOR, MINOR, MICRO)


def check_python_version():
    """Checks the python version, exits if < 3.3."""
    python_major, python_minor = sys.version_info[:2]

    if python_major != 3 or python_minor < 3:
        sys.stderr.write("genetest requires python 3 "
                         "(version 3.3 or higher)\n")
        sys.exit(1)


def write_version_file(fn=None):
    if fn is None:
        fn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join("genetest", "version.py"),
        )

    content = ("\n# THIS FILE WAS GENERATED AUTOMATICALLY\n"
               'genetest_version = "{version}"\n')

    a = open(fn, "w")
    try:
        a.write(content.format(version=VERSION))
    finally:
        a.close()


def build_grako_parser():
    base = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "genetest", "modelspec"
        )
    )
    target = os.path.join(base, "parser.py")

    grammar = os.path.join(base, "modelspec.ebnf")

    args = [
        "python", "-m", "grako", grammar, "-o", target
    ]
    print("Building modelspec grammar parser.")
    print(" ".join(args))
    subprocess.check_call(args)


def setup_package():
    # Checking the python version prior to installation
    check_python_version()

    # Saving the version into a file
    write_version_file()

    # Build the grecko parser.
    build_grako_parser()

    setup(
        name="genetest",
        version=VERSION,
        description="A package to process and analyze genotypic and "
                    "phenotypic data.",
        long_description="This package provides tools to process and analyse "
                         "genotypic and phenotypic data using different "
                         "statistical models. The 'genotypes' and "
                         "'phenotypes' sub-modules provide interfaces to "
                         "process genotypes and phenotypes in different "
                         "format. The 'statistics' sub-module will interface "
                         "with them to analyze the data using different "
                         "statistical models.",
        url="https://github.com/pgxcentre/genetest",
        license="CC BY-NC 4.0",
        test_suite="genetest.tests.test_suite",
        zip_safe=False,
        entry_points={
            "console_scripts": [
                "genetest=genetest.scripts.cli:main",
            ],
        },
        install_requires=["numpy >= 1.12.0", "pandas >= 0.19.0",
                          "setuptools >= 26.1.0", "statsmodels >= 0.8.0",
                          "grako >= 3.10.0", "scipy >= 0.19.0",
                          "geneparse >= 0.1.0", "pyyaml >= 3.12"],
        packages=find_packages(),
        package_data={"genetest.tests": ["data/genotypes/*",
                                         "data/statistics/*"]},
        classifiers=["Development Status :: 4 - Beta",
                     "Intended Audience :: Science/Research",
                     "License :: Free for non-commercial use",
                     "Operating System :: Unix",
                     "Operating System :: POSIX :: Linux",
                     "Operating System :: MacOS :: MacOS X",
                     "Operating System :: Microsoft",
                     "Programming Language :: Python",
                     "Programming Language :: Python :: 3.3",
                     "Programming Language :: Python :: 3.4",
                     "Programming Language :: Python :: 3.5",
                     "Topic :: Scientific/Engineering :: Bio-Informatics"],
        keywords="bioinformatics genetics statistics",
    )


if __name__ == "__main__":
    setup_package()
