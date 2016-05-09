#!/usr/bin/env python

# How to build source distribution
#   - python setup.py sdist --format bztar
#   - python setup.py sdist --format gztar
#   - python setup.py sdist --format zip
#   - python setup.py bdist_wheel


import os
import sys

from setuptools import setup, find_packages


MAJOR = 0
MINOR = 0
MICRO = 1
VERSION = "{0}.{1}.{2}".format(MAJOR, MINOR, MICRO)


def check_python_version():
    """Checks the python version, exits if < 3.3."""
    python_major, python_minor = sys.version_info[:2]

    if python_major != 3 or python_minor < 3:
        sys.stderr.write("genipe requires python 3 (version 3.3 or higher)\n")
        sys.exit(1)


def write_version_file(fn=None):
    if fn is None:
        fn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join("project_x", "version.py"),
        )

    content = ("\n# THIS FILE WAS GENERATED AUTOMATICALLY\n"
               'project_x_version = "{version}"\n')

    a = open(fn, "w")
    try:
        a.write(content.format(version=VERSION))
    finally:
        a.close()


def setup_package():
    # Checking the python version prior to installation
    check_python_version()

    # Saving the version into a file
    write_version_file()

    setup(
        name="project-x",
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
        url="https://github.com/legaultmarc/project_x",
        license="CC BY-NC 4.0",
        entry_points={
            "console_scripts": [
                "launch_project_x=project_x.scripts.cli:main",
            ],
        },
        test_suite="project_x.tests.test_suite",
        zip_safe=False,
        install_requires=["numpy >= 1.11.0", "pandas >= 0.18.0",
                          "pyplink >= 1.2.0", "setuptools >= 12.0.5",
                          "pysam >= 0.9.0"],
        packages=find_packages(),
        package_data={"project_x.tests": ["data/genotypes/*",
                                          "data/statistics/*"]},
        classifiers=["Development Status :: 1 - Planning",
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

    return


if __name__ == "__main__":
    setup_package()
