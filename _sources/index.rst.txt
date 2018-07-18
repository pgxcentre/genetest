.. genetest documentation master file, created by
   sphinx-quickstart on Wed Sep 14 13:01:46 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

genetest: efficient genetic association analysis
=================================================


Introduction
-------------

The :py:mod:`genetest` module provides an easy and efficient way of performing
genetics association analysis. It uses :py:mod:`geneparse` to efficiently parse
genetic data files (using indexes) of many commonly used format (*Plink*
binary, *IMPUTE2*, *BGEN* and *VCF* files) and can analyze the data using
linear and logistic regressions, mixed linear model (repeated measurements) or
survival regression (Cox Proportional Hazards). It allows for interactions and
multiple co-variables.

A command line interface tool was also added to facilitate genome-wide
association studies using a configuration file (*YAML* format) to describe the
analysis (:ref:`genotypes <yaml_genotypes_section>`,
:ref:`phenotypes <yaml_phenotypes_section>` and
:ref:`statistical model <yaml_statistics_section>`).

.. toctree::
    :maxdepth: 2

    installation
    analysis
    module_content/modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

