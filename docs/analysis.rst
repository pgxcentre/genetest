
Analysis
=========

Multiple statistic models exists. To execute a genome-wide study, you can use
the ``geneparse`` tool.


.. code-block:: console

    $ genetest --help
    usage: genetest [-h] [-v] [--test] [--nb-cpus NB] --configuration YAML
                    [--output FILE] [--extract FILE] [--keep FILE] [--maf MAF]

    Performs statistical analysis on genotypic data (version 0.3.0).

    optional arguments:
    -h, --help            show this help message and exit
    -v, --version         show program's version number and exit
    --test                Execute the test suite and exit.
    --nb-cpus NB          The number of processes to use for the analysis. [1]

    Input Options:
    --configuration YAML  The configuration file that describe the phenotypes,
                            genotypes, and model.

    Output Options:
    --output FILE         The output file prefix that will contain the results
                            and other information. [genetest_results]

    Other Options:
    --extract FILE        A file containing a list of marker to extract prior to
                            the statistical analysis (one marker per line).
    --keep FILE           A file containing a list of samples to keep prior to
                            the statistical analysis (one sample per line).
    --maf MAF             The MAF threshold to include a marker in the analysis.
                            [0.01]


A single configuration file (using the *YAML* format) describes the genetic and
phenotypic files and the statistical model to perform. The following describe
the different sections.


Genotypes
----------

The ``genotypes`` section describes the genetic part of the analysis. Multiple
file format are available (see below). The required keyword ``format``
describes the file format for the genotypes.

Each of the formats have their own required arguments and options.


impute2
^^^^^^^^

The following arguments and options are available for this format.

.. table::
    :widths: 20 70 10

    +---------------------------+--------------------------------------+----------+
    | Argument                  | Description                          | Required |
    +===========================+======================================+==========+
    | ``filename``              | The name of the *impute2* file.      | Yes      |
    +---------------------------+--------------------------------------+----------+
    | ``sample_filename``       | The name of the *sample* file.       | Yes      |
    +---------------------------+--------------------------------------+----------+
    | ``probability_threshold`` | The probability threshold. Genotypes |          |
    |                           | with the maximal probability lower   |          |
    |                           | to this value will be set as missing.|          |
    |                           | [Default: 0.9]                       |          |
    +---------------------------+--------------------------------------+----------+

Below is an example of a ``genotypes`` section of the *YAML* configuration
file for an IMPUTE2 format.

.. code-block:: yaml

    genotypes:
        format: impute2
        filename: cohort.impute2
        sample_filename: cohort.sample
        options:
            probability_threshold: 0.9


bed/bim/fam
^^^^^^^^^^^^

Only one argument is required.

.. table::
    :widths: 20 70 10

    +---------------------------+--------------------------------------+----------+
    | Argument                  | Description                          | Required |
    +===========================+======================================+==========+
    | ``prefix``                | The prefix of the BED/BIM/FAM files  | Yes      |
    +---------------------------+--------------------------------------+----------+

Below is an example of a ``genotypes`` section of the *YAML* configuration file
for a binary plink format.

.. code-block:: yaml

    genotypes:
        format: plink
        prefix: cohort


bgen
^^^^^

The following arguments and options are available for this format.

.. table::
    :widths: 20 70 10

    +---------------------------+--------------------------------------+----------+
    | Argument                  | Description                          | Required |
    +===========================+======================================+==========+
    | ``filename``              | The name of the *bgen* file.         | Yes      |
    +---------------------------+--------------------------------------+----------+
    | ``sample_filename``       | The name of the *sample* file.       | Yes      |
    +---------------------------+--------------------------------------+----------+
    | ``probability_threshold`` | The probability threshold. Genotypes |          |
    |                           | with the maximal probability lower   |          |
    |                           | to this value will be set as missing.|          |
    |                           | [Default: 0.9]                       |          |
    +---------------------------+--------------------------------------+----------+
    | ``cpus``                  | The number of CPUs to use while      |          |
    |                           | reading the *bgen* file. [Default: 1]|          |
    +---------------------------+--------------------------------------+----------+

Below is an example of a ``genotypes`` section of the *YAML* configuration file
for a *bgen* file.

.. code-block:: yaml

    genotypes:
        format: bgen
        filename: cohort.bgen
        sample_filename: cohort.sample
        options:
            probability_threshold: 0.9
            cpus: 1


vcf
^^^^

Only one argument is required.

.. table::
    :widths: 20 70 10

    +---------------------------+--------------------------------------+----------+
    | Argument                  | Description                          | Required |
    +===========================+======================================+==========+
    | ``filename``              | The name of the VCF file.            | Yes      |
    +---------------------------+--------------------------------------+----------+

Below is an example of a ``genotypes`` section of the *YAML* configuration file
for the VCF format.

.. code-block:: yaml

    genotypes:
        format: vcf
        filename: cohort.vcf


Phenotypes
-----------

The ``phenotypes`` section describes the phenotypes and variables that will be
used in the statistical model. At the moment, only one format is available.


text
^^^^^

The following arguments and options are available for this format.

.. table::
    :widths: 20 70 10

    +---------------------------+--------------------------------------+----------+
    | Argument                  | Description                          | Required |
    +===========================+======================================+==========+
    | ``filename``              | The name of the *bgen* file.         | Yes      |
    +---------------------------+--------------------------------------+----------+
    | ``sample_column``         | The name of the column containing the|          |
    |                           |  sample ID. This column will be used |          |
    |                           |  to match the phenotypes with the    |          |
    |                           |  genotypes. [Default: sample]        |          |
    +---------------------------+--------------------------------------+----------+
    | ``field_separator``       | The character that separate a field  |          |
    |                           | in the file. [Default: '\\t']        |          |
    +---------------------------+--------------------------------------+----------+
    | ``missing_values``        | The values that are considered as    |          |
    |                           | missing. An empty field is           |          |
    |                           | automaticaly considered as missing.  |          |
    +---------------------------+--------------------------------------+----------+
    | ``repeated_measurements`` | Enter 'Yes' if the file containes    |          |
    |                           | repeated measurements.               |          |
    +---------------------------+--------------------------------------+----------+
    | ``keep_sample_column``    | For now, if repeated measurements are|          |
    |                           | used (*i.e.* Yes at the previous     |          |
    |                           | option), enter 'Yes' to tell the     |          |
    |                           | parser to keep the sample column for |          |
    |                           | the statistical analysis (will be    |          |
    |                           | used for groups in the MixedLM       |          |
    |                           | analysis).                           |          |
    +---------------------------+--------------------------------------+----------+

Below is an example of a ``phenotypes`` section of the *YAML* configuration file
for a *text* file containing repeated measurements. The string ``-99999`` is
considered as a missing value.

.. code-block:: yaml

    phenotypes:
        format: text
        filename: phenotypes.txt
        options:
            sample_column: sample_id
            missing_values: "-99999"
            repeated_measurements: Yes
            keep_sample_column: Yes


Statistical model
------------------


Linear regression
^^^^^^^^^^^^^^^^^^


Logistic regression
^^^^^^^^^^^^^^^^^^^^


Repeated measurements
^^^^^^^^^^^^^^^^^^^^^^


Survival analysis
^^^^^^^^^^^^^^^^^^
