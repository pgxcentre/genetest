
Analysis
=========

Multiple statistic models exist. To execute a genome-wide study, you can use
the ``geneparse`` tool.


.. code-block:: console

    $ genetest --help
    usage: genetest [-h] [-v] [--test] [--nb-cpus NB] --configuration YAML
                    [--output FILE] [--extract FILE] [--keep FILE] [--maf MAF]
                    [--sexual-chromosome]

    Performs statistical analysis on genotypic data (version 0.4.1).

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
      --extract FILE        A file containing a list of markers to extract prior
                            to the statistical analysis (one marker per line).
      --keep FILE           A file containing a list of samples to keep prior to
                            the statistical analysis (one sample per line).
      --maf MAF             The MAF threshold to include a marker in the analysis.
                            [0.01]
      --sexual-chromosome   Analysis is performed on a sexual chromosome. This
                            will impact the MAF computation (as males are
                            hemizygotes on sexual chromosomes). This has an effect
                            only on a GWAS analysis.

A single configuration file (using the *YAML* format) describes the
:ref:`genotypes <yaml_genotypes_section>` and
:ref:`phenotypes <yaml_phenotypes_section>` files, and the
:ref:`statistical model <yaml_statistics_section>` to perform. The following
describe the different sections.

In all the cases, when values are optional, it needs to be inserted into a
``options`` subsection, otherwise, the default values will be used (see below
for examples).

.. warning::

    By default, the MAF computed during a GWAS uses the formula for autosomes
    (*i.e.* there is no check for sexual chromosomes). If the analysis is
    performed on a sexual chromosome, make sure to use the
    ``--sexual-chromosome`` flag and to use the YAML option ``sex_column`` (in
    the :ref:`Phenotype <yaml_phenotypes_section>` section) to specify the
    gender column in the phenotype file.

    Also note that when using the ``--sexual-chromosome`` option, all markers
    in the genotype file will be treated as being on a sexual chromosome. If
    the genotype file contains a mixture of autosomes and sexual chromosomes,
    make sure use a combination of the ``--extract`` (extracting only markers
    located on a sexual chromosome) and ``--sexual-chromosome`` in order to
    properly compute the MAF.

    Finally, the column containing the sex in the phenotype file (the
    ``sex_column`` option of the :ref:`Phenotype <yaml_phenotypes_section>`
    section in the YAML file) should have encoding of ``male=1`` and
    ``female=0``.


.. _yaml_genotypes_section:

Genotypes
----------

The ``genotypes`` section describes the genetic part of the analysis. Multiple
file formats are available (see below). The required keyword ``format``
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
for a binary Plink format.

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


.. _yaml_phenotypes_section:

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
    |                           | sample ID. This column will be used  |          |
    |                           | to match the phenotypes with the     |          |
    |                           | genotypes. [Default: sample]         |          |
    +---------------------------+--------------------------------------+----------+
    | ``field_separator``       | The character that separate a field  |          |
    |                           | in the file. [Default: '\\t']        |          |
    +---------------------------+--------------------------------------+----------+
    | ``missing_values``        | A string (using quotes) that         |          |
    |                           | represents missing values. An empty  |          |
    |                           | field, ``NA``, ``nan`` or ``NaN`` are|          |
    |                           | always considered as missing.        |          |
    +---------------------------+--------------------------------------+----------+
    | ``repeated_measurements`` | Enter 'Yes' if the file contains     |          |
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
    | ``sex_column``            | The name of the column containing the|          |
    |                           | sex information. Note that males need|          |
    |                           | to be coded as *1* and females, as   |          |
    |                           | *0*. The choice of this encoding is  |          |
    |                           | to speed up the MAF computation for  |          |
    |                           | sexual chromosomes. This column will |          |
    |                           | be used only if the analysis is      |          |
    |                           | performed using the                  |          |
    |                           | ``--sexual-chromosome`` option.      |          |
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
            sex_column: sex


.. _yaml_statistics_section:

Statistical model
------------------

For now, a total of 4 different analysis is possible: linear and logistic
regressions, repeated measurements analysis using a mixed linear model, and
survival analysis using the Cox proportional hazard regression. Each of those
models (with their configuration) are described below.

The model is described in the ``model`` section of the *YAML* configuration
file and using the ``test`` argument.


Linear regression
^^^^^^^^^^^^^^^^^^

The following arguments and options are available for the linear regression.

.. table::
    :widths: 20 70 10

    +---------------------------+--------------------------------------+----------+
    | Argument                  | Description                          | Required |
    +===========================+======================================+==========+
    | ``formula``               | The formula describing the analysis  | Yes      |
    |                           | to be performed. Note that the       |          |
    |                           | formula is similar to the one used in|          |
    |                           | R. The names of the variables need to|          |
    |                           | be the same as the columns in the    |          |
    |                           | phenotype file. The keyword ``SNPs`` |          |
    |                           | is used to perform a GWAS.           |          |
    +---------------------------+--------------------------------------+----------+
    | ``condition_value_t``     | The condition value threshold (for   |          |
    |                           | multicollinearity). Usually, values  |          |
    |                           | higher than 1000 indicate strong     |          |
    |                           | multicollinearity or other numerical |          |
    |                           | problems. [Default: 1000]            |          |
    +---------------------------+--------------------------------------+----------+
    | ``eigenvals_t``           | The Eigen value threshold (for       |          |
    |                           | multicollinearity). Usually, values  |          |
    |                           | lower than 1e-10 might indicate      |          |
    |                           | strong multicollinearity or singular |          |
    |                           | design matrix. [Default: 1e-10]      |          |
    +---------------------------+--------------------------------------+----------+

Below is an example of a ``model`` section of the *YAML* configuration file
for a linear regression analysis of the phenotype *Pheno* over the variables
*SNPs* (meaning a GWAS), *Age* and *Sex*. It also increases the conditional
value threshold from the default value of 1000 to 5000.

.. code-block:: yaml

    model:
        test: linear
        formula: "Pheno ~ SNPs + Age + factor(Sex)"
        options:
            condition_value_t: 5000

See :py:class:`genetest.statistics.models.linear.StatsLinear` for more
information about the class.


Logistic regression
^^^^^^^^^^^^^^^^^^^^

The logistic regression only requires the formula describing the model.

.. table::
    :widths: 20 70 10

    +---------------------------+--------------------------------------+----------+
    | Argument                  | Description                          | Required |
    +===========================+======================================+==========+
    | ``formula``               | The formula describing the analysis  | Yes      |
    |                           | to be performed. Note that the       |          |
    |                           | formula is similar to the one used in|          |
    |                           | R. The names of the variables need to|          |
    |                           | be the same as the columns in the    |          |
    |                           | phenotype file. The keyword ``SNPs`` |          |
    |                           | is used to perform a GWAS.           |          |
    +---------------------------+--------------------------------------+----------+

Below is an example of a ``model`` section of the *YAML* configuration file
for a logistic regression analysis of the phenotype *Status* over the variables
*SNPs* (meaning a GWAS), *Age* and *Sex*.

.. code-block:: yaml

    model:
        test: logistic
        formula: "Status ~ SNPs + Age + factor(Sex)"

See :py:class:`genetest.statistics.models.logistic.StatsLogistic` for more
information about the class.


Repeated measurements
^^^^^^^^^^^^^^^^^^^^^^

The repeated measurements analysis requires the following arguments and
options.

.. table::
    :widths: 20 70 10

    +---------------------------+--------------------------------------+----------+
    | Argument                  | Description                          | Required |
    +===========================+======================================+==========+
    | ``formula``               | The formula describing the analysis  | Yes      |
    |                           | to be performed. Note that the       |          |
    |                           | formula is similar to the one used in|          |
    |                           | R. The names of the variables need to|          |
    |                           | be the same as the columns in the    |          |
    |                           | phenotype file. The keyword ``SNPs`` |          |
    |                           | is used to perform a GWAS.           |          |
    +---------------------------+--------------------------------------+----------+
    | ``optimize``              | Should an optimization be performed  |          |
    |                           | by using a two-step approach by      |          |
    |                           | fitting one LMM in the first step    |          |
    |                           | without the genetic component and, in|          |
    |                           | the second step, fitting a simple    |          |
    |                           | regression model, for each SNP at a  |          |
    |                           | time. Then, if the p-value is lower  |          |
    |                           | than a user defined threshold, a     |          |
    |                           | complete LMM is fitted for this      |          |
    |                           | marker. Note that this optimization  |          |
    |                           | is invalid when using an             |          |
    |                           | genetic/environment interaction.     |          |
    |                           | [Default: True]                      |          |
    +---------------------------+--------------------------------------+----------+
    | ``p_threshold``           | The p-value threshold used for the   |          |
    |                           | MixedLM optimization (see above).    |          |
    |                           | [Default: 1e-4]                      |          |
    +---------------------------+--------------------------------------+----------+

Below is an example of a ``model`` section of the *YAML* configuration file
for a repeated measurements analysis of the phenotype *Pheno* over the
variables *SNPs* (meaning a GWAS), *Age*, *Sex* and *Visit* using the sample
IDs (*SampleID*) as the grouping variable.

.. code-block:: yaml

    model:
        test: mixedlm
        formula: "[outcome=Pheno, groups=SampleID] ~ SNPs + Age + factor(Sex) + factor(Visit)"
        options:
            optimize: Yes

See :py:class:`genetest.statistics.models.mixedlm.StatsMixedLM` for more
information about the class.


Survival analysis
^^^^^^^^^^^^^^^^^^

The Cox proportional hazard regression only requires the formula describing
the model.

.. table::
    :widths: 20 70 10

    +---------------------------+--------------------------------------+----------+
    | Argument                  | Description                          | Required |
    +===========================+======================================+==========+
    | ``formula``               | The formula describing the analysis  | Yes      |
    |                           | to be performed. Note that the       |          |
    |                           | formula is similar to the one used in|          |
    |                           | R. The names of the variables need to|          |
    |                           | be the same as the columns in the    |          |
    |                           | phenotype file. The keyword ``SNPs`` |          |
    |                           | is used to perform a GWAS.           |          |
    +---------------------------+--------------------------------------+----------+

Below is an example of a ``model`` section of the *YAML* configuration file
for a survival analysis (Cox proportional hazard regression) of the event
*Event* and time to event *TTE* over the variables *SNPs* (meaning a GWAS),
*Age* and *Sex*.

.. code-block:: yaml

    model:
        test: coxph
        formula: "[tte=TTE, event=Event] ~ SNPs + Age + factor(Sex)"

See :py:class:`genetest.statistics.models.survival.StatsCoxPH` for more
information about the class.


Execution
----------

Assuming the name of the configuration file ``analysis.yaml``, and that the
list of variant to extract for the analysis is in ``variants_to_extract.txt``
(on variant ID per line), the following command will launch the analysis using
6 CPUs. The resulting files will have the prefix ``results``.

Note that the ``--extract`` option should be used to extract only the variants
that pass quality control. Since genotypes file might be really big, extracting
only the variants suited for analysis will dramatically decrease the execution
time.


.. code-block:: bash

    genetest \
        --configuration analysis.yaml \
        --extract variants_to_extract.txt \
        --nb-cpus 6 \
        --output results


Output files
^^^^^^^^^^^^^

Using the previous command, three files will be generated (with the ``results``
prefix).

.. table::
    :widths: 20 80

    +-----------------------------+-----------------------------------------------+
    | File name                   | Description                                   |
    +=============================+===============================================+
    | ``results.log``             | File containing the LOG of the analysis.      |
    +-----------------------------+-----------------------------------------------+
    | ``results.txt``             | File containing the results of the analysis.  |
    |                             | The file is tab-separated and contain summary |
    |                             | information about each variant, along with    |
    |                             | the statistics specific to the statistical    |
    |                             | model.                                        |
    +-----------------------------+-----------------------------------------------+
    | ``results_failed_snps.txt`` | File containing the list of variants that     |
    |                             | failed the analysis. Failure can be           |
    |                             | attributed to low minor allele frequency or   |
    |                             | convergence issues, for example. A small      |
    |                             | description is added to describe the failure. |
    +-----------------------------+-----------------------------------------------+

