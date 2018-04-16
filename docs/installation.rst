
Installation
=============

The :py:mod:`genetest` module requires a Python version 3.4 or earlier. We
recommend using a Python virtual environment specifically for
:py:mod:`genetest`.


Python virtual environment
---------------------------

The following command will create a virtual environment for Python named
``genetest_venv`` in your *home* directory. Note that in order to use the
environment, it requires to be activated.

.. code-block:: bash

    # Creating the environment
    python3 -m venv $HOME/genetest_venv

    # Activating the environment
    source $HOME/genetest_venv/bin/activate


.. note::

    The text ``(gentest_venv)`` will be added at the beginning of the terminal
    *prompt*.

We recommend upgrading or installing some dependencies first.

.. code-block:: bash

    pip install -U pip
    pip install -U setuptools
    pip install -U Cython
    pip install -U numpy


Installing the modules
-----------------------

To install :py:mod:`genetest` module and all its required dependencies, use the
following command (making sure the virtual environment is activated).

.. code-block:: bash

    pip install genetest

The installation process should at least install the following dependencies.

- :py:mod:`geneparse`: parses multiple genotype file formats.
- :py:mod:`pybgen`: parses *BGEN* files (used by :py:mod:`geneparse`).
- :py:mod:`pyplink`: parses binary *Plink* files (used by :py:mod:`geneparse`).
- :py:mod:`cyvcf2`: parses *VCF* files (used by :py:mod:`geneparse`).

.. note::

    If you intend to parse *BGEN* files with the 1.3 format specifications (or
    more recent), you will require the :py:mod:`zstandard` module.

    .. code-block:: bash

        pip install zstandard


Testing the installation
-------------------------
To test the installation, you can perform the following commands.

.. code-block:: bash

    python -m genetest.tests
    python -m geneparse.tests
    python -m pyplink.tests
    python -m pybgen.tests
