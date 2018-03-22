
Installation
=============

There are two ways you can use :py:mod:`genetest`: a manual installation or via
*Docker*.


Manual installation
--------------------

The :py:mod:`genetest` module requires a Python version 3.4 or earlier. We
recommend using a Python virtual environment specifically for
:py:mod:`genetest`.


Python virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Installing the modules
^^^^^^^^^^^^^^^^^^^^^^^

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
    more recent), you will required the :py:mod:`zstandard` module.

    .. code-block:: bash

        pip install zstandard


Testing the installation
^^^^^^^^^^^^^^^^^^^^^^^^^

To test the installation, you can perform the following commands.

.. code-block:: bash

    python -m genetest.tests
    python -m geneparse.tests
    python -m pyplink.tests
    python -m pybgen.tests


Docker
------

If, for some reason, a recent version of Python 3 is not available, we provide
a docker file so that you can easily create an environment in which to use
:py:mod:`genetest`.


Container creation
^^^^^^^^^^^^^^^^^^^

Save this docker file into a directory.

.. code-block:: docker

    # We use python 3.6 as a base image
    FROM python:3.6

    # Installing the dependencies
    RUN pip install -U pip
    RUN pip install -U setuptools
    RUN pip install -U Cython
    RUN pip install -U numpy
    RUN pip install -U pandas
    RUN pip install -U zstandard
    RUN pip install -U geneparse
    RUN pip install -U grako
    RUN pip install -U genetest

Make sure the file is named ``Dockerfile`` and execute the following command.

.. code-block:: bash

    sudo docker build --tag genetest .


Testing the installation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    sudo docker run --rm genetest python -m genetest.tests
    sudo docker run --rm genetest python -m geneparse.tests
    sudo docker run --rm genetest python -m pyplink.tests
    sudo docker run --rm genetest python -m pybgen.tests
