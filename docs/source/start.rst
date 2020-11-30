.. _getting_started:

Getting Started
===============

1. Clone the repository
-----------------------
Start with cloning the repository to your local machine using:
 ``git clone git@github.com:zabaras/deep-turbulence.git``

2. Downloading Data
-------------------
To download both the training and testing data for both of the numerical examples, visit the following Zenodo repository.
`https://doi.org/10.5281/zenodo.4298896 <https://doi.org/10.5281/zenodo.4298896>`_

Extract the data using ``tar –xvzf data-set-name.tar.gz``, and move the training and testing data the desired directory.

The default directories in the config are:

- ``/deep-turbulence/cylinder-training/``
- ``/deep-turbulence/cylinder-testing/``
- ``/deep-turbulence/step-training/``
- ``/deep-turbulence/step-testing/``

But custom paths can be easily adjusted in the configuration file.

3. Set-up Conda Python Environment
----------------------------------
For your convenience, a requirements.txt file is provided in ``/tmglow/`` which will allow the easy creation of a conda environment that contains
the required packages used. 

``conda create --name <env_name> --file requirements.txt``

4. Start Training
-----------------
To start training the model simply run :doc:`tmglow/main`. 
There are many customizable parameters you can change in :doc:`tmglow/args` which can be view with ``python main.py --help``. 
Both numerical examples have pre-coded
configurations.

For example:

- Train the model for the backward-step example: ``python main.py --exp-type backward-step``
- Train the model for the backward-step example: ``python main.py --exp-type cylinder-array``
- Use a custom training directory: ``python main.py --training_data_dir <custom directory>``
- Change number of epochs: ``python main.py --epochs <# epochs>``

.. warning::
    Memory constraints on the GPU will likely be of concern. Parallel training is supported but only on a single 
    node (not multiple CPU) which can be controlled through ``--n_gpu`` and ``--parallel`` options. 
    Support is not provided for debugging GPU memory or parallel issues.

4. Running Pre-trained Models
-----------------------------
Alternatively, you can skip the training and simply run a pre-trained models.
These can be found in the ``example`` folder where there are several scripts that demonstrate how to load and test a model.