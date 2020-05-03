=======================================
Π-nets: Deep Polynomial Neural Networks
=======================================

Chainer implementation of the CVPR'20 paper "**Π-nets: Deep Polynomial Neural Networks**".

Specifically, we include the code for the fully-connected experiment of product of polynomials.



Browsing the folders
====================
The folder structure is the following:

*    ``gen_models``: The folder for the generator models; this is the primary folder you should modify.

*    ``dis_models``: The folder for the discriminator models; do not modify the optional arguments, unless you know what you are doing.

*    ``updater``: The folder that contains the core code for the updater, i.e. the chunk of code that runs in every iteration to update the modules.

*    ``jobs``: It contains a) the yml with the hyper-parameter setting, b) the main command to load the models and run the training (train_mn*.py).

*    ``source``: It contains auxiliary code, you should probably not modify any of that code.

*    ``evaluations``: It contains the code for validation (either during training or offline).

*    ``datasets``: Each file in the folder describes a parser for a dataset; normally you should not modify it for provided dataset(s).


Train the network
=================

To train the network, you can execute the following command::

   python jobs/product_fc/train_mn.py --config jobs/product_fc/facepointcloud1m_recursive_gen_prod_linear_proddis.yml --label my_experiment

The yml file describes the modules/dataset to train on. The hyper-parameters are included
in the yml, no need to hardcode them in the files. Probably, you should modify the path of 
the dataset (in the yml, find the 'dataset:' part) to match the one in your local pc.




Requirements
============

The code requires the chainer library.

Tested on a Linux machine with:

* chainer=4.0.0, chainercv=0.9.0,

* chainer=5.2.0, chainercv=0.12.0.

* chainer=6.1.0, chainercv=0.13.1.


The code is highly influenced by [1]_.

Apart from Chainer, the code depends on Pyaml [2]_. 


References
==========

.. [1] https://github.com/grigorisg9gr/rocgan/

.. [2] https://pypi.org/project/pyaml/

